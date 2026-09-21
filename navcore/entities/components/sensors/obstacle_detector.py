# navcore/sensor/obstacle_detector.py
"""Ray-casting obstacle sensor, including environment boundary detection.

Casts a fixed ray fan from the robot's position out to a max range and
reports, per ray, whether it hit something and the hit point's position
relative to the robot. Output is a fixed-length array set regardless of how
many rays actually hit something, so it drops straight into a neural-network
observation the same way ObservationEncoder pads neighbor slots.

Obstacles and the environment boundary are tested together per ray (same
ray, same nearest-hit logic) but reported with a `hit_type` so a downstream
network can tell "ran into a static obstacle" apart from "about to leave
the environment" -- the two need different reward/avoidance treatment even
though they're the same kind of ray-intersection event.

Ground-truth geometry only -- this reads geometries directly (e.g.
env.obstacles, env.boundary), never a planner's ObservableState. Matches the
project's existing split between ground-truth-only consumers
(collision_detector/) and planner-visible state (see overview.md:
"Ground-truth state is reserved for collision detection only").

OPEN DESIGN QUESTIONS (unresolved -- confirm before wiring into
ObservationEncoder/CrowdSimEnv):
    1. Ray angles are generated in the WORLD frame, evenly spaced over
       `fov_radians`, centered on a `heading` argument that defaults to
       0.0. I don't have visibility into the actual Robot/Pose classes in
       this repo, so I don't know whether the robot carries a meaningful
       orientation (VELOCITY action mode suggests a holonomic robot with
       no heading, in which case a fixed world-frame ray fan is probably
       right) or whether sensor rays should be robot-heading-relative.
    2. `boundary` is accepted as a single Shapely geometry (a Polygon's
       exterior ring, or a LineString/MultiLineString of walls) -- I don't
       know how the environment boundary is actually represented elsewhere
       in navcore (EnvironmentBuilder is the likely owner). If it's stored
       as a filled Polygon rather than its boundary line, pass
       `polygon.exterior` (or `polygon.boundary`) in, not the polygon
       itself, or every ray originating inside it will register a
       same-point "hit" at distance 0.
    Everything here is written against plain floats/geometries, not the
    Robot/Pose/Environment types, so swapping in the real attributes at the
    call site shouldn't require changing this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from shapely.geometry.base import BaseGeometry
from shapely.geometry import LineString


class HitType(IntEnum):
    """What kind of ground-truth geometry a ray hit, if anything."""

    NONE = 0
    OBSTACLE = 1


@dataclass(slots=True, frozen=True)
class ObstacleDetectorConfig:
    """Ray-casting sensor parameters.

    Attributes:
        num_rays: Number of evenly spaced rays per scan.
        max_range: Ray length in meters. A ray that hits nothing reports
            this as its distance and is marked invalid in `hit_mask`.
        fov_radians: Angular spread of the ray fan, centered on `heading`.
            2*pi gives a full 360-degree ring (e.g. a spinning lidar);
            anything smaller gives a forward-facing cone.
    """

    num_rays: int = 60
    max_range: float = 5.0
    fov_radians: float = 2.0 * math.pi


@dataclass(slots=True, frozen=True)
class ObstacleScan:
    """One ray-casting sensor reading, robot-relative.

    Every array is shape (num_rays,), (num_rays, 2), or (num_rays,) int --
    fixed width no matter how many rays hit something. Rows for a non-hit
    ray are zeroed in `relative_positions`, `distances` is set to
    `config.max_range` there, and `hit_type` is `HitType.NONE` -- so
    `hit_mask` is the only thing a consumer needs to check before trusting
    a row, and `hit_type` is only meaningful where `hit_mask` is True.

    Attributes:
        hit_mask: (num_rays,) bool -- True where that ray intersected
            *something* (obstacle or boundary) within max_range.
        hit_type: (num_rays,) int8 -- HitType.NONE / OBSTACLE / BOUNDARY
            per ray, letting a consumer distinguish the two without a
            second pass. Always NONE where hit_mask is False.
        distances: (num_rays,) float32 -- distance to the nearest
            intersection (obstacle or boundary, whichever is closer), or
            max_range where hit_mask is False.
        relative_positions: (num_rays, 2) float32 -- hit point minus robot
            position, as (dx, dy) in world-axis-aligned coordinates
            (*not* rotated into a robot-heading frame -- rotate at the
            call site if the consumer needs that). Zero where hit_mask
            is False.
        ray_angles: (num_rays,) float32 -- world-frame angle (radians)
            each ray was cast at, in the same order as the other arrays,
            so absolute bearing is recoverable without re-deriving the fan.
    """

    hit_mask: np.ndarray
    hit_type: np.ndarray
    # distances: np.ndarray
    relative_positions: np.ndarray
    # ray_angles: np.ndarray


#: Per-ray feature layout produced by `scan_to_features`:
#: [hit_mask, distance_norm, dx_norm, dy_norm, sin(ray_angle), cos(ray_angle)].
RAY_FEATURE_DIM = 3


def scan_to_features(scan: ObstacleScan, max_range: float) -> np.ndarray:
    """Convert one `ObstacleScan` into a `(num_rays, RAY_FEATURE_DIM)` array.

    Lives here, not in `policies.crowdnav_pp.obstacle_encoder` (where
    this used to live) -- this is a pure sensor-encoding concern with
    no torch dependency, and keeping it in the policy package would
    force anything that wants ray features (e.g. `ObservationEncoder`)
    to import from the policy layer, backwards from how this project
    layers environment/observation code under policies.

    Returns:
        `(num_rays, RAY_FEATURE_DIM)` float32 array:
        `[hit_mask, distance / max_range, dx / max_range, dy / max_range,
        sin(theta), cos(theta)]` per ray.

        NOTE: theta is recomputed as a uniform `2*pi*i/num_rays` sweep
        starting at angle 0 -- it does NOT read `scan.ray_angles`. Only
        correct for a scan cast with `heading=0.0` and a full `2*pi`
        fov (what `ObservationEncoder` always uses -- see its own
        docstring). Not fixed here; flagged as a trap for whoever wires
        a heading-relative or partial-fov scan in later.
    """
    num_rays = scan.hit_mask.shape[0]
    features = np.zeros((num_rays, RAY_FEATURE_DIM), dtype=np.float32)

    features[:, 0] = scan.hit_mask.astype(np.float32)
    # features[:, 1] = scan.distances / max_range
    features[:, 1] = scan.relative_positions[:, 0] / max_range
    features[:, 2] = scan.relative_positions[:, 1] / max_range

    indices = np.arange(num_rays, dtype=np.float32)
    theta = 2.0 * np.pi * indices / num_rays
    # features[:, 3] = np.sin(theta)
    # features[:, 4] = np.cos(theta)

    return features


class ObstacleDetector:
    """Casts a ray fan from a point against ground-truth obstacles and boundary.

    Stateless across calls (obstacle/boundary layout is fixed per-episode,
    and this detector has nothing that needs resetting at an episode
    boundary) -- geometry is passed in per `sense()` call rather than held
    internally.
    """

    def __init__(self, config: ObstacleDetectorConfig | None = None) -> None:
        self.config = config or ObstacleDetectorConfig()
        self._ray_offsets = self._build_ray_offsets()

    def _build_ray_offsets(self) -> np.ndarray:
        """Angle offsets from heading=0, evenly spaced over fov_radians.

        A full-circle fov drops the final ray (endpoint=False) so angle 0
        and angle 2*pi don't duplicate the same direction. A partial fov
        keeps both endpoints so the edges of the cone are actually covered.
        """
        n = self.config.num_rays
        fov = self.config.fov_radians
        if math.isclose(fov, 2.0 * math.pi):
            return np.linspace(0.0, fov, n, endpoint=False, dtype=np.float32)
        return np.linspace(-fov / 2.0, fov / 2.0, n, endpoint=True, dtype=np.float32)

    def sense(
        self,
        robot_x: float,
        robot_y: float,
        obstacles: list[BaseGeometry],
        # boundary: BaseGeometry | None = None,
        heading: float = 0.0,
    ) -> ObstacleScan:
        """Cast this scan's ray fan from (robot_x, robot_y).

        Each ray is tested against every entry in `obstacles` and, if
        given, `boundary`; the nearest intersection overall (obstacle or
        boundary) wins that ray and sets its `hit_type`.

        Args:
            robot_x, robot_y: Robot position, world frame.
            obstacles: Ground-truth obstacle geometries (e.g. env.obstacles).
                Each is tested as-is against every ray -- pass whatever
                Shapely geometry type env.obstacles already holds
                (Polygon, its boundary, etc.) elsewhere in the codebase.
            boundary: Ground-truth environment boundary geometry (walls /
                free-space perimeter), or None to skip boundary detection
                entirely (e.g. an unbounded environment). Must be a line
                geometry (a Polygon's `.exterior`/`.boundary`, or a
                LineString/MultiLineString of walls) -- passing a filled
                Polygon means every ray "hits" it at distance 0 from
                inside. See module docstring's open question #2.
            heading: World-frame angle (radians) the ray fan is centered
                on. See the module docstring's open question #1 before
                wiring a real robot heading through here.

        Returns:
            An ObstacleScan with num_rays entries, ordered to match
            self._ray_offsets, so the ordering is reproducible across
            calls for a fixed config.
        """
        n = self.config.num_rays
        max_range = self.config.max_range

        hit_mask = np.zeros(n, dtype=bool)
        hit_type = np.full(n, HitType.NONE, dtype=np.int8)
        distances = np.full(n, max_range, dtype=np.float32)
        relative_positions = np.zeros((n, 2), dtype=np.float32)
        ray_angles = (heading + self._ray_offsets).astype(np.float32)

        origin = (robot_x, robot_y)
        for i, angle in enumerate(ray_angles):
            end_x = robot_x + max_range * math.cos(angle)
            end_y = robot_y + max_range * math.sin(angle)
            ray = LineString([origin, (end_x, end_y)])

            nearest_distance = self._nearest_hit_distance(ray, obstacles, origin)
            nearest_type = (
                HitType.OBSTACLE if nearest_distance is not None else HitType.NONE
            )

            # if boundary is not None:
            #     boundary_distance = _ray_geometry_distance(ray, boundary, origin)
            #     if boundary_distance is not None and (
            #         nearest_distance is None or boundary_distance < nearest_distance
            #     ):
            #         nearest_distance = boundary_distance
            #         nearest_type = HitType.BOUNDARY

            if nearest_distance is not None and nearest_distance <= max_range:
                hit_mask[i] = True
                hit_type[i] = nearest_type
                distances[i] = nearest_distance
                relative_positions[i, 0] = nearest_distance * math.cos(angle)
                relative_positions[i, 1] = nearest_distance * math.sin(angle)

        return ObstacleScan(
            hit_mask=hit_mask,
            hit_type=hit_type,
            relative_positions=relative_positions,
        )

    @staticmethod
    def _nearest_hit_distance(
        ray: LineString,
        obstacles: list[BaseGeometry],
        origin: tuple[float, float],
    ) -> float | None:
        """Nearest intersection distance between `ray` and any of `obstacles`."""
        nearest: float | None = None
        for obstacle in obstacles:
            candidate = _ray_geometry_distance(ray, obstacle, origin)
            if candidate is None:
                continue
            if nearest is None or candidate < nearest:
                nearest = candidate
        return nearest


def _ray_geometry_distance(
    ray: LineString, geometry: BaseGeometry, origin: tuple[float, float]
) -> float | None:
    """Distance from `origin` to `ray`'s nearest intersection with `geometry`."""
    if not ray.intersects(geometry):
        return None
    return _nearest_point_distance(ray.intersection(geometry), origin)


def _nearest_point_distance(
    intersection: BaseGeometry, origin: tuple[float, float]
) -> float | None:
    """Distance from `origin` to the nearest point in a ray intersection.

    `ray.intersection(geometry)` can come back as a Point, MultiPoint,
    LineString, or GeometryCollection depending on whether the ray clips a
    corner, crosses an edge, or passes through the interior -- Shapely does
    not normalize this to one geometry type, so every constituent
    point/vertex is checked rather than assuming a single Point.
    """
    if intersection.is_empty:
        return None

    geoms = (
        list(intersection.geoms) if hasattr(intersection, "geoms") else [intersection]
    )

    best: float | None = None
    for geom in geoms:
        if geom.geom_type == "Point":
            coords = [(geom.x, geom.y)]
        else:
            coords = list(geom.coords) if hasattr(geom, "coords") else []
        for x, y in coords:
            d = math.hypot(x - origin[0], y - origin[1])
            if best is None or d < best:
                best = d
    return best
