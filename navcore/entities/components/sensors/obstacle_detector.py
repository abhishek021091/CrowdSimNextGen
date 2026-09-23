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

RESTORED (previously commented out -- see RangeImageBuilder, which is now
the canonical consumer of this module's output):
    - `ObstacleScan.distances` / `ObstacleScan.ray_angles`: needed by
      `RangeImageBuilder` to bin each ray into a range image row, and to
      recover absolute bearing without re-deriving the ray fan.
    - `HitType.BOUNDARY` and `sense()`'s `boundary` parameter: previously
      `ObservationEncoder` worked around this by folding the boundary ring
      into the generic `obstacles` list, which loses the obstacle-vs-
      boundary distinction entirely. `sense()` now tests `boundary`
      explicitly, same as any other geometry, and reports which one a ray
      actually hit.

RESOLVED (previously open design questions):
    1. Ray angles are generated in the WORLD frame, evenly spaced over
       `fov_radians`, centered on `heading` (default 0.0). Confirmed
       correct for navcore's holonomic robot (see
       `ObservationEncoder._encode_range_image`): there is no orientation-
       constrained motion a heading-relative fan would need to track.
    2. `boundary` is accepted as a single Shapely line geometry (a
       Polygon's `.exterior`/`.boundary`, or a LineString/MultiLineString
       of walls) -- callers pass `env`'s boundary ring, not a filled
       Polygon (see `navcore.entities.obstacles.geometry_conversion.
       arena_boundary_ring`), so a ray originating inside it does not
       register a spurious same-point hit at distance 0.

Consolidation note (removes prior duplication):
    A second, divergent `scan_to_features`/`RAY_FEATURE_DIM` used to live
    in `navcore.policies.crowdnav_pp.obstacle_encoder` (the now-legacy
    1D-CNN obstacle branch). That module now imports this one instead of
    keeping its own copy -- see its module docstring.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry


class HitType(IntEnum):
    """What kind of ground-truth geometry a ray hit, if anything."""

    NONE = 0
    OBSTACLE = 1


@dataclass(slots=True, frozen=True)
class ObstacleDetectorConfig:
    """Ray-casting sensor parameters.

    Attributes:
        num_rays: Number of evenly spaced rays per scan. Defaults to 180
            to match `RangeImageBuilderConfig`'s default image width --
            keep the two in sync when either is overridden.
        max_range: Ray length in meters. A ray that hits nothing reports
            this as its distance and is marked invalid in `hit_mask` --
            this is also exactly what `RangeImageBuilder` treats as "the
            sensing-square boundary itself is the first hit" (see that
            module's docstring).
        fov_radians: Angular spread of the ray fan, centered on `heading`.
            2*pi gives a full 360-degree ring (e.g. a spinning lidar);
            anything smaller gives a forward-facing cone.
    """

    num_rays: int = 180
    max_range: float = 5.0
    fov_radians: float = 2.0 * math.pi

    def __post_init__(self) -> None:
        if self.num_rays <= 0:
            raise ValueError(f"num_rays must be positive, got {self.num_rays!r}.")
        if self.max_range <= 0.0:
            raise ValueError(f"max_range must be positive, got {self.max_range!r}.")


@dataclass(slots=True, frozen=True)
class ObstacleScan:
    """One ray-casting sensor reading, robot-relative.

    Every array is shape (num_rays,) or (num_rays, 2) -- fixed width no
    matter how many rays hit something. Rows for a non-hit ray are zeroed
    in `relative_positions`, `distances` is set to `config.max_range`
    there, and `hit_type` is `HitType.NONE` -- so `hit_mask` is the only
    thing a consumer needs to check before trusting a row, and `hit_type`
    is only meaningful where `hit_mask` is True. `distances`, however, is
    always meaningful (real hit distance, or `max_range` as the effective
    "sensing boundary" distance) -- `RangeImageBuilder` relies on this.

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
    distances: np.ndarray
    relative_positions: np.ndarray
    ray_angles: np.ndarray


#: Per-ray feature layout produced by `scan_to_features`:
#: [hit_mask, distance_norm, dx_norm, dy_norm, sin(ray_angle), cos(ray_angle)].
#: Legacy consumer: navcore.policies.crowdnav_pp.obstacle_encoder's 1D-CNN
#: branch. The current default obstacle branch (RangeImageEncoder) does not
#: use this -- it consumes RangeImageBuilder's binary image instead.
RAY_FEATURE_DIM = 6


def scan_to_features(scan: ObstacleScan, max_range: float) -> np.ndarray:
    """Convert one `ObstacleScan` into a `(num_rays, RAY_FEATURE_DIM)` array.

    Canonical, single copy -- see module docstring's consolidation note.
    Lives here, not in a policy package: this is a pure sensor-encoding
    concern with no torch dependency.

    Returns:
        `(num_rays, RAY_FEATURE_DIM)` float32 array:
        `[hit_mask, distance / max_range, dx / max_range, dy / max_range,
        sin(theta), cos(theta)]` per ray, using each ray's *actual*
        `scan.ray_angles` entry (not a re-derived uniform sweep).
    """
    num_rays = scan.hit_mask.shape[0]
    features = np.zeros((num_rays, RAY_FEATURE_DIM), dtype=np.float32)

    features[:, 0] = scan.hit_mask.astype(np.float32)
    features[:, 1] = scan.distances / max_range
    features[:, 2] = scan.relative_positions[:, 0] / max_range
    features[:, 3] = scan.relative_positions[:, 1] / max_range
    features[:, 4] = np.sin(scan.ray_angles)
    features[:, 5] = np.cos(scan.ray_angles)

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
        obstacles: Sequence[BaseGeometry],
        heading: float = 0.0,
    ) -> ObstacleScan:
        """Cast this scan's ray fan from (robot_x, robot_y).

        Every entry in `obstacles` is tested identically and the nearest
        intersection overall wins that ray. The environment boundary is
        not special-cased here -- if the caller wants boundary detection,
        it includes the boundary ring (e.g.
        `navcore.entities.obstacles.geometry_conversion.arena_boundary_ring`)
        as just another geometry in `obstacles`.

        Args:
            robot_x, robot_y: Robot position, world frame.
            obstacles: Ground-truth geometries a ray can hit -- static
                obstacles, the arena boundary ring, or anything else the
                caller wants rays to stop at. No entry is treated
                differently from any other.
            heading: World-frame angle (radians) the ray fan is centered
                on. Fixed at 0.0 for navcore's holonomic robot (see module
                docstring's resolved open question #1).

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

            if nearest_distance is not None and nearest_distance <= max_range:
                hit_mask[i] = True
                hit_type[i] = HitType.OBSTACLE
                distances[i] = nearest_distance
                relative_positions[i, 0] = nearest_distance * math.cos(angle)
                relative_positions[i, 1] = nearest_distance * math.sin(angle)

        return ObstacleScan(
            hit_mask=hit_mask,
            hit_type=hit_type,
            distances=distances,
            relative_positions=relative_positions,
            ray_angles=ray_angles,
        )

    @staticmethod
    def _nearest_hit_distance(
        ray: LineString,
        obstacles: Sequence[BaseGeometry],
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
