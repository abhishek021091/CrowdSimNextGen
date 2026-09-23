"""RangeImageBuilder: converts a raw laser scan into a binary occupancy
range image, robot-centric, for use by
`navcore.policies.crowdnav_pp.range_image_encoder.RangeImageEncoder`.

This is the single canonical preprocessing path for the obstacle branch --
see `ObstacleDetector`'s module docstring for the sensing/detection side.
This module does exactly one thing (`ObstacleScan` -> range image tensor)
and contains no torch dependency and no network code, matching the
project's existing split between sensor-encoding utilities (this,
`obstacle_detector.scan_to_features`) and the policy layer that consumes
them.

Binary semantics:
    1.0 = traversable (free). 0.0 = blocked. There is no separate
    "unknown" channel -- an unseen cell beyond the first hit along a ray
    is treated identically to a directly-observed blocked cell, and
    everything strictly before the first hit is free. This mirrors how
    a real occupancy grid built from a single instantaneous scan behaves:
    the sensor has no way to distinguish "unknown" from "not yet reached
    by this ray" within one tick, so this module does not pretend to
    encode that distinction either.

Maximum-range handling (the sensing square's edge is always a hit):
    `ObstacleDetector.sense()` already sets `distances[i] = max_range`
    for any ray that hit nothing within range (see its own docstring).
    That is exactly the semantics this module needs for "the sensing
    square's boundary blocks": a ray that reaches the edge of the
    represented area without hitting anything is treated as blocked
    right at that edge, not as infinitely free space. No extra logic is
    required here beyond binning `scan.distances` as given -- this
    builder assumes the scan was produced by an `ObstacleDetector`
    configured with the same `max_range` as this builder's `config`.

Coordinate system (robot-centric, never rotated into world coordinates):
    - Row 0 is the far edge of the sensing square (distance = max_range).
    - The last row is the robot's own position (distance = 0).
    - Column 0 is the left-most ray; the last column is the right-most.
      `ObstacleDetector`'s ray fan is ordered by *increasing* angle
      (most negative offset -- i.e. clockwise/rightmost -- first; see
      its own `_build_ray_offsets`), and increasing angle sweeps
      counter-clockwise, i.e. left, under the standard math convention
      this project's `Vector2.rotate` also uses. This builder therefore
      reverses ray order when laying out columns, so column 0 ends up
      left-most as specified, without changing `ObstacleDetector`'s own
      ray ordering (other consumers of `ObstacleScan` still see rays in
      their original, increasing-angle order).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from navcore.entities.components.sensors.obstacle_detector import ObstacleScan


@dataclass(slots=True, frozen=True)
class RangeImageBuilderConfig:
    """Configuration for `RangeImageBuilder`.

    Attributes:
        num_rays: Number of laser rays in the scan (image width, W).
            Must match the `ObstacleScan` this builder consumes --
            i.e. the `num_rays` of the `ObstacleDetectorConfig` used to
            produce that scan.
        num_range_bins: Number of discrete range buckets (image height,
            H). Larger values give finer radial resolution at the cost
            of a taller image for `RangeImageEncoder` to process.
        max_range: The sensing radius, in meters, represented by the
            image's far edge (row 0). Must match the `max_range` of the
            `ObstacleDetectorConfig` used to produce the scans this
            builder consumes -- see module docstring.
    """

    num_rays: int = 180
    num_range_bins: int = 128
    max_range: float = 5.0

    def __post_init__(self) -> None:
        if self.num_rays <= 0:
            raise ValueError(f"num_rays must be positive, got {self.num_rays!r}.")
        if self.num_range_bins <= 0:
            raise ValueError(
                f"num_range_bins must be positive, got {self.num_range_bins!r}."
            )
        if self.max_range <= 0.0:
            raise ValueError(f"max_range must be positive, got {self.max_range!r}.")


class RangeImageBuilder:
    """Builds a binary, robot-centric range image from one `ObstacleScan`.

    Stateless across calls (same rationale as `ObstacleDetector`) --
    holds only its own configuration.

    Attributes:
        config: This builder's configuration.
    """

    def __init__(self, config: RangeImageBuilderConfig | None = None) -> None:
        self.config = config or RangeImageBuilderConfig()

    def build(self, scan: ObstacleScan) -> np.ndarray:
        """Convert `scan` into a `[1, H, W]` binary range image.

        Args:
            scan: One tick's ray-casting result. `scan.distances` must
                be populated (see `ObstacleDetector.sense`) and its ray
                count must match `config.num_rays`.

        Returns:
            A `(1, num_range_bins, num_rays)` float32 array. `1.0` means
            free/traversable, `0.0` means blocked. Row 0 is the far edge
            of the sensing square (`max_range`); the last row is the
            robot's own position. Column 0 is the left-most ray, the
            last column the right-most -- see module docstring.

        Raises:
            ValueError: If `scan.distances`'s length doesn't match
                `config.num_rays`.
        """
        distances = scan.distances
        if distances.shape[0] != self.config.num_rays:
            raise ValueError(
                f"scan has {distances.shape[0]} rays, but this builder is "
                f"configured for num_rays={self.config.num_rays}."
            )

        H, W = self.config.num_range_bins, self.config.num_rays
        clipped = np.clip(distances, 0.0, self.config.max_range)

        # Index of the hit bin, measured outward from the robot (0 = right
        # at the robot, H-1 = the far edge) -- a real hit at max_range, or
        # the sensing-square boundary treated as a hit (see module
        # docstring), always lands in the last bin, so every ray
        # terminates by construction.
        bin_from_robot = np.minimum(
            (clipped / self.config.max_range * H).astype(np.int64), H - 1
        )

        row_index = np.arange(H)[:, None]  # [H, 1]; 0 = far edge, H-1 = robot
        # Free (1.0) strictly before the hit bin; the hit bin itself and
        # everything beyond it (toward the far edge) stays blocked (0.0)
        # -- "everything after the first hit becomes zero."
        is_free = row_index >= (H - bin_from_robot[None, :])

        image = np.zeros((H, W), dtype=np.float32)
        image[is_free] = 1.0

        # Reverse column order: ObstacleDetector's ray fan increases in
        # angle (rightmost first, see module docstring); this builder's
        # column 0 must be left-most.
        image = image[:, ::-1]

        return image[None, :, :].astype(np.float32)
