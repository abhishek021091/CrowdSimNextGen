"""NavCoreKinematicBackend: navcore's own in-process integrator.

The default, always-available ``ExecutionBackend``. Used by every
pedestrian unconditionally, and by the robot whenever no
physics-validated (Gazebo) or real-hardware backend is configured --
i.e. it's what every headless RL training run uses, since it has no
external process, no I/O, and no per-tick overhead beyond a handful of
float operations.

Deliberately simple: straight-line Euler integration at the commanded
velocity, no collision response, no dynamics. That is intentional --
avoidance is the ``Policy``'s job upstream of this backend; adding
physics here would duplicate what a Gazebo-backed implementation
exists to provide, for no benefit during fast headless training.
"""

from __future__ import annotations

from navcore.entities.components.pose import Pose
from navcore.entities.components.velocity import Velocity


class NavCoreKinematicBackend:
    """Integrates a commanded velocity directly into pose, every tick."""

    def __init__(self, pose: Pose) -> None:
        self._pose = pose
        self._velocity = Velocity(0.0, 0.0)
        self._commanded = Velocity(0.0, 0.0)

    def apply_command(self, velocity: Velocity) -> None:
        self._commanded = velocity

    def advance(self, dt: float) -> None:
        self._velocity = self._commanded
        self._pose = Pose(
            self._pose.px + self._velocity.vx * dt,
            self._pose.py + self._velocity.vy * dt,
            self._pose.theta,
        )

    def read_pose(self) -> Pose:
        return self._pose

    def read_velocity(self) -> Velocity:
        return self._velocity

    def reset(self, pose: Pose) -> None:
        self._pose = pose
        self._velocity = Velocity(0.0, 0.0)
        self._commanded = Velocity(0.0, 0.0)

    def close(self) -> None:
        """No resources to release for an in-process backend."""
        return None
