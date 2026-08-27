"""ExecutionBackend: turns a commanded velocity into an actual resulting pose.

This is the seam between *deciding* what to do (``Policy``/``Mission``,
which never change) and *what actually happens physically* -- which
does change, from an in-process kinematic integrator during headless
RL training, to a physics-simulated robot in Gazebo for validation, to
a real robot on a real network for deployment. Swapping this one
object is the entire mechanism for measuring the sim-to-sim and
sim-to-real gap: the same Policy runs unmodified against every tier.

Only the robot uses a swappable backend today -- pedestrians always use
``NavCoreKinematicBackend`` (crowd dynamics are not something Gazebo or
real hardware need to be involved in). Nothing here assumes that will
always be true, but multi-agent Gazebo fleets are explicitly out of
scope for this first version.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from navcore.entities.components.pose import Pose
from navcore.entities.components.velocity import Velocity


@runtime_checkable
class ExecutionBackend(Protocol):
    """Owns one agent's actual physical (or physically-simulated) motion."""

    def apply_command(self, velocity: Velocity) -> None:
        """Record the velocity commanded for the upcoming ``advance()``.

        Implementations must not act on this immediately -- only once
        ``advance()`` is called, so every agent's command for a tick is
        captured before any agent's physics actually moves. See
        ``navcore.simulation.Simulation.step`` for why this ordering
        matters (it avoids an iteration-order-dependent first-mover
        bias across agents).
        """
        ...

    def advance(self, dt: float) -> None:
        """Advance this backend's physics by ``dt`` seconds.

        For a real-hardware backend, real time has already elapsed by
        however long the previous tick took; implementations should
        treat this as a synchronization point rather than literally
        forcing time to pass by exactly ``dt``.
        """
        ...

    def read_pose(self) -> Pose:
        """Return the pose resulting from the most recent ``advance()``."""
        ...

    def read_velocity(self) -> Velocity:
        """Return the actual resulting velocity.

        May differ from the last commanded velocity (saturation, slip,
        controller dynamics, ...). That discrepancy is itself a useful
        sim2real research signal -- implementations must not silently
        echo the commanded value.
        """
        ...

    def reset(self, pose: Pose) -> None:
        """Reset this agent to ``pose`` for a new episode.

        Raises:
            NotImplementedError: Implementations backed by real
                hardware cannot teleport a physical robot and should
                raise here rather than silently ignoring the request;
                callers running against a real backend must treat
                episodes as logical time windows, not state resets.
        """
        ...

    def close(self) -> None:
        """Release any resources held by this backend (connections, etc.)."""
        ...
