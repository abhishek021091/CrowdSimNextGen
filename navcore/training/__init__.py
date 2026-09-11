"""Training-only utilities.

Nothing in this package is part of a Gym observation.  In particular,
trajectory labels belong in a rollout/training pipeline, never in policy
inputs.
"""

from navcore.training.trajectory_targets import (
    FutureTrajectoryTargets,
    TrajectoryTargetRecorder,
)

__all__ = ["FutureTrajectoryTargets", "TrajectoryTargetRecorder"]
