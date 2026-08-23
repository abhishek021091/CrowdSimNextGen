from dataclasses import dataclass

from navcore.entities.components.goal import Goal
from navcore.entities.components.pose import Pose
from navcore.entities.components.velocity import Velocity


@dataclass(slots=True)
class FullState:
    pose: Pose
    goal: Goal
    velocity: Velocity
    radius: float
    preferred_speed: float


@dataclass(slots=True)
class ObservableState:
    pose: Pose
    velocity: Velocity
    radius: float
