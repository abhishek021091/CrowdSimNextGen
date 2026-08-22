from dataclasses import dataclass


@dataclass(slots=True)
class Pose:
    px: float
    py: float
    theta: float
