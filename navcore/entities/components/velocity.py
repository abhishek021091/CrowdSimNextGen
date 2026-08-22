from dataclasses import dataclass


@dataclass(slots=True)
class Velocity:
    vx: float
    vy: float
