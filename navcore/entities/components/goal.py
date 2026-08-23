from dataclasses import dataclass


@dataclass(slots=True)
class Goal:
    gx: float
    gy: float
