from dataclasses import dataclass


@dataclass(slots=True)
class FullState:
    px: float
    py: float
    vx: float
    vy: float
    radius: float
    gx: float
    gy: float
    preferred_speed: float
    theta: float


@dataclass(slots=True)
class ObservableState:
    px: float
    py: float
    vx: float
    vy: float
    radius: float
