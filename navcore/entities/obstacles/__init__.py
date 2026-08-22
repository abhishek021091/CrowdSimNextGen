"""Static obstacle domain model.

Exposes the generic ``Obstacle`` type plus the concrete obstacle kinds
supported today (``Wall``, ``Pillar``, ``Boundary``, ``BoundaryGate``,
``Table``). Each concrete kind composes an ``Obstacle`` via a
``to_obstacle()`` factory method rather than subclassing it, so new
obstacle kinds -- ``Shelf``, ``ChargingStation``, ``Dock``, ``Plant``,
``TrafficCone``, and so on -- can be added later as new sibling modules
without modifying any of the classes exported here.

Only static obstacles live in this package. Dynamic obstacles (moving
objects, robots, pedestrians) are explicitly out of scope and will be
handled by a separate module.
"""

from .boundary import Boundary, BoundaryGate
from .obstacle import Obstacle
from .pillar import Pillar
from .table import Table, TableShape
from .wall import Wall

__all__ = [
    "Boundary",
    "BoundaryGate",
    "Obstacle",
    "Pillar",
    "Table",
    "TableShape",
    "Wall",
]
