from __future__ import annotations

from abc import ABC, abstractmethod

from navcore.entities.components.geometry.vector2 import Vector2


class Geometry(ABC):
    """Abstract base class for all local-frame mathematical shapes.

    A ``Geometry`` describes *only* the shape of an object in its own
    local coordinate frame -- it has no notion of where that shape sits
    or how it is oriented in the world. World placement is the job of a
    separate ``Pose`` type (position + rotation) defined elsewhere.

    Subclasses must remain pure math: no rendering, no collision
    detection, no physics, no simulation state.
    """

    @abstractmethod
    def area(self) -> float:
        """Return the area enclosed by the shape, in local units squared."""
        raise NotImplementedError

    @abstractmethod
    def centroid(self) -> Vector2:
        """Return the geometric centroid of the shape, in its local frame."""
        raise NotImplementedError
