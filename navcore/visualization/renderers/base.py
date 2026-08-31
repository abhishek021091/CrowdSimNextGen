"""Shared contract for every renderer.

Each renderer owns exactly one visual concern (robot, crowd, obstacles,
sensor, planning overlays, ...). They are constructed once against an
Axes and a config, create their artists lazily on first ``update``, and
mutate those artists in place afterwards -- no patches/lines are ever
recreated per frame, which is what keeps 500-1000 agent scenes fast.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..config import ColorScheme, LayerToggles
from ..state import SceneSnapshot


class Renderer(ABC):
    """Base class for a single-responsibility scene layer."""

    def __init__(self, ax: Any, scheme: ColorScheme) -> None:
        self.ax = ax
        self.scheme = scheme

    def set_scheme(self, scheme: ColorScheme) -> None:
        """Called on light/dark mode switch. Default: store and let the
        next ``update`` call re-apply colors. Override for artists whose
        color can be patched immediately without a full update."""
        self.scheme = scheme

    @abstractmethod
    def update(self, scene: SceneSnapshot, toggles: LayerToggles) -> None:
        """Sync this layer's artists to the given (possibly interpolated)
        scene snapshot. Must be idempotent and cheap to call every frame."""

    def clear(self) -> None:
        """Remove all artists owned by this renderer (e.g. on reset)."""

    def bounds(self, scene: SceneSnapshot) -> tuple[list[float], list[float]]:
        """Optional: contribute x/y points this layer wants included when
        auto-fitting the camera (e.g. obstacle vertices, goals)."""
        return [], []
