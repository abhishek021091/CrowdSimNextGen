"""Keyboard and mouse interaction wiring.

This module only *routes* input events to callbacks the ``Visualizer``
supplies -- it holds no simulation or rendering logic itself, which
keeps it trivially testable and swappable (e.g. for a Qt front-end).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import LayerToggles

KEY_TOGGLE_MAP: dict[str, str] = {
    "i": "ids",
    "v": "velocity",
    "g": "goals",
    "t": "trails",
    "s": "sensor",
    "p": "planning",
    "o": "orca",
    "l": "labels",
}


@dataclass(slots=True)
class InteractionCallbacks:
    on_pause_toggle: Callable[[], None] = lambda: None
    on_reset: Callable[[], None] = lambda: None
    on_fullscreen_toggle: Callable[[], None] = lambda: None
    on_pedestrian_click: Callable[[int | None], None] = lambda _pid: None
    on_toggle_changed: Callable[[str, bool], None] = lambda _name, _val: None


class InteractionManager:
    """Binds keyboard shortcuts and pedestrian-click picking.

    Shortcuts (also documented in ``Visualizer`` docstring):
        Space  pause/resume      R  reset        F  fullscreen
        T  trails   G  goals     I  ids          V  velocity
        S  sensor   P  planning  O  orca         L  labels
    """

    def __init__(
        self,
        fig: Any,
        ax: Any,
        toggles: LayerToggles,
        callbacks: InteractionCallbacks,
        pedestrian_position_lookup: Callable[[], dict[int, tuple[float, float, float]]],
    ) -> None:
        self.fig = fig
        self.ax = ax
        self.toggles = toggles
        self.callbacks = callbacks
        self._pedestrian_position_lookup = pedestrian_position_lookup

        self._cid_key = fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._cid_click = fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_key(self, event: Any) -> None:
        key = (event.key or "").lower()
        if key == " ":
            self.callbacks.on_pause_toggle()
        elif key == "r":
            self.callbacks.on_reset()
        elif key == "f":
            self.callbacks.on_fullscreen_toggle()
        elif key in KEY_TOGGLE_MAP:
            name = KEY_TOGGLE_MAP[key]
            new_val = self.toggles.toggle(name)
            self.callbacks.on_toggle_changed(name, new_val)
            self.fig.canvas.draw_idle()

    def _on_click(self, event: Any) -> None:
        if event.inaxes != self.ax or event.button != 3:
            # Right-click selects a pedestrian; left-click is reserved
            # for panning (see CameraController), so this stays a
            # separate mouse button rather than fighting for drag events.
            return
        if event.xdata is None or event.ydata is None:
            return
        positions = self._pedestrian_position_lookup()
        best_id, best_dist = None, float("inf")
        for pid, (x, y, radius) in positions.items():
            dist = math.hypot(event.xdata - x, event.ydata - y)
            if dist <= max(radius * 1.5, 0.3) and dist < best_dist:
                best_id, best_dist = pid, dist
        self.callbacks.on_pedestrian_click(best_id)
