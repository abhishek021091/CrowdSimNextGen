"""Viewport management: auto-fit, manual zoom/pan, and robot-following.

Kept independent of any renderer so it can be reused by other views
(e.g. a future mini-map) and so ``InteractionManager`` has one obvious
place to route mouse events.
"""

from __future__ import annotations

from typing import Any


class CameraController:
    def __init__(self, ax: Any, padding: float = 1.5) -> None:
        self.ax = ax
        self.padding = padding
        self.follow_robot_enabled = False
        self.follow_zoom_radius = 8.0
        self._dragging = False
        self._drag_start: tuple[float, float] | None = None
        self._xlim_start: tuple[float, float] | None = None
        self._ylim_start: tuple[float, float] | None = None

        self._cid_scroll = ax.figure.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._cid_press = ax.figure.canvas.mpl_connect("button_press_event", self._on_press)
        self._cid_release = ax.figure.canvas.mpl_connect("button_release_event", self._on_release)
        self._cid_motion = ax.figure.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def follow_robot(self, enabled: bool) -> None:
        self.follow_robot_enabled = enabled

    def fit_bounds(self, xs: list[float], ys: list[float]) -> None:
        if not xs or not ys:
            self.ax.set_xlim(-10, 10)
            self.ax.set_ylim(-10, 10)
            return
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xpad = max(self.padding, 0.1 * max(1.0, xmax - xmin))
        ypad = max(self.padding, 0.1 * max(1.0, ymax - ymin))
        self.ax.set_xlim(xmin - xpad, xmax + xpad)
        self.ax.set_ylim(ymin - ypad, ymax + ypad)

    def center_on(self, x: float, y: float) -> None:
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        half_w = (xmax - xmin) / 2
        half_h = (ymax - ymin) / 2
        self.ax.set_xlim(x - half_w, x + half_w)
        self.ax.set_ylim(y - half_h, y + half_h)

    def maybe_follow(self, robot_x: float | None, robot_y: float | None) -> None:
        if not self.follow_robot_enabled or robot_x is None:
            return
        r = self.follow_zoom_radius
        self.ax.set_xlim(robot_x - r, robot_x + r)
        self.ax.set_ylim(robot_y - r, robot_y + r)

    # -- mouse handlers -----------------------------------------------
    def _on_scroll(self, event: Any) -> None:
        if event.inaxes != self.ax:
            return
        scale = 0.9 if event.button == "up" else 1.1
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        xdata = event.xdata if event.xdata is not None else (xmin + xmax) / 2
        ydata = event.ydata if event.ydata is not None else (ymin + ymax) / 2
        new_w = (xmax - xmin) * scale
        new_h = (ymax - ymin) * scale
        fx = (xdata - xmin) / (xmax - xmin) if xmax != xmin else 0.5
        fy = (ydata - ymin) / (ymax - ymin) if ymax != ymin else 0.5
        self.ax.set_xlim(xdata - fx * new_w, xdata + (1 - fx) * new_w)
        self.ax.set_ylim(ydata - fy * new_h, ydata + (1 - fy) * new_h)
        self.follow_robot_enabled = False
        self.ax.figure.canvas.draw_idle()

    def _on_press(self, event: Any) -> None:
        if event.inaxes != self.ax or event.button != 1:
            return
        self._dragging = True
        self._drag_start = (event.xdata, event.ydata)
        self._xlim_start = self.ax.get_xlim()
        self._ylim_start = self.ax.get_ylim()

    def _on_release(self, event: Any) -> None:
        self._dragging = False
        self._drag_start = None

    def _on_motion(self, event: Any) -> None:
        if not self._dragging or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None or self._drag_start is None:
            return
        dx = event.xdata - self._drag_start[0]
        dy = event.ydata - self._drag_start[1]
        xmin, xmax = self._xlim_start
        ymin, ymax = self._ylim_start
        self.ax.set_xlim(xmin - dx, xmax - dx)
        self.ax.set_ylim(ymin - dy, ymax - dy)
        self.follow_robot_enabled = False
        self.ax.figure.canvas.draw_idle()
