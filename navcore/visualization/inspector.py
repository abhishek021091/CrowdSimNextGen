"""Click-to-inspect info box for a single selected pedestrian."""

from __future__ import annotations

import math
from typing import Any

from .state import AgentSnapshot, SceneSnapshot


class PedestrianInspector:
    def __init__(self, ax: Any, scheme: Any) -> None:
        self.ax = ax
        self.scheme = scheme
        self.selected_id: int | None = None
        self._box = ax.text(
            0.98, 0.98, "", transform=ax.transAxes, va="top", ha="right",
            fontsize=9, family="monospace", color=scheme.text_color,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=scheme.panel_face,
                      edgecolor=scheme.panel_edge, alpha=0.92),
            zorder=10,
        )
        self._box.set_visible(False)

    def select(self, ped_id: int | None) -> None:
        self.selected_id = ped_id
        if ped_id is None:
            self._box.set_visible(False)

    def set_scheme(self, scheme: Any) -> None:
        self.scheme = scheme
        self._box.set_color(scheme.text_color)
        self._box.set_bbox(dict(boxstyle="round,pad=0.4", facecolor=scheme.panel_face,
                                 edgecolor=scheme.panel_edge, alpha=0.92))

    def update(self, scene: SceneSnapshot) -> None:
        if self.selected_id is None:
            self._box.set_visible(False)
            return
        ped = scene.pedestrians.get(self.selected_id)
        if ped is None:
            self._box.set_visible(False)
            self.selected_id = None
            return
        dist_to_robot = None
        if scene.robot is not None:
            dist_to_robot = math.hypot(ped.x - scene.robot.x, ped.y - scene.robot.y)
        self._box.set_text(self._format(ped, dist_to_robot))
        self._box.set_visible(True)

    def _format(self, ped: AgentSnapshot, dist_to_robot: float | None) -> str:
        lines = [
            f"Pedestrian #{ped.id}",
            f"pos      ({ped.x:6.2f}, {ped.y:6.2f})",
            f"vel      ({ped.vx:6.2f}, {ped.vy:6.2f})",
            f"speed     {math.hypot(ped.vx, ped.vy):5.2f}",
            f"radius    {ped.radius:5.2f}",
        ]
        if ped.pref_speed is not None:
            lines.append(f"pref spd  {ped.pref_speed:5.2f}")
        if ped.goal_x is not None:
            lines.append(f"goal     ({ped.goal_x:6.2f}, {ped.goal_y:6.2f})")
        lines.append(f"group     {ped.group_id if ped.group_id is not None else '-'}")
        lines.append(f"leader    {'yes' if ped.is_leader else 'no'}")
        if ped.state is not None:
            lines.append(f"state     {ped.state}")
        if dist_to_robot is not None:
            lines.append(f"->robot   {dist_to_robot:5.2f}")
        return "\n".join(lines)
