from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from ..config import LayerToggles, TrailStyle
from ..state import SceneSnapshot
from .base import Renderer


class _PedestrianArtists:
    """Per-agent artists that can't be batched cheaply (circle body, text
    label, goal marker, trail). Heading/velocity arrows are *not* here --
    they're drawn as two shared ``Quiver`` collections owned by the
    renderer, since recreating a ``FancyArrow`` per agent per frame does
    not scale to hundreds of pedestrians."""

    __slots__ = ("body", "label", "goal_marker", "trail", "history")

    def __init__(self) -> None:
        self.body: Circle | None = None
        self.label = None
        self.goal_marker = None
        self.trail: LineCollection | None = None
        self.history: deque[tuple[float, float]] = deque()


class CrowdRenderer(Renderer):
    """Draws every pedestrian: body, arrows, labels, goal, and a fading
    trail. Colors follow group membership; leaders get a bold outline.

    Performance note: circles/labels/goal-markers/trails are one artist
    per agent (matplotlib has no batched circle-with-per-instance-radius
    primitive that also supports independent edge styling for leaders),
    but they are created once and mutated in place thereafter. Heading
    and velocity arrows *are* batched, via two ``Quiver`` collections
    updated with a single ``set_offsets``/``set_UVC`` call per frame --
    this is what keeps 500-1000 agent scenes responsive.
    """

    LEADER_LINEWIDTH = 2.6
    FOLLOWER_LINEWIDTH = 1.3

    def __init__(self, ax: Any, scheme: Any, trail: TrailStyle) -> None:
        super().__init__(ax, scheme)
        self.trail_style = trail
        self._agents: dict[int, _PedestrianArtists] = {}
        self._heading_quiver = None
        self._velocity_quiver = None

    def clear(self) -> None:
        for artists in self._agents.values():
            for artist in (
                artists.body,
                artists.label,
                artists.goal_marker,
                artists.trail,
            ):
                if artist is not None:
                    artist.remove()
        self._agents.clear()
        if self._heading_quiver is not None:
            self._heading_quiver.remove()
            self._heading_quiver = None
        if self._velocity_quiver is not None:
            self._velocity_quiver.remove()
            self._velocity_quiver = None

    def _group_color(self, group_id: int | None):
        if group_id is None:
            return self.scheme.ungrouped_ped_color
        import matplotlib.pyplot as plt

        cmap = self.scheme.cmap(plt)
        return cmap(group_id % cmap.N)

    def _prune_missing(self, scene: SceneSnapshot) -> None:
        stale = [pid for pid in self._agents if pid not in scene.pedestrians]
        for pid in stale:
            artists = self._agents.pop(pid)
            for artist in (
                artists.body,
                artists.label,
                artists.goal_marker,
                artists.trail,
            ):
                if artist is not None:
                    artist.remove()

    def update(self, scene: SceneSnapshot, toggles: LayerToggles) -> None:
        self._prune_missing(scene)
        for ped_id, ped in scene.pedestrians.items():
            artists = self._agents.setdefault(ped_id, _PedestrianArtists())
            if artists.history.maxlen != self.trail_style.length:
                artists.history = deque(artists.history, maxlen=self.trail_style.length)
            self._update_body(ped, artists, toggles)
            self._update_label(ped_id, ped, artists, toggles)
            self._update_goal(ped, artists, toggles)
            self._update_trail(ped, artists, toggles)
        self._update_arrows(scene, toggles)

    def _update_body(
        self, ped, artists: _PedestrianArtists, toggles: LayerToggles
    ) -> None:
        color = self._group_color(ped.group_id)
        edge = self.scheme.leader_edge if ped.is_leader else color
        lw = self.LEADER_LINEWIDTH if ped.is_leader else self.FOLLOWER_LINEWIDTH
        alpha = 0.5 if ped.is_leader else 0.38
        if artists.body is None:
            artists.body = Circle(
                (ped.x, ped.y),
                ped.radius,
                facecolor=color,
                edgecolor=edge,
                linewidth=lw,
                alpha=alpha,
                zorder=5,
            )
            self.ax.add_patch(artists.body)
        else:
            artists.body.center = (ped.x, ped.y)
            artists.body.radius = ped.radius
            artists.body.set_facecolor(color)
            artists.body.set_edgecolor(edge)
            artists.body.set_linewidth(lw)
            artists.body.set_alpha(alpha)

    def _update_arrows(self, scene: SceneSnapshot, toggles: LayerToggles) -> None:
        peds = list(scene.pedestrians.values())
        if not peds:
            if self._heading_quiver is not None:
                self._heading_quiver.set_offsets(np.empty((0, 2)))
            if self._velocity_quiver is not None:
                self._velocity_quiver.set_offsets(np.empty((0, 2)))
            return

        xs = np.array([p.x for p in peds])
        ys = np.array([p.y for p in peds])
        colors = np.array([_to_rgba(self._group_color(p.group_id)) for p in peds])

        if toggles.heading:
            radii = np.array([p.radius for p in peds])
            hx = np.cos([p.heading for p in peds]) * radii
            hy = np.sin([p.heading for p in peds]) * radii
            if self._heading_quiver is None:
                self._heading_quiver = self.ax.quiver(
                    xs,
                    ys,
                    hx,
                    hy,
                    color=colors,
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    width=0.003,
                    alpha=0.85,
                    zorder=6,
                )
            else:
                self._heading_quiver.set_offsets(np.column_stack([xs, ys]))
                self._heading_quiver.set_UVC(hx, hy)
                self._heading_quiver.set_color(colors)
            self._heading_quiver.set_visible(True)
        elif self._heading_quiver is not None:
            self._heading_quiver.set_visible(False)

        if toggles.velocity:
            vx = np.array([p.vx for p in peds]) * 0.5
            vy = np.array([p.vy for p in peds]) * 0.5
            vel_colors = np.tile(_to_rgba("#457b9d"), (len(peds), 1))
            if self._velocity_quiver is None:
                self._velocity_quiver = self.ax.quiver(
                    xs,
                    ys,
                    vx,
                    vy,
                    color=vel_colors,
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    width=0.003,
                    alpha=0.75,
                    zorder=6,
                )
            else:
                self._velocity_quiver.set_offsets(np.column_stack([xs, ys]))
                self._velocity_quiver.set_UVC(vx, vy)
            self._velocity_quiver.set_visible(True)
        elif self._velocity_quiver is not None:
            self._velocity_quiver.set_visible(False)

    def _update_label(self, ped_id: int, ped, artists, toggles: LayerToggles) -> None:
        want_label = toggles.ids or (toggles.names and ped.name)
        if not want_label:
            if artists.label is not None:
                artists.label.set_visible(False)
            return
        text = ped.name if (toggles.names and ped.name) else str(ped_id)
        if ped.is_leader:
            text += "\u2605"
        if artists.label is None:
            artists.label = self.ax.text(
                ped.x,
                ped.y,
                text,
                ha="center",
                va="center",
                fontsize=7.5,
                color=self.scheme.text_color,
                weight="bold" if ped.is_leader else "normal",
                zorder=7,
            )
        else:
            artists.label.set_visible(True)
            artists.label.set_position((ped.x, ped.y))
            artists.label.set_text(text)

    def _update_goal(self, ped, artists, toggles: LayerToggles) -> None:
        if not toggles.goals or ped.goal_x is None:
            if artists.goal_marker is not None:
                artists.goal_marker.set_visible(False)
            return
        color = self._group_color(ped.group_id)
        if artists.goal_marker is None:
            (artists.goal_marker,) = self.ax.plot(
                [],
                [],
                marker="x",
                markersize=7,
                linestyle="None",
                color=color,
                markeredgewidth=1.6,
                zorder=4,
            )
        artists.goal_marker.set_visible(True)
        artists.goal_marker.set_data([ped.goal_x], [ped.goal_y])
        artists.goal_marker.set_color(color)

    def _update_trail(self, ped, artists, toggles: LayerToggles) -> None:
        artists.history.append((ped.x, ped.y))
        if not toggles.trails or len(artists.history) < 2:
            if artists.trail is not None:
                artists.trail.set_visible(False)
            return
        color = self._group_color(ped.group_id)
        pts = np.array(artists.history)
        if self.trail_style.spline_smoothing and len(pts) >= 4:
            pts = _smooth_polyline(pts, self.trail_style.spline_points)
        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        n = len(segments)
        if self.trail_style.fade:
            alphas = np.linspace(0.05, 0.75, n)
        else:
            alphas = np.full(n, 0.5)
        rgba = np.tile(np.array(_to_rgba(color)), (n, 1))
        rgba[:, 3] = alphas
        if artists.trail is None:
            artists.trail = LineCollection(
                [], linewidths=self.trail_style.linewidth, zorder=3
            )
            self.ax.add_collection(artists.trail)
        artists.trail.set_visible(True)
        artists.trail.set_segments(segments)
        artists.trail.set_color(rgba)

    def bounds(self, scene: SceneSnapshot):
        xs, ys = [], []
        for ped in scene.pedestrians.values():
            xs.append(ped.x)
            ys.append(ped.y)
            if ped.goal_x is not None:
                xs.append(ped.goal_x)
                ys.append(ped.goal_y)
        return xs, ys


def _to_rgba(color) -> tuple[float, float, float, float]:
    from matplotlib.colors import to_rgba

    return to_rgba(color)


def _smooth_polyline(pts: np.ndarray, n_out: int) -> np.ndarray:
    """Cheap Catmull-Rom-ish smoothing without a scipy dependency."""
    if len(pts) < 4:
        return pts
    t = np.linspace(0, 1, len(pts))
    t_out = np.linspace(0, 1, n_out)
    x = np.interp(t_out, t, pts[:, 0])
    y = np.interp(t_out, t, pts[:, 1])
    return np.stack([x, y], axis=1)
