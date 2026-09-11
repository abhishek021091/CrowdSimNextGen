"""Visualizer: renders the live Environment for manual/debug viewing.

Two windows:
  - Main view: the full arena, fixed extent, for overview/orientation.
  - Local view: a small popup docked beside the main window, always
    centered on the robot and clipped to a circle matching its sensor
    range -- a "radar" view for situations where the arena is too large
    (or too tall/narrow) for per-agent detail to read at full-arena zoom.

Both views are drawn through the exact same sub-visualizers
(ObstacleVisualizer/CrowdVisualizer/RobotVisualizer/CellVisualizer),
which already take a plain (data, ax) and know nothing about which
figure owns that ax. This means the two views can never visually
diverge -- same colors, same markers, same logic -- only the axes
extent (and, for the local view, a circular clip) differs. No drawing
logic is duplicated here.

Why plt.pause() is never used in refresh():
    plt.pause() calls show(block=False) internally, which on Qt
    backends re-asserts/raises the figure window on every call -- not
    just "process pending GUI events" the way its name suggests. Called
    once per simulation tick (10Hz+) across two figures, this repeatedly
    steals window focus/activation from whatever else is running on the
    machine. draw_idle() + flush_events() give the same "repaint and
    pump the event queue" behavior without ever calling show() or
    activating the window, so that's what refresh() uses instead. If a
    frame-rate cap is ever needed, use time.sleep() directly -- it does
    not touch window focus.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import tomllib
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle as MplCircle

import navcore.configs
from navcore.boustropheden.boustropheden import DecompositionResult
from navcore.entities.environment.environment import Environment
from navcore.visualization.entities.cell_visualizer import CellVisualizer
from navcore.visualization.entities.crowd_visualizer import CrowdVisualizer
from navcore.visualization.entities.obstacle_visualizer import ObstacleVisualizer
from navcore.visualization.entities.robot_visualizer import RobotVisualizer


class Visualizer:
    """Renders one Environment across a main overview window and a
    robot-centered local ("radar") popup.

    Attributes:
        arena_width: Full arena width, read once from env.toml.
        arena_height: Full arena height, read once from env.toml.
        local_view_size: Fallback local-view diameter (world units),
            used only if the robot has no sensor yet -- normally the
            local view's radius is the robot's live sensor range, not
            this fixed value. Kept as a separate constructor parameter
            from the popup's *physical* figure size (also 5x5 by
            default) since those are two independent quantities that
            happen to share a default value.
        env: The most recently visualized Environment.
        decomposition: Sticky cell overlay -- see set_decomposition().
    """

    def __init__(self, local_view_size: float = 10.0) -> None:
        env_path = Path(navcore.configs.__file__).parent / "env.toml"
        with open(env_path, "rb") as f:
            arena = tomllib.load(f)["arenaSize"]
        self.arena_width = arena["width"]
        self.arena_height = arena["height"]
        self.local_view_size = local_view_size

        self.env: Environment | None = None
        self.decomposition: DecompositionResult | None = None

        # Interactive mode must be on for a script (not just IPython/Jupyter)
        # to get non-blocking windows at all. Without this, fig.show() below
        # either does nothing or blocks depending on backend state.
        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.manager.set_window_title("CrowdSimNextGen - Arena")
        self.fig.show()  # Show once, here only -- refresh() must never call this again.

        self._init_local_view()

    # -- construction ------------------------------------------------------

    def _init_local_view(self) -> None:
        self.fig_local, self.ax_local = plt.subplots(figsize=(10, 10))
        self.fig_local.canvas.manager.set_window_title("CrowdSimNextGen - Robot View")
        self.fig_local.show()  # Same one-time show as the main figure.
        self._dock_local_view_beside_main()

    def _dock_local_view_beside_main(self) -> None:
        main_window = self.fig.canvas.manager.window
        local_window = self.fig_local.canvas.manager.window
        main_geom = main_window.geometry()
        local_window.move(main_geom.x() + main_geom.width() + 10, main_geom.y())

    # -- public API ----------------------------------------------------------

    def set_decomposition(self, decomposition: DecompositionResult | None) -> None:
        """Set (or clear, with ``None``) the cell overlay drawn by future
        ``visualize()``/``refresh()`` calls that don't pass their own.
        """
        self.decomposition = decomposition

    def visualize(
        self,
        env: Environment,
        mission=None,
        decomposition: DecompositionResult | None = None,
    ) -> None:
        self.env = env
        if decomposition is not None:
            self.decomposition = decomposition

        self._draw_main_view(env, mission)
        self._draw_local_view(env, mission)

    def refresh(
        self,
        env: Environment,
        mission=None,
        decomposition: DecompositionResult | None = None,
    ) -> None:
        """Redraw both windows for the current tick.

        Deliberately does not call plt.pause() -- see module docstring
        for why that repeatedly steals window focus in a live loop.
        draw_idle() + flush_events() repaint and pump each figure's own
        GUI event queue without ever activating/raising the window.
        """
        self.visualize(env, mission=mission, decomposition=decomposition)

        self.fig.canvas.draw_idle()
        self.fig_local.canvas.draw_idle()
        self._flush_events(self.fig)
        self._flush_events(self.fig_local)

    @staticmethod
    def _flush_events(fig) -> None:
        try:
            fig.canvas.flush_events()
        except Exception:
            # Some backends implement flush_events() as a no-op or omit
            # it entirely -- rendering must degrade gracefully, not crash
            # the simulation loop over a cosmetic repaint step.
            pass

    def animate(
        self, env: Environment, n_frames: int = 30, interval: int = 100
    ) -> FuncAnimation:
        self.env = env
        self.ani = FuncAnimation(
            self.fig, self.refresh, frames=range(n_frames), interval=interval
        )
        plt.show(block=False)

    # -- main view -----------------------------------------------------------

    def _draw_main_view(self, env: Environment, mission) -> None:
        self.ax.clear()
        self.ax.set_xlim(-self.arena_width / 2 - 0.5, self.arena_width / 2 + 0.5)
        self.ax.set_ylim(-self.arena_height / 2 - 0.5, self.arena_height / 2 + 0.5)
        self.ax.set_aspect("equal")

        if self.decomposition is not None:
            CellVisualizer(self.decomposition, self.ax).draw()

        CrowdVisualizer(env, self.ax).draw()
        ObstacleVisualizer(env, self.ax).draw()
        RobotVisualizer(env, self.ax, mission=mission).draw()

    # -- local ("radar") view -------------------------------------------------

    # -- local ("radar") view -------------------------------------------------

    def _draw_local_view(self, env: Environment, mission) -> None:
        """Redraw the robot-centered local view, reusing the same
        sub-visualizers as the main view.

        The view is a fixed-size square window centered on the robot --
        not clipped to the sensor-range circle -- so it shows a
        consistent 10m x 10m area regardless of the robot's sensor
        configuration. Falls back to ``local_view_size`` if the robot
        has no pose yet (should be transient, not steady-state).
        """
        self.ax_local.clear()
        if env.robot.pose is None:
            return

        half_extent = self.local_view_size
        cx, cy = env.robot.pose.px, env.robot.pose.py

        self.ax_local.set_xlim(cx - half_extent, cx + half_extent)
        self.ax_local.set_ylim(cy - half_extent, cy + half_extent)
        self.ax_local.set_aspect("equal")
        self.ax_local.set_facecolor("white")
        self.ax_local.set_xticks([])
        self.ax_local.set_yticks([])
        for spine in self.ax_local.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)

        if self.decomposition is not None:
            CellVisualizer(self.decomposition, self.ax_local).draw()

        CrowdVisualizer(env, self.ax_local).draw()
        ObstacleVisualizer(env, self.ax_local).draw()
        RobotVisualizer(env, self.ax_local, mission=mission).draw()

    def _clip_local_view_to_circle(self, cx: float, cy: float, radius: float) -> None:
        """Clip every artist drawn in ``ax_local`` to a circle of
        ``radius`` centered on ``(cx, cy)``.

        Matplotlib has no native circular-axes mode -- the standard
        workaround is a rectangular axes whose *contents* are
        individually clipped to a Circle patch. This must be applied
        per-artist, not once on the axes, since clip paths don't
        propagate to children automatically. The figure/window itself
        remains square with white corners outside the circle; a truly
        round window silhouette would require OS-level window masking
        (e.g. Qt's QRegion-based setMask), which is out of scope here.
        """
        clip_circle = MplCircle((cx, cy), radius, transform=self.ax_local.transData)
        for artist in self.ax_local.get_children():
            artist.set_clip_path(clip_circle)

        # Draw the boundary itself last so it isn't clipped away too --
        # this is what makes the circular edge read clearly against the
        # otherwise-square figure.
        self.ax_local.add_patch(
            MplCircle(
                (cx, cy),
                radius,
                fill=False,
                edgecolor="black",
                linewidth=1.5,
                zorder=10,
            )
        )
