"""Top-level entry point: ``Visualizer``.

Wires together the camera, scene renderer, statistics/metrics panels,
interaction manager, and recorder into the public API described in the
module docstring below. Every sub-system is a small, independently
testable class (see ``camera.py``, ``scene.py``, ``renderers/``,
``panels/``, ``interaction.py``, ``recorder.py``); this file's only job
is composition and the animation loop.

Keyboard shortcuts
-------------------
    Space   pause / resume
    R       reset
    F       toggle fullscreen
    T       toggle trails            G   toggle goals
    I       toggle ids               V   toggle velocity
    S       toggle sensor            P   toggle planning
    O       toggle ORCA              L   toggle labels

Mouse
-----
    Scroll          zoom (centered on cursor)
    Left-drag        pan
    Right-click      inspect a pedestrian

Example
-------
    viz = MatplotlibVisualizer(env, stepper)
    viz.show()
    viz.save("episode.mp4")
    viz.toggle("sensor")
    viz.toggle("trails")
    viz.follow_robot(True)
    viz.dark_mode()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from .camera import CameraController
from .config import DARK_SCHEME, LIGHT_SCHEME, VisualizerConfig
from .inspector import PedestrianInspector
from .interaction import InteractionCallbacks, InteractionManager
from .panels import MetricsPanel, StatisticsPanel
from .recorder import Recorder
from .renderers import (
    CrowdRenderer,
    ObstacleRenderer,
    OverlayRenderer,
    RobotRenderer,
    SensorRenderer,
)
from .scene import SceneRenderer
from .state import SceneSnapshot, build_scene_snapshot, interpolate_scene

_METRIC_GRID_KEYS = (
    "distance_to_goal",
    "speed",
    "clearance",
    "reward",
    "collisions",
    "discomfort",
    "orca_solve_time",
    "episode_reward",
)


class MatplotlibVisualizer:
    """A modular, extensible research/dashboard-style visualizer for
    CrowdSimNextGen. See module docstring for shortcuts and API.
    """

    def __init__(
        self,
        env: Any,
        stepper: Any | None = None,
        config: VisualizerConfig | None = None,
    ) -> None:
        self.env = env
        self.stepper = stepper
        self.config = config or VisualizerConfig()

        self._sim_dt = float(
            getattr(env, "time_step", getattr(env, "dt", 0.25)) or 0.25
        )
        self._sim_time = 0.0
        self._step_count = 0
        self._paused = False
        self._closed = False
        self._sub_frame_index = 0

        self._build_figure()
        self._build_renderers()
        self._build_panels()

        self.camera = CameraController(
            self.scene_ax, padding=self.config.layout.padding
        )
        self.inspector = PedestrianInspector(self.scene_ax, self.config.scheme)

        self._prev_scene: SceneSnapshot = build_scene_snapshot(env, 0, 0.0)
        self._curr_scene: SceneSnapshot = self._prev_scene
        self.scene.build_static(getattr(env, "obstacles", {}) or {})
        self._fit_camera(self._curr_scene)
        self.scene.update(self._curr_scene, self.config.layers)
        self.inspector.update(self._curr_scene)

        self._build_interaction()

        self._anim: FuncAnimation | None = None
        self.recorder = Recorder(self.fig)

    # -- construction ---------------------------------------------------
    def _build_figure(self) -> None:
        layout = self.config.layout
        self.fig = plt.figure(figsize=layout.figsize)
        self.fig.patch.set_facecolor(self.config.scheme.figure_face)

        n_metric_cols = 4
        n_metric_rows = 2
        outer = gridspec.GridSpec(
            2,
            2,
            figure=self.fig,
            width_ratios=[
                1 - layout.stats_panel_width_ratio,
                layout.stats_panel_width_ratio,
            ],
            height_ratios=[
                1 - layout.metrics_panel_height_ratio,
                layout.metrics_panel_height_ratio,
            ],
            hspace=0.28,
            wspace=0.05,
        )
        self.scene_ax = self.fig.add_subplot(outer[0, 0])
        self.stats_ax = self.fig.add_subplot(outer[0, 1])
        self.stats_ax.set_visible(self.config.layout.show_stats_panel)

        metrics_gs = gridspec.GridSpecFromSubplotSpec(
            n_metric_rows,
            n_metric_cols,
            subplot_spec=outer[1, :],
            hspace=0.7,
            wspace=0.35,
        )
        self._metric_axes = {}
        for idx, key in enumerate(_METRIC_GRID_KEYS):
            r, c = divmod(idx, n_metric_cols)
            ax = self.fig.add_subplot(metrics_gs[r, c])
            self._metric_axes[key] = ax
        if not self.config.layout.show_metrics_panel:
            for ax in self._metric_axes.values():
                ax.set_visible(False)

    def _build_renderers(self) -> None:
        scheme = self.config.scheme
        from navcore.policies.base_orca_planner import obstacle_to_vertices

        robot = RobotRenderer(self.scene_ax, scheme)
        crowd = CrowdRenderer(self.scene_ax, scheme, self.config.trail)
        obstacles = ObstacleRenderer(self.scene_ax, scheme, obstacle_to_vertices)
        sensor = SensorRenderer(self.scene_ax, scheme)
        overlay = OverlayRenderer(self.scene_ax, scheme)
        self.scene = SceneRenderer(
            self.scene_ax,
            scheme,
            robot,
            crowd,
            obstacles,
            sensor,
            overlay,
            self.config.title,
        )

    def _build_panels(self) -> None:
        self.stats_panel = StatisticsPanel(
            self.stats_ax,
            self.config.scheme,
            goal_threshold=self.config.goal_reach_threshold,
        )
        self.metrics_panel = MetricsPanel(
            self._metric_axes,
            self.config.scheme,
            history=self.config.layout.metrics_history,
        )

    def _build_interaction(self) -> None:
        callbacks = InteractionCallbacks(
            on_pause_toggle=self._toggle_pause,
            on_reset=self.reset,
            on_fullscreen_toggle=self._toggle_fullscreen,
            on_pedestrian_click=self.inspector.select,
            on_toggle_changed=lambda _name, _val: None,
        )
        self.interaction = InteractionManager(
            self.fig,
            self.scene_ax,
            self.config.layers,
            callbacks,
            pedestrian_position_lookup=self._pedestrian_positions,
        )
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _pedestrian_positions(self) -> dict[int, tuple[float, float, float]]:
        return {
            pid: (p.x, p.y, p.radius) for pid, p in self._curr_scene.pedestrians.items()
        }

    # -- lifecycle --------------------------------------------------------
    def _on_close(self, _event: Any) -> None:
        self._closed = True
        if self._anim is not None and self._anim.event_source is not None:
            self._anim.event_source.stop()

    def _toggle_pause(self) -> None:
        self._paused = not self._paused

    def _toggle_fullscreen(self) -> None:
        manager = getattr(self.fig.canvas, "manager", None)
        toggler = getattr(manager, "full_screen_toggle", None)
        if callable(toggler):
            toggler()

    def reset(self) -> None:
        """Reset step/time counters and panel history. Also resets the
        underlying stepper/env if it exposes a ``reset`` method."""
        if self.stepper is not None and hasattr(self.stepper, "reset"):
            self.stepper.reset()
        self._sim_time = 0.0
        self._step_count = 0
        self._sub_frame_index = 0
        self.stats_panel.reset()
        self.metrics_panel.reset()
        self.scene.robot.clear()
        self.scene.crowd.clear()
        self._prev_scene = build_scene_snapshot(self.env, 0, 0.0)
        self._curr_scene = self._prev_scene
        self._fit_camera(self._curr_scene)

    def _fit_camera(self, scene: SceneSnapshot) -> None:
        xs, ys = self.scene.collect_bounds(scene)
        self.camera.fit_bounds(xs, ys)

    # -- public toggles -----------------------------------------------
    def toggle(self, name: str) -> bool:
        """Toggle a named debug layer. Returns the new value.

        Valid names match ``LayerToggles`` fields, e.g. "sensor",
        "trails", "orca", "planning", "ids", "velocity", "goals",
        "labels", "grid", "prediction", "collision_cones", "heading",
        "preferred_velocity", "names".
        """
        return self.config.layers.toggle(name)

    def follow_robot(self, enabled: bool = True) -> None:
        self.camera.follow_robot(enabled)

    def dark_mode(self) -> None:
        self._apply_scheme(DARK_SCHEME)

    def light_mode(self) -> None:
        self._apply_scheme(LIGHT_SCHEME)

    def _apply_scheme(self, scheme) -> None:
        self.config.scheme = scheme
        self.fig.patch.set_facecolor(scheme.figure_face)
        self.scene_ax.set_facecolor(scheme.axes_face)
        self.stats_ax.set_facecolor(scheme.panel_face)
        self.scene.set_scheme(scheme)
        self.stats_panel.set_scheme(scheme)
        self.metrics_panel.set_scheme(scheme)
        self.inspector.set_scheme(scheme)
        self.fig.canvas.draw_idle()

    # -- animation loop -----------------------------------------------
    def _advance_step(self) -> None:
        t0 = time.perf_counter()
        if self.stepper is not None:
            self.stepper.step()
        planner_ms = (time.perf_counter() - t0) * 1000.0
        self.stats_panel.set_planner_latency_ms(planner_ms)

        self._step_count += 1
        self._sim_time += self._sim_dt
        self._prev_scene = self._curr_scene
        self._curr_scene = build_scene_snapshot(
            self.env, self._step_count, self._sim_time
        )

    # in visualizer.py, MatplotlibVisualizer
    def refresh(self) -> None:
        """Manually redraw the current env state — for callers driving
        their own step loop instead of handing a stepper to the animation."""
        self._step_count += 1
        self._sim_time += self._sim_dt
        self._prev_scene = self._curr_scene
        self._curr_scene = build_scene_snapshot(
            self.env, self._step_count, self._sim_time
        )

        self.scene.update(self._curr_scene, self.config.layers)
        self.inspector.update(self._curr_scene)
        self.camera.maybe_follow(
            self._curr_scene.robot.x if self._curr_scene.robot else None,
            self._curr_scene.robot.y if self._curr_scene.robot else None,
        )
        self.stats_panel.update(self._curr_scene, self.env)
        self.metrics_panel.update(self._curr_scene, self.stats_panel.collisions)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)  # lets the GUI event loop process/repaint without blocking

    def _frame(self, _frame_index: int):
        if self._closed:
            return []
        frame_start = self.stats_panel.mark_frame_start()

        if not self._paused:
            sub_steps = max(1, self.config.animation.sub_steps)
            if self._sub_frame_index == 0:
                if self.stepper is not None:
                    self.update_scene()
                self._advance_step()
            self._sub_frame_index = (self._sub_frame_index + 1) % sub_steps

            if self.config.animation.interpolate and sub_steps > 1:
                t = self._sub_frame_index / sub_steps
                display_scene = interpolate_scene(self._prev_scene, self._curr_scene, t)
            else:
                display_scene = self._curr_scene
        else:
            display_scene = self._curr_scene

        self.scene.update(display_scene, self.config.layers)
        self.inspector.update(display_scene)
        self.camera.maybe_follow(
            display_scene.robot.x if display_scene.robot else None,
            display_scene.robot.y if display_scene.robot else None,
        )
        self.stats_panel.update(display_scene, self.env)
        self.metrics_panel.update(display_scene, self.stats_panel.collisions)

        self.stats_panel.mark_frame_end(frame_start)
        return []

    # -- public entry points -------------------------------------------
    def show(self, *, block: bool = True) -> FuncAnimation:
        sub_steps = max(1, self.config.animation.sub_steps)
        interval = self.config.animation.interval_ms / sub_steps
        self._anim = FuncAnimation(
            self.fig,
            self._frame,
            interval=interval,
            blit=False,
            cache_frame_data=False,
        )
        plt.show(block=block)
        self._closed = True
        if self._anim is not None and self._anim.event_source is not None:
            self._anim.event_source.stop()
        return self._anim

    def save(
        self, path: str | Path, *, fps: int = 20, dpi: int = 150, n_frames: int = 300
    ) -> None:
        """Render ``n_frames`` frames and export as GIF/MP4 (by extension)."""
        sub_steps = max(1, self.config.animation.sub_steps)
        interval = self.config.animation.interval_ms / sub_steps
        anim = FuncAnimation(
            self.fig,
            self._frame,
            frames=n_frames,
            interval=interval,
            blit=False,
            cache_frame_data=False,
        )
        self.recorder.save_animation(anim, path, fps=fps, dpi=dpi)

    def save_png_sequence(
        self, out_dir: str | Path, *, n_frames: int = 300, dpi: int = 150
    ) -> list[Path]:
        return self.recorder.save_png_sequence(self._frame, n_frames, out_dir, dpi=dpi)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False


def visualize_environment(
    env: Any, *, title: str = "CrowdSimNextGen Visualization"
) -> MatplotlibVisualizer:
    """Show a static snapshot of the environment (no stepping)."""
    config = VisualizerConfig(title=title)
    viz = MatplotlibVisualizer(env=env, stepper=None, config=config)
    viz.pause()
    viz.show()
    return viz


def visualize_simulation(
    env: Any,
    stepper: Any,
    *,
    interval: int = 80,
    trail_length: int = 30,
    title: str = "CrowdSimNextGen Simulation",
) -> MatplotlibVisualizer:
    """Show a live simulation, advancing one sim step every ``interval`` ms."""
    config = VisualizerConfig(title=title)
    config.animation.interval_ms = interval
    config.trail.length = trail_length
    viz = MatplotlibVisualizer(env=env, stepper=stepper, config=config)
    viz.show()
    return viz
