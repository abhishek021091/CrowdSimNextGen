"""Configuration objects for the CrowdSimNextGen visualizer.

All tunables live here so renderers stay declarative and the whole
system can be reconfigured (or serialized) without touching render code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(slots=True)
class LayerToggles:
    """Which optional visual layers are currently enabled.

    This is the single source of truth that every renderer reads from.
    ``InteractionManager`` mutates it in response to keyboard shortcuts;
    renderers never own their own "enabled" flag.
    """

    ids: bool = True
    names: bool = False
    velocity: bool = True
    preferred_velocity: bool = False
    goals: bool = True
    trails: bool = True
    sensor: bool = True
    orca: bool = False
    collision_cones: bool = False
    planning: bool = False
    prediction: bool = False
    grid: bool = True
    heading: bool = True
    labels: bool = True

    def toggle(self, name: str) -> bool:
        """Flip a toggle by name and return its new value.

        Raises ``AttributeError`` if ``name`` isn't a known toggle, so
        callers (e.g. keyboard shortcuts) fail loudly on typos.
        """
        current = getattr(self, name)
        new_value = not current
        setattr(self, name, new_value)
        return new_value

    def as_dict(self) -> dict[str, bool]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(slots=True)
class ColorScheme:
    """A named, swappable palette. Two instances (light/dark) ship below."""

    name: str
    figure_face: str
    axes_face: str
    text_color: str
    grid_color: str
    grid_alpha: float
    robot_color: str
    robot_edge: str
    obstacle_face: str
    obstacle_edge: str
    ungrouped_ped_color: str
    leader_edge: str
    sensor_ray_color: str
    sensor_hit_color: str
    sensor_range_color: str
    orca_color: str
    prediction_color: str
    path_color: str
    panel_face: str
    panel_edge: str
    cmap_name: str = "tab20"

    def cmap(self, plt_module: Any):
        return plt_module.get_cmap(self.cmap_name)


LIGHT_SCHEME = ColorScheme(
    name="light",
    figure_face="#f7f7f9",
    axes_face="#ffffff",
    text_color="#1a1a1a",
    grid_color="#000000",
    grid_alpha=0.08,
    robot_color="#d62728",
    robot_edge="#1a1a1a",
    obstacle_face="#d9d9d9",
    obstacle_edge="#5c5c5c",
    ungrouped_ped_color="#6e6e6e",
    leader_edge="#000000",
    sensor_ray_color="#f2c14e",
    sensor_hit_color="#e07a5f",
    sensor_range_color="#3a86ff",
    orca_color="#8338ec",
    prediction_color="#219ebc",
    path_color="#2a9d8f",
    panel_face="#ffffff",
    panel_edge="#c9c9c9",
)

DARK_SCHEME = ColorScheme(
    name="dark",
    figure_face="#121212",
    axes_face="#1b1b1f",
    text_color="#e8e8e8",
    grid_color="#ffffff",
    grid_alpha=0.08,
    robot_color="#ff5c5c",
    robot_edge="#f2f2f2",
    obstacle_face="#3a3a3f",
    obstacle_edge="#8a8a90",
    ungrouped_ped_color="#a0a0a6",
    leader_edge="#ffffff",
    sensor_ray_color="#ffd166",
    sensor_hit_color="#ff8fa3",
    sensor_range_color="#4cc9f0",
    orca_color="#c77dff",
    prediction_color="#4ea8de",
    path_color="#57cc99",
    panel_face="#1e1e22",
    panel_edge="#3a3a3f",
)


@dataclass(slots=True)
class TrailStyle:
    enabled: bool = True
    length: int = 30
    fade: bool = True
    spline_smoothing: bool = False
    spline_points: int = 60
    linewidth: float = 1.4


@dataclass(slots=True)
class AnimationConfig:
    interval_ms: int = 80
    interpolate: bool = True
    sub_steps: int = 0
    """How many interpolated draw-frames per simulation step."""


@dataclass(slots=True)
class LayoutConfig:
    figsize: tuple[float, float] = (16.0, 9.0)
    padding: float = 1.5
    show_stats_panel: bool = True
    show_metrics_panel: bool = True
    stats_panel_width_ratio: float = 0.22
    metrics_panel_height_ratio: float = 0.32
    metrics_history: int = 300


@dataclass(slots=True)
class VisualizerConfig:
    """Top-level configuration bag passed into ``Visualizer``."""

    title: str = "CrowdSimNextGen \u2014 Visualization"
    layers: LayerToggles = field(default_factory=LayerToggles)
    trail: TrailStyle = field(default_factory=TrailStyle)
    animation: AnimationConfig = field(default_factory=AnimationConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    scheme: ColorScheme = field(default_factory=lambda: LIGHT_SCHEME)
    goal_reach_threshold: float = 0.1
    max_agents_warning: int = 1000
