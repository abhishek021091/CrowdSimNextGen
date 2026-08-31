"""CrowdSimNextGen visualization system.

A modular, RViz-Lite-style dashboard built on Matplotlib:

    Visualizer (MatplotlibVisualizer)
        SceneRenderer
            RobotRenderer
            CrowdRenderer
            ObstacleRenderer
            SensorRenderer
            OverlayRenderer      (planning / ORCA / predictions)
        StatisticsPanel
        MetricsPanel            (live research-metric plots)
        PedestrianInspector
        InteractionManager
        CameraController
        Recorder                (GIF / MP4 / PNG-sequence export)

Quick start
-----------
    from crowdsim_viz import MatplotlibVisualizer, VisualizerConfig

    viz = MatplotlibVisualizer(env, stepper)
    viz.show()

Or, for drop-in compatibility with the previous module::

    from crowdsim_viz import visualize_environment, visualize_simulation
"""

from .config import (
    AnimationConfig,
    ColorScheme,
    DARK_SCHEME,
    LIGHT_SCHEME,
    LayerToggles,
    LayoutConfig,
    TrailStyle,
    VisualizerConfig,
)
from .visualizer import MatplotlibVisualizer, visualize_environment, visualize_simulation

__all__ = [
    "MatplotlibVisualizer",
    "visualize_environment",
    "visualize_simulation",
    "VisualizerConfig",
    "LayerToggles",
    "TrailStyle",
    "AnimationConfig",
    "LayoutConfig",
    "ColorScheme",
    "LIGHT_SCHEME",
    "DARK_SCHEME",
]
