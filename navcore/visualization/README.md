# crowdsim_viz

A ground-up redesign of the CrowdSimNextGen Matplotlib visualizer:
a modular, RViz-Lite-style dashboard instead of a single monolithic
class.

## Architecture

```
MatplotlibVisualizer            (visualizer.py — composition + animation loop)
    CameraController            (camera.py — zoom / pan / follow-robot)
    SceneRenderer                (scene.py — axes chrome, legend, bounds)
        RobotRenderer            (renderers/robot.py)
        CrowdRenderer            (renderers/crowd.py)
        ObstacleRenderer         (renderers/obstacle.py)
        SensorRenderer           (renderers/sensor.py)
        OverlayRenderer          (renderers/overlay.py — planning/ORCA/predictions)
    StatisticsPanel              (panels/statistics.py)
    MetricsPanel                 (panels/metrics.py — 8 live research plots)
    PedestrianInspector          (inspector.py — right-click to inspect)
    InteractionManager           (interaction.py — keyboard + mouse routing)
    Recorder                     (recorder.py — GIF / MP4 / PNG-sequence export)
```

Each renderer owns exactly one visual concern, reads from a
`SceneSnapshot` (`state.py` — a plain-data snapshot of the environment
taken once per simulation step), and mutates its own artists in place.
Nothing is torn down and rebuilt every frame except where Matplotlib
gives no batched alternative (see Performance below).

## Quick start

```python
from crowdsim_viz import MatplotlibVisualizer, VisualizerConfig

viz = MatplotlibVisualizer(env, stepper)
viz.show()
viz.save("episode.mp4")          # or "episode.gif"
viz.toggle("sensor")
viz.toggle("trails")
viz.follow_robot(True)
viz.dark_mode()
```

Drop-in replacement for the old module-level functions:

```python
from crowdsim_viz import visualize_environment, visualize_simulation
```

### Configuration

Every tunable lives in `VisualizerConfig` (`config.py`) — layer
toggles, trail style, animation/interpolation, panel layout, and the
light/dark color scheme — so behavior can be set up-front instead of
threaded through constructor kwargs:

```python
from crowdsim_viz import VisualizerConfig

config = VisualizerConfig()
config.animation.sub_steps = 4       # interpolated draw-frames per sim step
config.trail.length = 50
config.trail.spline_smoothing = True
config.layers.sensor = True
config.layout.show_metrics_panel = False

viz = MatplotlibVisualizer(env, stepper, config=config)
```

## Optional data sources

Sensor, planning, ORCA, and prediction overlays are entirely
duck-typed against optional attributes on `env.robot` /
`env`, read once per step in `state.build_scene_snapshot`:

| Overlay      | Looked up on           | Expected shape |
|--------------|-------------------------|----------------|
| Sensor       | `robot.sensor_observation` (or `.sensor_data`/`.lidar`) | `{"rays": [(x0,y0,x1,y1,hit), ...], "visibility_polygon": [(x,y), ...]}` |
| Planned path | `robot.planned_path` (or `.path`/`.waypoints`) | list of `(x, y)` or objects with `.x`/`.y` |
| ORCA debug   | `robot.orca_debug` (or `.orca_data`) | object/dict with `preferred_velocity`, `chosen_velocity`, `collision_cones` |
| Predictions  | `env.predicted_trajectories` | `{agent_id: [(x, y), ...]}` |
| Research metrics | `scene.extra["metrics"]` per step (populate via a custom stepper) | `{"reward": ..., "discomfort": ..., "orca_solve_time_ms": ...}` |

If an env doesn't provide one of these, that layer simply renders
nothing when toggled on — it never errors.

## Interaction

Keyboard: `Space` pause/resume · `R` reset · `F` fullscreen ·
`T` trails · `G` goals · `I` ids · `V` velocity · `S` sensor ·
`P` planning · `O` ORCA · `L` labels.

Mouse: scroll to zoom (cursor-centered), left-drag to pan, right-click
a pedestrian to open the inspector panel.

## Performance

- All per-agent circles, labels, goal markers, and trails are created
  once and mutated (`set_data`/`set_center`/...) thereafter — never
  recreated per frame.
- Pedestrian heading and velocity arrows are drawn as **two shared
  `Quiver` collections** (one `set_offsets`/`set_UVC` call each per
  frame) rather than one `FancyArrow` patch per agent — this is the
  single biggest lever for scenes with hundreds of pedestrians.
- The robot has its own (recreated-per-frame) arrows since there's
  only ever one robot; the cost is negligible.
- Measured on this machine (headless `Agg` backend, no GPU): ~600
  pedestrians with labels/goals/trails on renders at roughly 10 FPS.
  Disabling `ids`/`labels` and trails roughly doubles that — text
  layout is the dominant cost at high agent counts, which is a
  Matplotlib-wide limitation rather than something specific to this
  renderer.

## Recording

```python
viz.save("episode.mp4", fps=20, dpi=150, n_frames=600)   # requires ffmpeg on PATH
viz.save("episode.gif", fps=15, n_frames=300)
viz.save_png_sequence("frames/", n_frames=300)
```

## Extending

Add a new layer by subclassing `renderers.base.Renderer`, wiring it
into `SceneRenderer`/`Visualizer`, and adding a toggle field to
`LayerToggles` if it should be keyboard-switchable. Add a new
research metric by adding a key to `panels/metrics.METRIC_KEYS` (and
populating it via `scene.extra["metrics"]`).
