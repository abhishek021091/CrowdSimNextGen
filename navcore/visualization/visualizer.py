from pathlib import Path

import matplotlib.pyplot as plt
import tomllib
from matplotlib.animation import FuncAnimation

import navcore.configs
from navcore.boustropheden.boustropheden import DecompositionResult
from navcore.entities.environment.environment import Environment
from navcore.visualization.entities.cell_visualizer import CellVisualizer
from navcore.visualization.entities.crowd_visualizer import CrowdVisualizer
from navcore.visualization.entities.obstacle_visualizer import ObstacleVisualizer
from navcore.visualization.entities.robot_visualizer import RobotVisualizer


class Visualizer:
    def __init__(self) -> None:
        env_path = Path(navcore.configs.__file__).parent / "env.toml"
        with open(env_path, "rb") as f:
            arena = tomllib.load(f)["arenaSize"]
        self.arena_width = arena["width"]
        self.arena_height = arena["height"]

        self.env: Environment | None = None
        # Sticky decomposition overlay: a decomposition is computed once
        # per episode (see boustropheden.decompose), not per tick, so
        # callers driving a per-tick refresh() loop (GlobalPlanner,
        # test_sweep.py-style scripts) shouldn't have to keep re-passing
        # the same DecompositionResult on every single call. Passing an
        # explicit `decomposition=` to visualize()/refresh() always wins
        # and updates this; passing nothing keeps drawing whatever was
        # set last, if anything.
        self.decomposition: DecompositionResult | None = None
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

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

        self.ax.clear()
        self.ax.set_xlim(-self.arena_width / 2 - 0.5, self.arena_width / 2 + 0.5)
        self.ax.set_ylim(-self.arena_height / 2 - 0.5, self.arena_height / 2 + 0.5)
        self.ax.set_aspect("equal")

        if self.decomposition is not None:
            CellVisualizer(self.decomposition, self.ax).draw()

        CrowdVisualizer(env, self.ax).draw()
        ObstacleVisualizer(env, self.ax).draw()
        RobotVisualizer(env, self.ax, mission=mission).draw()

    def refresh(
        self,
        env: Environment,
        mission=None,
        decomposition: DecompositionResult | None = None,
    ) -> None:
        self.visualize(env, mission=mission, decomposition=decomposition)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)  # Allow the GUI event loop to process events

    def animate(
        self, env: Environment, n_frames: int = 30, interval: int = 100
    ) -> FuncAnimation:
        self.env = env
        self.ani = FuncAnimation(
            self.fig, self.refresh, frames=range(n_frames), interval=interval
        )
        plt.show(block=False)
