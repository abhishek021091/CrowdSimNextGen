import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import navcore.configs
from navcore.entities.environment.environment import Environment
from navcore.visualization_1.entities.crowd_visualizer import CrowdVisualizer
from navcore.visualization_1.entities.obstacle_visualizer import ObstacleVisualizer
from navcore.visualization_1.entities.robot_visualizer import RobotVisualizer


class Visualizer:
    def __init__(self) -> None:
        env_path = Path(navcore.configs.__file__).parent / "env.toml"
        with open(env_path, "rb") as f:
            arena = tomllib.load(f)["arenaSize"]
        self.arena_width = arena["width"]
        self.arena_height = arena["height"]

        self.env: Environment | None = None
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

    def visualize(self, env: Environment) -> None:
        """Draw the environment, robot, obstacles, and pedestrians."""
        self.env = env
        self.ax.clear()
        self.ax.set_xlim(-self.arena_width / 2 - 0.5, self.arena_width / 2 + 0.5)
        self.ax.set_ylim(-self.arena_height / 2 - 0.5, self.arena_height / 2 + 0.5)
        self.ax.set_aspect("equal")

        CrowdVisualizer(env, self.ax).draw()
        ObstacleVisualizer(env, self.ax).draw()
        RobotVisualizer(env, self.ax).draw()

    def refresh(self, env: Environment) -> None:
        self.visualize(env)
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
