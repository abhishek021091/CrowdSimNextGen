import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from navcore.entities.environment.environment import Environment
import numpy as np


class RobotVisualizer:
    def __init__(self, environment: Environment, ax):
        self.environment = environment
        self.ax = ax

    # Draw robot
    def _draw_robot(self) -> None:
        robot = self.environment.robot
        if robot.pose is not None:
            self.ax.add_patch(
                Circle(
                    (robot.pose.px, robot.pose.py),
                    radius=robot.radius,
                    fill=True,
                    color="yellow",
                    linewidth=2,
                )
            )

        if robot.goal is not None:
            self.ax.plot(
                robot.goal.gx,
                robot.goal.gy,
                marker="*",
                markersize=10,
                color="red",
                label="Goal",
            )

        if robot.sensor is not None and robot.pose is not None:
            sensor_range = robot.sensor.range
            self.ax.add_patch(
                Circle(
                    (robot.pose.px, robot.pose.py),
                    radius=sensor_range,
                    fill=False,
                    color="blue",
                    linestyle="--",
                    linewidth=1,
                )
            )

        if robot.velocity is not None and robot.pose is not None:
            u = np.cos(robot.pose.theta)
            v = np.sin(robot.pose.theta)

            self.ax.quiver(
                robot.pose.px,
                robot.pose.py,
                u,
                v,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.005,
                color="red",
            )

    def draw(self) -> None:
        self._draw_robot()
