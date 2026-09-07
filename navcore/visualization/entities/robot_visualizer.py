import numpy as np
from matplotlib.patches import Circle, Rectangle

from navcore.entities.environment.environment import Environment
from navcore.missions.sweeping import SweepingMission


class RobotVisualizer:
    def __init__(
        self, environment: Environment, ax, mission: SweepingMission | None = None
    ):
        self.environment = environment
        self.ax = ax
        self.mission = mission

    def _draw_swept_area(self) -> None:
        """Highlight the sweep's coverage region so far.

        Deliberately a single bounding rectangle from the sweep's
        starting corner to the current lane position, not a real
        coverage grid -- this mirrors what SweepingMission itself
        tracks (lanes_completed + partial lane progress via
        total_area_swept()), so the overlay never claims more precision
        than the mission's own bookkeeping actually has. A true
        cell-by-cell overlay belongs with the planned CoverageGrid
        integration, not here.
        """
        mission = self.mission
        if mission is None or not mission.started:
            return

        robot_goal = self.environment.robot.goal
        if robot_goal is None:
            return

        half_width = float(self.environment.info.arena_width) / 2
        half_height = float(self.environment.info.arena_height) / 2

        if mission.sweep_axes == 0:
            y_min, y_max = sorted((mission.sweep_start[1], robot_goal.gy))
            x_min, x_max = -half_width, half_width
        else:
            x_min, x_max = sorted((mission.sweep_start[0], robot_goal.gx))
            y_min, y_max = -half_height, half_height

        self.ax.add_patch(
            Rectangle(
                (x_min, y_min),
                width=x_max - x_min,
                height=y_max - y_min,
                fill=True,
                color="skyblue",
                alpha=0.3,
                zorder=0,  # behind crowd/robot/obstacles
            )
        )

    def _draw_safe_point(self) -> None:
        mission = self.mission
        if mission is None or mission.current_safe_point is None:
            return
        sx, sy = mission.current_safe_point
        self.ax.plot(
            sx,
            sy,
            marker="x",
            markersize=10,
            markeredgewidth=2,
            color="green",
        )
        self.ax.text(
            sx, sy, "Safe pt", fontsize=7, ha="center", va="bottom", color="green"
        )

    def _draw_robot(self) -> None:
        robot = self.environment.robot
        avoiding = self.mission is not None and self.mission.avoiding_obstacle
        robot_color = "red" if avoiding else "yellow"
        velocity_color = "yellow" if avoiding else "red"

        if robot.pose is not None:
            self.ax.add_patch(
                Circle(
                    (robot.pose.px, robot.pose.py),
                    radius=robot.radius,
                    fill=True,
                    color=robot_color,
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
            self.ax.add_patch(
                Circle(
                    (robot.pose.px, robot.pose.py),
                    radius=robot.sensor.range,
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
                color=velocity_color,
            )

    def draw(self) -> None:
        self._draw_swept_area()
        self._draw_robot()
        self._draw_safe_point()
