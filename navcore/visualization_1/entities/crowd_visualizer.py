from matplotlib.axes import Axes
from matplotlib.patches import Circle

from navcore.entities.environment.environment import Environment


class CrowdVisualizer:
    def __init__(self, environment: Environment, ax: Axes):
        self.environment = environment
        self.ax = ax

    def _draw_crowd(self) -> None:
        for person in self.environment.crowd.values():
            if person.pose is None:
                continue

            color = "purple" if person.group_id is not None else "blue"

            self.ax.add_patch(
                Circle(
                    (person.pose.px, person.pose.py),
                    radius=person.radius,
                    fill=True,
                    color=color,
                    linewidth=2,
                )
            )
            text: str = (
                f"{person.id}, Grp: {person.group_id}"
                if person.group_id is not None
                else f"{person.id}"
            )
            self.ax.text(
                person.pose.px,
                person.pose.py,
                text,
                fontsize=8,
                ha="center",
                va="center",
                color="white",
            )

            if person.goal is not None:
                self.ax.plot(
                    person.goal.gx,
                    person.goal.gy,
                    marker="*",
                    markersize=10,
                    color="blue",
                    label="Goal",
                )

                self.ax.text(
                    person.goal.gx,
                    person.goal.gy,
                    f"Goal {person.id}",
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="black",
                )

    def draw(self) -> None:
        self._draw_crowd()
