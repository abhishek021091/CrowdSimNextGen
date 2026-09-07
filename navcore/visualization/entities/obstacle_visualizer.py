from matplotlib.patches import Circle, Polygon, Rectangle

from navcore.entities.components.geometry.circle import Circle as CircleGeometry
from navcore.entities.components.geometry.polygon import Polygon as PolygonGeometry
from navcore.entities.components.geometry.rectangle import (
    Rectangle as RectangleGeometry,
)
from navcore.entities.environment.environment import Environment


class ObstacleVisualizer:
    def __init__(self, environment: Environment, ax):
        self.environment = environment
        self.ax = ax

    # Draw obstacles
    def _draw_obstacles(self) -> None:
        obstacles = self.environment.obstacles
        for obstacle in obstacles.values():
            if isinstance(obstacle.geometry, CircleGeometry):
                self.ax.add_patch(
                    Circle(
                        (obstacle.geometry.center.x, obstacle.geometry.center.y),
                        radius=obstacle.geometry.radius,
                        fill=True,
                        color="brown",
                    )
                )
            elif isinstance(obstacle.geometry, RectangleGeometry):
                self.ax.add_patch(
                    Rectangle(
                        (
                            obstacle.geometry.center.x - obstacle.geometry.width / 2,
                            obstacle.geometry.center.y - obstacle.geometry.height / 2,
                        ),
                        width=obstacle.geometry.width,
                        height=obstacle.geometry.height,
                        fill=True,
                        color="brown",
                    )
                )
            elif isinstance(obstacle.geometry, PolygonGeometry):
                vertices = [
                    (vertex.x, vertex.y) for vertex in obstacle.geometry.vertices
                ]
                self.ax.add_patch(
                    Polygon(
                        vertices,
                        fill=True,
                        color="brown",
                    )
                )

    def draw(self) -> None:
        self._draw_obstacles()
