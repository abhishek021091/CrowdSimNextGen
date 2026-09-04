"""ObstacleBuilder: places the episode's static obstacles.

Takes its random generator as a constructor argument for the same
reason as ``RobotBuilder`` -- see that module's docstring. Only the
RNG-sourcing changed here; placement logic is unchanged.
"""

from pathlib import Path

import numpy as np
import tomllib

import navcore.configs
from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.obstacles.boundary import Boundary, BoundaryGate
from navcore.entities.obstacles.obstacle import Obstacle
from navcore.entities.obstacles.table import Table


class ObstacleBuilder:
    assert navcore.configs.__file__ is not None
    config_path = Path(navcore.configs.__file__).parent / "obstacle.toml"
    env_path = Path(navcore.configs.__file__).parent / "env.toml"
    with open(Path(config_path), "rb") as f:
        config = tomllib.load(f)
    with open(Path(env_path), "rb") as f:
        env_config = tomllib.load(f)

    def __init__(self, rand: np.random.Generator | None = None) -> None:
        self.rand = (
            rand
            if rand is not None
            else np.random.default_rng(seed=self.env_config["random"]["seed"])
        )
        self.obstacles: dict[str, Obstacle] = {}

    def build_boundary(self) -> None:
        if self.config["boundary"]["gate"]["num_gates"] > 0:
            gate_num = self.config["boundary"]["gate"]["num_gates"]
            gates: list[BoundaryGate] = []
            edge_index = int(self.rand.choice([0, 1, 2, 3]))
            if edge_index == 0 or edge_index == 2:
                offset: float = self.rand.uniform(
                    0.0, float(self.env_config["arenaSize"]["width"]) / 2
                )
            else:
                offset: float = self.rand.uniform(
                    0.0, float(self.env_config["arenaSize"]["height"]) / 2
                )
            width: float = self.rand.uniform(1.0, 4.0)
            for _ in range(gate_num):
                gates.append(BoundaryGate(edge_index, offset, width))

            w: float = self.env_config["arenaSize"]["width"] / 2
            h: float = self.env_config["arenaSize"]["height"] / 2
            edges = (Vector2(-w, -h), Vector2(w, -h), Vector2(w, h), Vector2(-w, h))
            boundary_wall = Boundary(id="boundary", vertices=edges, gates=tuple(gates))
            self.obstacles["boundary"] = boundary_wall.to_obstacle()

    def build_table(self) -> None:
        num__circular_tables = self.config["circular_table"]["num_tables"]
        num__rectangular_tables = self.config["rectangular_table"]["num_tables"]

        for i in range(num__circular_tables):
            table_id = f"table_{i}"
            table_radius = float(self.config["circular_table"]["radius"])
            if self.config["circular_table"]["randomize_radius"]:
                table_radius = table_radius + self.rand.uniform(-0.2, 0.2)
            table_center = Vector2(
                self.rand.uniform(
                    -float(self.env_config["arenaSize"]["width"]) / 2 + table_radius,
                    float(self.env_config["arenaSize"]["width"]) / 2 - table_radius,
                ),
                self.rand.uniform(
                    -float(self.env_config["arenaSize"]["height"]) / 2 + table_radius,
                    float(self.env_config["arenaSize"]["height"]) / 2 - table_radius,
                ),
            )
            table = Table.circular(
                id=table_id,
                center=table_center,
                radius=table_radius,
                name=f"Table {i}",
            )
            self.obstacles[table_id] = table.to_obstacle()
        for i in range(num__rectangular_tables):
            table_id = f"table_{i + num__circular_tables}"
            table_width = float(self.config["rectangular_table"]["width"])
            table_height = float(self.config["rectangular_table"]["height"])
            if self.config["rectangular_table"]["randomize_dimensions"]:
                table_width = table_width + self.rand.uniform(-0.2, 0.2)
                table_height = table_height + self.rand.uniform(-0.2, 0.2)
            table_center = Vector2(
                self.rand.uniform(
                    -float(self.env_config["arenaSize"]["width"]) / 2 + table_width / 2,
                    float(self.env_config["arenaSize"]["width"]) / 2 - table_width / 2,
                ),
                self.rand.uniform(
                    -float(self.env_config["arenaSize"]["height"]) / 2
                    + table_height / 2,
                    float(self.env_config["arenaSize"]["height"]) / 2
                    - table_height / 2,
                ),
            )

            table = Table.rectangular(
                id=table_id,
                center=table_center,
                width=table_width,
                height=table_height,
                name=f"Table {i}",
            )

            self.obstacles[table_id] = table.to_obstacle()
