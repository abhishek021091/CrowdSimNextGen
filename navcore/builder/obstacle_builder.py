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

    def __init__(self, rand: np.random.Generator) -> None:
        self.rand = rand
        self.obstacles: list[Obstacle] = []

    def build_boundary(self) -> None:
        if self.config["boundary"]["gate"]["num_gates"] > 0:
            gate_num = self.config["boundary"]["gate"]["num_gates"]
            gates: list[BoundaryGate] = []
            edge_index = int(self.rand.choice([0, 1, 2, 3]))
            if edge_index == 0 or edge_index == 2:
                offset = self.rand.uniform(0, self.env_config["arenaSize"]["width"])
            else:
                offset = self.rand.uniform(
                    0, self.env_config["arenaSize"]["height"] / 2
                )
            width = self.rand.uniform(1, 4)
            for _ in range(gate_num):
                gates.append(BoundaryGate(edge_index, offset, width))

            w = self.env_config["arenaSize"]["width"] / 2
            h = self.env_config["arenaSize"]["height"] / 2
            edges = (Vector2(-w, -h), Vector2(w, -h), Vector2(w, h), Vector2(-w, h))
            boundary_wall = Boundary(id="boundary", vertices=edges, gates=tuple(gates))
            self.obstacles.append(boundary_wall.to_obstacle())

    def build_table(self) -> None:
        num_tables = self.config["table"]["num_tables"]
        table_shape = self.config["table"]["shape"]

        for i in range(num_tables):
            table_id = f"table_{i}"
            if table_shape == "circular":
                table_radius = self.config["table"]["radius"]
                if self.config["table"]["randomize_radius"]:
                    table_radius = table_radius + self.rand.uniform(-0.5, 0.5)
                table_center = Vector2(
                    self.rand.uniform(
                        -self.env_config["arenaSize"]["width"] / 2 + table_radius,
                        self.env_config["arenaSize"]["width"] / 2 - table_radius,
                    ),
                    self.rand.uniform(
                        -self.env_config["arenaSize"]["height"] / 2 + table_radius,
                        self.env_config["arenaSize"]["height"] / 2 - table_radius,
                    ),
                )
                table = Table.circular(
                    id=table_id,
                    center=table_center,
                    radius=table_radius,
                    name=f"Table {i}",
                )
            elif table_shape == "rectangular":
                table_width = self.config["table"]["width"]
                table_height = self.config["table"]["height"]
                if self.config["table"]["randomize_width_height"]:
                    table_width = table_width + self.rand.uniform(-0.5, 0.5)
                    table_height = table_height + self.rand.uniform(-0.5, 0.5)
                table_center = Vector2(
                    self.rand.uniform(
                        -self.env_config["arenaSize"]["width"] / 2 + table_width / 2,
                        self.env_config["arenaSize"]["width"] / 2 - table_width / 2,
                    ),
                    self.rand.uniform(
                        -self.env_config["arenaSize"]["height"] / 2 + table_height / 2,
                        self.env_config["arenaSize"]["height"] / 2 - table_height / 2,
                    ),
                )
                table = Table.rectangular(
                    id=table_id,
                    center=table_center,
                    width=table_width,
                    height=table_height,
                    name=f"Table {i}",
                )
            else:
                raise ValueError(f"Invalid table shape: {table_shape}")
            self.obstacles.append(table.to_obstacle())
