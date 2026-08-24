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
    rand = np.random.default_rng(seed=env_config["random"]["seed"])

    def __init__(self):
        self.obstacles: list[Obstacle] = []

    def build_boundary(self):
        if self.config["boundary"]["gate"]["num_gates"] > 0:
            gateNum = self.config["boundary"]["gate"]["num_gates"]
            gates: list[BoundaryGate] = []
            edge_index = np.random.choice([0, 1, 2, 3])
            if edge_index == 0 or edge_index == 2:
                offset = np.random.uniform(0, self.env_config["arenaSize"]["width"])
            else:
                offset = np.random.uniform(
                    0, self.env_config["arenaSize"]["height"] / 2
                )
            width = np.random.uniform(1, 4)
            for _ in range(gateNum):
                gate = BoundaryGate(edge_index, offset, width)
                gates.append(gate)

            edges: tuple[Vector2, Vector2, Vector2, Vector2] = (
                Vector2(
                    -self.env_config["arenaSize"]["width"] / 2,
                    -self.env_config["arenaSize"]["height"] / 2,
                ),
                Vector2(
                    self.env_config["arenaSize"]["width"] / 2,
                    -self.env_config["arenaSize"]["height"] / 2,
                ),
                Vector2(
                    self.env_config["arenaSize"]["width"] / 2,
                    self.env_config["arenaSize"]["height"] / 2,
                ),
                Vector2(
                    -self.env_config["arenaSize"]["width"] / 2,
                    self.env_config["arenaSize"]["height"] / 2,
                ),
            )
            boundaryWall = Boundary(id="boundary", vertices=edges, gates=tuple(gates))
            self.obstacles.append(boundaryWall.to_obstacle())

    def build_table(self):
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
                    traversable=False,
                    visible=True,
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
                    traversable=False,
                    visible=True,
                )
            else:
                raise ValueError(f"Invalid table shape: {table_shape}")
            self.obstacles.append(table.to_obstacle())
