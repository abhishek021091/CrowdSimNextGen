from navcore.configs import obstacle
from navcore.entities.components.geometry import Vector2
from navcore.entities.obstacles.boundary import Boundary, BoundaryGate
from navcore.entities.obstacles.pillar import Pillar
from navcore.entities.obstacles.table import Table, TableShape
from navcore.entities.obstacles.wall import Wall


class ObstacleBuilder:
    def __init__(self):
        self.obstacle_config = obstacle
        if self.obstacle_config["boundary"]["boundary"]:
            self.boundary = Boundary(
                self.obstacle_config["boundary"]["width"],
                self.obstacle_config["boundary"]["height"],
            )

    def generate_pillars(self):
        pillars = []
        for i in range(self.obstacle_config["pillar"]["num_pillars"]):
            pillar = Pillar(
                id=f"pillar_{i}",
                center=Vector2(
                    self.rng.uniform(
                        -self.env_config["arena_width"] / 2,
                        self.env_config["arena_width"] / 2,
                    ),
                    self.rng.uniform(
                        -self.env_config["arena_height"] / 2,
                        self.env_config["arena_height"] / 2,
                    ),
                ),
                radius=self.obstacle_config["pillar"]["radius"],
            )
            pillars.append(pillar)
        return pillars

    def generate_walls(self):
        walls = []
        for i in range(self.obstacle_config["wall"]["num_walls"]):
            wall = Wall(
                id=f"wall_{i}",
                start=Vector2(
                    self.rng.uniform(
                        -self.env_config["arena_width"] / 2,
                        self.env_config["arena_width"] / 2,
                    ),
                    self.rng.uniform(
                        -self.env_config["arena_height"] / 2,
                        self.env_config["arena_height"] / 2,
                    ),
                ),
                end=Vector2(
                    self.rng.uniform(
                        -self.env_config["arena_width"] / 2,
                        self.env_config["arena_width"] / 2,
                    ),
                    self.rng.uniform(
                        -self.env_config["arena_height"] / 2,
                        self.env_config["arena_height"] / 2,
                    ),
                ),
            )
            walls.append(wall)
        return wall

    def generate_boundaries(self):
        if self.obstacle_config["boundary"]["boundary"]:
            gates = []
            for i in range(
                self.obstacle_config["boundary"]["boundary_gates"]["num_gates"]
            ):
                gate = self.generate_boudary_gates(
                    gate_positions=self.obstacle_config["boundary"]["boundary_gates"][
                        "positions"
                    ][i]
                )
                gates.append(gate)

            boundary = Boundary(
                id="arena_boundary",
                vertices=[
                    Vector2(
                        -self.env_config["arena_width"] / 2,
                        -self.env_config["arena_height"] / 2,
                    ),
                    Vector2(
                        self.env_config["arena_width"] / 2,
                        -self.env_config["arena_height"] / 2,
                    ),
                    Vector2(
                        self.env_config["arena_width"] / 2,
                        self.env_config["arena_height"] / 2,
                    ),
                    Vector2(
                        -self.env_config["arena_width"] / 2,
                        self.env_config["arena_height"] / 2,
                    ),
                ],
                gates=tuple(gates),
            )
            return boundary

    def generate_boudary_gates(self, gate_positions):
        right_up_edge = (
            self.env_config["arena_width"] / 2,
            self.env_config["arena_height"] / 2,
        )
        right_down_edge = (
            self.env_config["arena_width"] / 2,
            -self.env_config["arena_height"] / 2,
        )
        left_up_edge = (
            -self.env_config["arena_width"] / 2,
            self.env_config["arena_height"] / 2,
        )
        left_down_edge = (
            -self.env_config["arena_width"] / 2,
            -self.env_config["arena_height"] / 2,
        )
        gate_positions = self.obstacle_config["boundary"]["gates"]["positions"]

        if gate_positions[0] > left_up_edge[0] and gate_positions[0] < right_up_edge[0]:
            if (
                gate_positions[1] > right_down_edge[1]
                and gate_positions[1] < right_up_edge[1]
            ):
                edge_index = 1
            if (
                gate_positions[1] > left_down_edge[1]
                and gate_positions[1] < left_up_edge[1]
            ):
                edge_index = 3
            if gate_positions[1] == right_up_edge[1]:
                edge_index = 2
            if gate_positions[1] == left_up_edge[1]:
                edge_index = 1

        return BoundaryGate(
            edge_index=edge_index,
            offset=self.obstacle_config["boundary"]["gates"]["offset"],
            width=self.obstacle_config["boundary"]["gates"]["width"],
        )

    #TODO: Add table generation function
    def table(self):
        tables = []
        for i in range(self.obstacle_config["table"]["num_tables"]):
            shape = self.obstacle_config["table"]["shape"]
            table = Table.shape(
                id=f"table_{i}",
                center=Vector2(self.obstacle_config["table"]["center_x"], self.obstacle_config["table"]["center_y"]
                ),
                shape=TableShape[self.obstacle_config["table"]["shape"]],
                width=self.obstacle_config["table"]["width"],
                height=self.obstacle_config["table"]["height"],
            )
            tables.append(table)
        return tables
