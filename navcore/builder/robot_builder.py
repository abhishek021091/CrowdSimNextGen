"""RobotBuilder: places the robot's episode start pose and goal.

Takes its random generator as a constructor argument rather than
seeding one internally — see module history for the reproducibility
fix this mirrors in CrowdSpawner/ObstacleBuilder.

Obstacle-aware placement:
    generate_pose()/generate_goal() reject any sample that lands inside
    a non-traversable obstacle's footprint and resample, up to
    MAX_PLACEMENT_ATTEMPTS times. This does NOT add clearance for the
    robot's own radius -- a point right at an obstacle's edge still
    passes -- see _inside_obstacle()'s docstring.
"""

from pathlib import Path

import numpy as np
import tomllib

import navcore.configs
from navcore.entities.agents.robot import Robot
from navcore.entities.components.geometry.containment import point_in_geometry
from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.components.goal import Goal
from navcore.entities.components.pose import Pose
from navcore.entities.obstacles.obstacle import Obstacle


class RobotBuilder:
    assert navcore.configs.__file__ is not None
    env_path = Path(navcore.configs.__file__).parent / "env.toml"
    with open(Path(env_path), "rb") as f:
        env_config = tomllib.load(f)

    #: Rejection-sampling cap for placing the robot's pose/goal outside
    #: every non-traversable obstacle. Bounded rather than an unbounded
    #: while-loop so a pathological obstacle layout (near-total arena
    #: coverage) fails loudly instead of hanging the builder.
    MAX_PLACEMENT_ATTEMPTS = 200

    def __init__(self, rand: np.random.Generator | None = None) -> None:
        self.rand = (
            rand
            if rand is not None
            else np.random.default_rng(seed=self.env_config["random"]["seed"])
        )
        self.robot: Robot

    def build_robot(self, obstacles: dict[str, Obstacle] | None = None) -> None:
        """Construct the robot and place its start pose and goal.

        Args:
            obstacles: This episode's static obstacles, if any. When
                given, both the start pose and the goal are resampled
                until they fall outside every non-traversable
                obstacle's footprint. ``None``/empty reproduces the
                previous obstacle-blind behavior (obstacle-free
                training configs).
        """
        robot = Robot()
        robot.set_state(
            self.generate_pose(obstacles),
            self.generate_goal(obstacles),
            robot.v_pref,
            robot.radius,
        )
        self.robot = robot

    def generate_pose(self, obstacles: dict[str, Obstacle] | None = None) -> Pose:
        width: float = self.env_config["arenaSize"]["width"]
        height: float = self.env_config["arenaSize"]["height"]

        for _ in range(self.MAX_PLACEMENT_ATTEMPTS):
            theta: float = self.rand.uniform(0, 2 * np.pi)
            px: float = self.rand.uniform(-width / 2, width / 2)
            py: float = self.rand.uniform(-height / 2, height / 2)
            if not self._inside_obstacle(Vector2(px, py), obstacles):
                return Pose(px, py, theta)

        raise RuntimeError(
            f"RobotBuilder.generate_pose() failed to find a pose clear of "
            f"every obstacle after {self.MAX_PLACEMENT_ATTEMPTS} attempts -- "
            f"obstacle layout may cover too much of the arena."
        )

    def generate_goal(self, obstacles: dict[str, Obstacle] | None = None) -> Goal:
        """Sample a goal clear of every non-traversable obstacle *and* of
        the arena boundary edge.

        Obstacle clearance was already handled by `_inside_obstacle`, but
        that check explicitly skips the "boundary" key (see its own
        docstring -- boundary's Polygon is the allowed interior region,
        not a forbidden footprint). That meant a goal could legally land
        right up against the arena wall. `CollisionChecker` uses inverted
        semantics for the boundary (solid on the *outside*, and also
        flags any point within `robot.radius + safety_distance` of the
        edge from the inside) -- so a wall-hugging goal could get the
        robot flagged as colliding on the very tick it reaches its own
        goal. Shrinking the sampling rectangle by that same clearance
        up front prevents that case by construction, rather than
        rejecting after the fact.
        """
        width: float = self.env_config["arenaSize"]["width"]
        height: float = self.env_config["arenaSize"]["height"]
        clearance = self._boundary_clearance()

        half_width = width / 2 - clearance
        half_height = height / 2 - clearance
        if half_width <= 0.0 or half_height <= 0.0:
            raise RuntimeError(
                f"RobotBuilder.generate_goal(): arena is too small for the "
                f"required boundary clearance ({clearance!r}) -- "
                f"arenaSize={width}x{height}."
            )

        for _ in range(self.MAX_PLACEMENT_ATTEMPTS):
            gx: float = self.rand.uniform(-half_width, half_width)
            gy: float = self.rand.uniform(-half_height, half_height)
            if not self._inside_obstacle(Vector2(gx, gy), obstacles):
                return Goal(gx, gy)

        raise RuntimeError(
            f"RobotBuilder.generate_goal() failed to find a goal clear of "
            f"every obstacle after {self.MAX_PLACEMENT_ATTEMPTS} attempts -- "
            f"obstacle layout may cover too much of the arena."
        )

    @staticmethod
    def _boundary_clearance() -> float:
        """Minimum distance a goal must be kept from the arena boundary edge.

        `robot.radius + env.info.safety_distance` is exactly the margin
        `CollisionChecker.check_collision` adds when testing a point
        against the "boundary" obstacle -- see its `effective_radius`
        computation. Read from the class-level configs directly (not an
        instance's `self.robot`, which doesn't exist yet at the point
        `generate_goal` is called from `build_robot`) since neither the
        robot's radius nor the safety distance is ever randomized.
        """
        robot_radius = float(Robot.config["physical"]["radius"])
        safety_distance = float(RobotBuilder.env_config["safety"]["distance"])
        return robot_radius + safety_distance

    @staticmethod
    def _inside_obstacle(point: Vector2, obstacles: dict[str, Obstacle] | None) -> bool:
        """Return whether `point` falls inside a non-traversable obstacle.

        The "boundary" entry (see ObstacleBuilder.build_boundary /
        Boundary.to_obstacle) is excluded here on purpose. Its geometry is
        the full interior polygon enclosed by the arena walls -- the
        *allowed* region, not a forbidden one -- so testing it the same way
        as a Table/Pillar rejects every point in the entire arena.
        boustropheden.decompose() already excludes the same key for the
        same reason (its own `interior_obstacles` filter), treating
        "boundary" as the positive free-space region instead of a solid
        obstacle. This mirrors that existing convention rather than
        inventing a second way to recognize "this Obstacle actually denotes
        the containing region, not a forbidden one."

        Traversable obstacles are also skipped -- an agent may legally
        occupy that space. No clearance margin is added for the robot's own
        radius (see module docstring); `containment.py` would need an
        inflated/Minkowski-sum test to support that for `Polygon`, which it
        doesn't have yet.
        """
        if not obstacles:
            return False
        for key, obstacle in obstacles.items():
            if key == "boundary":
                continue
            if obstacle.traversable:
                continue
            if point_in_geometry(point, obstacle.geometry):
                return True
        return False

    def __repr__(self) -> str:
        return f"RobotBuilder(rand={self.rand}, robot={self.robot})"
