"""CrowdSimEnv: Gymnasium environment for training the robot with RL.

Task-agnostic: the caller injects a Task (see task.py), which owns the
robot's Mission and defines reward/termination. CrowdSimEnv only owns
episode lifecycle (build/reset the Environment, drive Step, encode
observations, enforce the step limit) and action decoding.

Two action modes (ActionMode):
    VELOCITY: the policy outputs (vx, vy) directly; Step is told to use
        it verbatim for the robot, bypassing the ORCA planner entirely
        for the robot's own motion. Standard for CrowdNav-style
        benchmarks where collision avoidance is meant to be learned.
    WAYPOINT: the policy outputs a target point; that point becomes an
        RLWaypointMission's target, and Step still routes the robot
        through the same ORCA planner as any other agent. Use this to
        study RL-for-navigation-goals with ORCA as a fixed local safety
        layer, rather than learning collision avoidance from scratch.

Pedestrians are unaffected by action_mode either way -- they always go
through DecentralizedORCAPlanner, exactly as in the non-RL test
scripts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from navcore.builder.environment_builder import EnvironmentBuilder
from navcore.entities.agents.robot import Robot
from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.components.velocity import Velocity
from navcore.entities.environment.environment import Environment
from navcore.gym_wrapper.observation_encoder import ObservationEncoder
from navcore.gym_wrapper.rl_missions import RLWaypointMission
from navcore.gym_wrapper.task import Task
from navcore.middleware.orca_middleware import DecentralizedORCAPlanner
from navcore.step.step import Step


class ActionMode(Enum):
    VELOCITY = "velocity"
    WAYPOINT = "waypoint"


@dataclass(slots=True, frozen=True)
class CrowdSimEnvConfig:
    """Tunables for CrowdSimEnv that aren't part of the injected Task.

    Attributes:
        action_mode: See module docstring.
        max_neighbors: Fixed neighbor-slot count for observations.
        history_steps: Temporal frames retained for each visible neighbor.
        max_episode_steps: Truncation limit; independent of the Task's
            own termination logic.
        robot_visible: Whether pedestrians can see the robot in their
            own sensor observations this episode.
        include_static_obstacles: Whether to build static obstacles for
            this scenario. CrowdNav-style goal walking defaults to an
            obstacle-free arena; non-RL EnvironmentBuilder callers retain
            their existing obstacle-filled default.
        orca_config_file: ORCA reasoning-parameter TOML, forwarded to
            DecentralizedORCAPlanner (see its own docstring).
    """

    action_mode: ActionMode = ActionMode.VELOCITY
    max_neighbors: int = 10
    history_steps: int = 8
    max_episode_steps: int = 500
    robot_visible: bool = False
    include_static_obstacles: bool = False
    orca_config_file: str = "orca.toml"


class CrowdSimEnv(gym.Env):
    """Gymnasium environment wrapping navcore's crowd simulation.

    Rendering is deliberately out of scope here -- construct one of
    the project's Visualizer classes yourself against ``env.env`` if
    you want to watch an episode; keeping that decoupled means training
    never pays for rendering cost. See module docstring for why this
    stays task-agnostic.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, task: Task, config: CrowdSimEnvConfig | None = None) -> None:
        super().__init__()
        self.task = task
        self.config = config if config is not None else CrowdSimEnvConfig()

        self._env_builder = EnvironmentBuilder(
            include_static_obstacles=self.config.include_static_obstacles
        )
        self._obs_encoder = ObservationEncoder(
            max_neighbors=self.config.max_neighbors,
            history_steps=self.config.history_steps,
        )
        self._waypoint_mission: RLWaypointMission | None = None
        self._step_driver: Step | None = None
        self._velocity_override: Velocity | None = None
        self._elapsed_steps = 0

        self.observation_space = self._obs_encoder.space
        self.action_space = self._build_action_space()

        self.env: Environment  # assigned in reset(); no episode before that

    def _build_action_space(self) -> gym.Space:
        if self.config.action_mode is ActionMode.VELOCITY:
            v_max = float(Robot.config["kinematics"]["v_pref"])
            return spaces.Box(low=-v_max, high=v_max, shape=(2,), dtype=np.float32)

        # WAYPOINT: normalized [-1, 1] per axis, scaled to arena
        # half-extents at decode time -- keeps the action space's scale
        # independent of arena size, which matters if arenaSize ever
        # varies across scenarios/curricula.
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        episode_seed = (
            seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        )
        self.env = self._env_builder.reset(random_seed=episode_seed)
        self.task.reset(self.env)
        self._obs_encoder.reset()

        robot_mission = None
        self._waypoint_mission = None
        if self.config.action_mode is ActionMode.WAYPOINT:
            assert self.env.robot.pose is not None
            self._waypoint_mission = RLWaypointMission(
                initial_target=Vector2(self.env.robot.pose.px, self.env.robot.pose.py)
            )
            robot_mission = self._waypoint_mission

        planner = DecentralizedORCAPlanner(
            config_file=self.config.orca_config_file,
            # obstacles=self.env.obstacles,
        )
        self._step_driver = Step(
            planner=planner,
            env=self.env,
            robot_visible=self.config.robot_visible,
            robot_mission=robot_mission,
        )
        self._velocity_override = None
        self._elapsed_steps = 0

        return self._obs_encoder.encode(self.env), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._step_driver is None:
            raise RuntimeError("CrowdSimEnv.step() called before reset().")

        self._apply_action(np.asarray(action, dtype=np.float32))
        step_result = self._step_driver.step(
            robot_velocity_override=self._velocity_override
        )
        collided = self.env.did_collision_happened()

        observation = self._obs_encoder.encode(self.env)
        reward = self.task.reward(self.env, collided)
        terminated = self.task.is_terminated(self.env, collided)

        self._elapsed_steps += 1
        truncated = self._elapsed_steps >= self.config.max_episode_steps

        info: dict[str, Any] = {
            "collision": collided,
            "robot_reached_goal": step_result.robot_reached_goal,
        }
        return observation, reward, terminated, truncated, info

    def _apply_action(self, action: np.ndarray) -> None:
        if self.config.action_mode is ActionMode.VELOCITY:
            self._velocity_override = self._decode_velocity_action(action)
            return

        self._velocity_override = None
        assert self._waypoint_mission is not None
        self._waypoint_mission.set_target(self._decode_waypoint_action(action))

    @staticmethod
    def _decode_velocity_action(action: np.ndarray) -> Velocity:
        v_max = float(Robot.config["kinematics"]["v_pref"])
        vx, vy = float(action[0]), float(action[1])
        speed = math.hypot(vx, vy)
        if speed > v_max:
            # Box gives independent per-axis bounds, so (v_max, v_max)
            # is a legal action with speed v_max*sqrt(2). Clip the
            # resulting vector's magnitude rather than each axis
            # separately, so direction is preserved.
            scale = v_max / speed
            vx, vy = vx * scale, vy * scale
        return Velocity(vx, vy)

    @staticmethod
    def _decode_waypoint_action(action: np.ndarray) -> Vector2:
        arena = EnvironmentBuilder.env_config["arenaSize"]
        half_width = float(arena["width"]) / 2.0
        half_height = float(arena["height"]) / 2.0
        return Vector2(float(action[0]) * half_width, float(action[1]) * half_height)
