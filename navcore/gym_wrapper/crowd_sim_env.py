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
from navcore.entities.components.sensors.obstacle_detector import ObstacleDetectorConfig
from navcore.entities.components.velocity import Velocity
from navcore.entities.environment.environment import Environment
from navcore.gym_wrapper.observation_encoder import ObservationEncoder
from navcore.gym_wrapper.rl_missions import RLWaypointMission
from navcore.gym_wrapper.task import Task
from navcore.middleware.orca_middleware import DecentralizedORCAPlanner
from navcore.step.step import Step
from navcore.visualization.visualizer import Visualizer


class ActionMode(Enum):
    VELOCITY = "velocity"
    WAYPOINT = "waypoint"


@dataclass(slots=True, frozen=True)
class CrowdSimEnvConfig:
    action_mode: ActionMode = ActionMode.VELOCITY
    max_neighbors: int = 10
    history_steps: int = 8
    max_episode_steps: int = 1500
    robot_visible: bool = False
    include_static_obstacles: bool = True
    orca_config_file: str = "orca.toml"
    obstacle_num_rays: int = 60
    obstacle_max_range: float = 5.0


class CrowdSimEnv(gym.Env[dict[str, Any], ActionMode]):
    """Gymnasium environment wrapping navcore's crowd simulation.

    Rendering is deliberately out of scope here -- construct one of
    the project's Visualizer classes yourself against ``env.env`` if
    you want to watch an episode; keeping that decoupled means training
    never pays for rendering cost. See module docstring for why this
    stays task-agnostic.
    """

    metadata: dict[str, Any] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        task: Task,
        config: CrowdSimEnvConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.task = task
        self.config = config if config is not None else CrowdSimEnvConfig()

        self.render_mode = render_mode
        self.visualizer: Visualizer | None = None

        self._env_builder = EnvironmentBuilder(
            include_static_obstacles=self.config.include_static_obstacles
        )
        self._obs_encoder = ObservationEncoder(
            max_neighbors=self.config.max_neighbors,
            history_steps=self.config.history_steps,
            obstacle_detector_config=ObstacleDetectorConfig(
                num_rays=self.config.obstacle_num_rays,
                max_range=self.config.obstacle_max_range,
            ),
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
        self._obs_encoder.reset(self.env)

        robot_mission = None
        self._waypoint_mission = None
        if self.config.action_mode is ActionMode.WAYPOINT:
            assert self.env.robot.pose is not None
            self._waypoint_mission = RLWaypointMission(
                initial_target=Vector2(self.env.robot.pose.px, self.env.robot.pose.py)
            )
            robot_mission = self._waypoint_mission

        robot_planner = None
        if self.config.action_mode is ActionMode.WAYPOINT:
            robot_planner = DecentralizedORCAPlanner(
                config_file=self.config.orca_config_file,
                obstacles=self.env.obstacles,
            )
        crowd_planner = DecentralizedORCAPlanner(
            config_file=self.config.orca_config_file,
        )
        self._step_driver = Step(
            crowd_planner=crowd_planner,
            env=self.env,
            robot_visible=self.config.robot_visible,
            robot_planner=robot_planner,
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
        self._respawn_pedestrians(step_result)
        collided = self.env.did_collision_happened()
        out_of_bounds = self.env.out_of_bounds()

        observation = self._obs_encoder.encode(self.env)
        reward = self.task.reward(self.env, collided, out_of_bounds)
        terminated = self.task.is_terminated(self.env, collided, out_of_bounds)

        self._elapsed_steps += 1
        truncated = self._elapsed_steps >= self.config.max_episode_steps
        # if truncated:
        #     print(
        #         f"Episode truncated after {self._elapsed_steps} steps (max_episode_steps={self.config.max_episode_steps})."
        #     )

        info: dict[str, Any] = {
            "collision": collided,
            "out_of_bounds": out_of_bounds,
            "truncated": truncated,
            "robot_reached_goal": step_result.robot_reached_goal,
            "terminated": terminated,
        }
        return observation, reward, terminated, truncated, info

    def _setup_visualizer(self) -> None:
        """Instantiate the project's native visualizer."""
        # Initialize the visualizer (pass self.env if your Visualizer requires it)
        if self.visualizer is None:
            self.visualizer = Visualizer()

    def render(self) -> np.ndarray | None:
        """Computes the render frames as specified by render_mode during init."""
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return None

        if self.visualizer is None:
            self._setup_visualizer()

        if self.render_mode == "human":
            # Update the on-screen display
            # self.visualizer.render(self.env)
            pass
        elif self.render_mode == "rgb_array":
            # Return a numpy array of the frame
            # return self.visualizer.get_rgb_array(self.env)
            return np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder

    def close(self) -> None:
        """Clean up rendering resources."""
        if self.visualizer is not None:
            # self.visualizer.close()
            self.visualizer = None

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

    def _robot_at_goal(self) -> bool:
        """Whether the robot's current (post-move) pose is within the
        configured goal-reach tolerance.

        Computed independently here, rather than reused from Step's
        StepResult.robot_reached_goal, because Step's flag is checked
        *before* this tick's movement is integrated (correct for its
        other callers, which loop until it becomes true) -- using it
        here would report the wrong tick's position relative to
        `terminated`, which is decided from the post-move pose.
        """
        robot = self.env.robot
        assert robot.pose is not None and robot.goal is not None
        distance = math.hypot(
            robot.goal.gx - robot.pose.px, robot.goal.gy - robot.pose.py
        )
        return distance <= robot.radius + self.env.info.goal_reach_tolerance

    def _respawn_pedestrians(self, step_result) -> None:
        """Rebuild any pedestrian that reached its goal this tick with a
        fresh pose/goal, so the crowd stays dynamic for the full episode
        instead of progressively freezing in place. Mirrors the same
        pattern test_sweep.py/GlobalPlanner already use for the coverage
        pipeline (see EnvironmentBuilder.rebuild_pedestrian).
        """
        for ped_id, reached in step_result.pedestrian_reached_goals.items():
            if reached:
                self.env = self._env_builder.rebuild_pedestrian(
                    env=self.env,
                    ped_id=ped_id,
                    random_seed=self.env.info.random_seed + self._elapsed_steps,
                )
