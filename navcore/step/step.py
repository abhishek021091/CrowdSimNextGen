"""Step: advances the simulation by exactly one tick.

Per agent, per tick, this ties together the pieces described in
mission.py / policy.py: Mission (what's my target right now) feeds a
VelocityPlanner (currently always DecentralizedORCAPlanner), whose
output is integrated into the agent's pose. Missions are optional --
an agent with none targets its own `agent.goal` directly, identical to
the previous hardcoded behavior.

Design note -- why the mission's target is folded into FullState.goal
rather than written onto agent.goal:
    mission.py is explicit that a Mission's per-tick target and an
    agent's persistent `Goal` are different things (a sweeping robot's
    target is a coverage waypoint, not its "real" destination). Step
    therefore builds a throwaway FullState with `goal` set to whatever
    the Mission returned, for planning purposes only -- agent.goal
    itself is never touched here, so anything else that reads it
    (goal-reached checks, UI, metrics) keeps seeing the agent's real
    destination.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import tomllib

import navcore.configs
from navcore.entities.agents.agent import Agent
from navcore.entities.agents.pedestrians import Pedestrian
from navcore.entities.agents.robot import Robot
from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.components.goal import Goal
from navcore.entities.components.state import FullState, ObservableState
from navcore.entities.components.velocity import Velocity
from navcore.environment.environment import Environment
from navcore.middleware.orca_middleware import VelocityPlanner
from navcore.missions.group_goal_reaching import GroupGoalReachingMission
from navcore.missions.mission import Mission


@dataclass(slots=True)
class StepResult:
    robot_velocity: Velocity
    crowd_velocities: dict[int, Velocity]


class Step:
    """Advances one tick: Mission -> VelocityPlanner -> integration.

    Attributes:
        env: The live environment this Step mutates.
        robot: Convenience alias for ``env.robot``.
        crowd: Convenience alias for ``env.crowd``.
        planner: Computes velocities for a self-agent plus its visible
            neighbors (see ``VelocityPlanner`` protocol).
        robot_visible: Whether the robot should appear in pedestrians'
            sensor observations this episode.
        robot_mission: Optional. Produces the robot's per-tick target
            (e.g. a ``SweepMission``). If ``None``, the robot always
            targets ``robot.goal`` directly.
        crowd_missions: Optional, keyed by pedestrian id. Same idea as
            ``robot_mission``, per pedestrian. Pedestrians absent from
            this mapping target their own ``goal`` directly.
    """

    assert navcore.configs.__file__ is not None
    env_path = Path(navcore.configs.__file__).parent / "env.toml"
    with open(env_path, "rb") as f:
        env_config = tomllib.load(f)

    ROBOT_KEY = -1
    dt = env_config["policy"]["time_step"]

    def __init__(
        self,
        planner: VelocityPlanner,
        env: Environment,
        robot_visible: bool,
        robot_mission: Mission | None = None,
        crowd_missions: Mapping[int, Mission] | None = None,
    ) -> None:
        self.env = env
        self.robot = env.robot
        self.crowd = env.crowd
        self.planner = planner
        self.robot_visible = robot_visible
        self.robot_mission = robot_mission
        self.crowd_missions: Mapping[int, Mission] = crowd_missions or {}
        # Cached per pedestrian id, rebuilt only if that member's group
        # instance changes (e.g. after a future split/merge) -- avoids
        # reconstructing (and re-validating) a GroupGoalReachingMission
        # for every group member on every single tick.
        self._group_missions: dict[int, GroupGoalReachingMission] = {}

    def step(self) -> StepResult:
        self._validate()
        self._change_group_goals()
        result = self._compute_velocities()
        self._set_velocities(result.robot_velocity, result.crowd_velocities)

        self._advance_agent(self.env.robot)
        for ped in self.env.crowd.values():
            self._advance_agent(ped)

        return result

    def _validate(self) -> None:
        if (
            self.env.robot.pose is None
            or self.env.robot.velocity is None
            or self.env.robot.goal is None
        ):
            raise RuntimeError("Robot must have pose, velocity, and goal initialized.")
        if self.env.robot.sensor is None:
            raise RuntimeError("Robot sensor must be initialized.")

        for ped in self.env.crowd.values():
            if ped.pose is None or ped.velocity is None or ped.goal is None:
                raise RuntimeError(
                    f"Pedestrian {ped.id} must have pose, velocity, and goal initialized."
                )
            if ped.sensor is None:
                raise RuntimeError(f"Pedestrian {ped.id} sensor must be initialized.")

    # -- Mission -> FullState plumbing --------------------------------

    def _target_for(
        self,
        agent: Agent,
        mission: Mission | None,
        neighbors: Sequence[ObservableState],
    ) -> Vector2:
        """Return this tick's target: mission-derived, or the agent's own goal."""
        if mission is not None:
            return mission.get_target(agent, neighbors)
        assert agent.goal is not None  # guaranteed by _validate()
        return Vector2(agent.goal.gx, agent.goal.gy)

    def _full_state_for(self, agent: Agent, target: Vector2) -> FullState:
        """Build a planning-only FullState with ``target`` standing in for goal."""
        assert agent.pose is not None and agent.velocity is not None
        return FullState(
            pose=agent.pose,
            goal=Goal(target.x, target.y),
            velocity=agent.velocity,
            radius=agent.radius,
            preferred_speed=agent.v_pref,
        )

    # -- velocity computation -------------------------------------------

    def _compute_velocities(self) -> StepResult:
        robot_velocity = self._compute_robot_velocity()
        crowd_velocities = self._compute_crowd_velocities()
        return StepResult(
            robot_velocity=robot_velocity, crowd_velocities=crowd_velocities
        )

    def _compute_robot_velocity(self) -> Velocity:
        assert self.env.robot.sensor is not None
        robot_obs = self.env.robot.sensor.observe(
            self.env, robot_visible=self.robot_visible
        )
        robot_obs[self.ROBOT_KEY] = self.env.robot.get_observable_state()

        target = self._target_for(
            self.env.robot, self.robot_mission, list(robot_obs.values())
        )
        full_state = self._full_state_for(self.env.robot, target)

        robot_velocity, _ = self.planner.compute_velocities(
            self.ROBOT_KEY, full_state, robot_obs
        )
        return robot_velocity

    def _compute_crowd_velocities(self) -> dict[int, Velocity]:
        crowd_velocities: dict[int, Velocity] = {}

        for ped_id, ped in self.env.crowd.items():
            assert ped.sensor is not None
            ped_obs = ped.sensor.observe(self.env, robot_visible=self.robot_visible)
            ped_obs[ped_id] = ped.get_observable_state()

            mission = self.crowd_missions.get(ped_id)
            target = self._target_for(ped, mission, list(ped_obs.values()))
            full_state = self._full_state_for(ped, target)

            ped_velocity, _ = self.planner.compute_velocities(
                ped_id, full_state, ped_obs
            )
            crowd_velocities[ped_id] = ped_velocity

        return crowd_velocities

    # -- apply + integrate ------------------------------------------------

    def _set_velocities(
        self,
        robot_velocity: Velocity,
        crowd_velocities: dict[int, Velocity],
    ) -> None:
        self._apply_robot_velocity(self.env.robot, {self.ROBOT_KEY: robot_velocity})
        self._apply_crowd_velocities(list(self.env.crowd.values()), crowd_velocities)

    def _advance_agent(self, agent: Agent) -> None:
        if agent.pose is None:
            raise RuntimeError("Agent pose is missing.")
        if agent.velocity is None:
            raise RuntimeError("Agent velocity is missing.")

        agent.pose.px += agent.velocity.vx * self.dt
        agent.pose.py += agent.velocity.vy * self.dt

        speed_sq = (
            agent.velocity.vx * agent.velocity.vx
            + agent.velocity.vy * agent.velocity.vy
        )
        if speed_sq > 1e-12:
            agent.pose.theta = math.atan2(agent.velocity.vy, agent.velocity.vx)

    def _apply_robot_velocity(
        self,
        robot: Robot,
        velocities: dict[int, Velocity],
    ) -> Velocity:
        if self.ROBOT_KEY not in velocities:
            raise KeyError("Planner did not return a velocity for the robot.")
        robot.velocity = velocities[self.ROBOT_KEY]
        return robot.velocity

    def _apply_crowd_velocities(
        self,
        crowd: list[Pedestrian],
        velocities: dict[int, Velocity],
    ) -> dict[int, Velocity]:
        crowd_velocities: dict[int, Velocity] = {}

        for ped in crowd:
            if ped.id not in velocities:
                raise KeyError(
                    f"Planner did not return a velocity for pedestrian '{ped.id}'."
                )
            ped.velocity = velocities[ped.id]
            crowd_velocities[ped.id] = ped.velocity

        return crowd_velocities

    def _change_group_goals(self) -> None:
        for group in self.env.group_state().values():
            for member_id in group:
                mission = self._group_missions.get(member_id)
                if mission is None or mission.group is not group:
                    mission = GroupGoalReachingMission(
                        agent_id=member_id,
                        group=group,
                        agent_lookup=self.env.crowd.__getitem__,
                    )
                    self._group_missions[member_id] = mission
                mission.set_goal()
