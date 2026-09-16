"""Live visualization: CrowdNavPPPolicy driving the robot inside CrowdSimNextGen.

Ties the assembled policy (``policy.py``) into a real running episode --
``ObservationEncoder`` builds the same observation a training loop would
see, ``CrowdNavPPPolicy.act()`` picks the robot's velocity, ``Step``
integrates it alongside ORCA-driven pedestrians, and ``Visualizer`` draws
every tick, mirroring how ``test_sweep.py``/``test_global_planner.py``
drive their own loops.

IMPORTANT -- this policy is untrained (freshly initialized weights).
This script demonstrates that the observation -> CrowdNav++ -> action ->
simulation pipeline is wired correctly end-to-end; it does not
demonstrate learned collision-avoidance behavior. The robot's motion
will look undirected/random until an actual training loop (PPO or
otherwise) has updated these weights -- that is expected, not a bug.

Run directly:

    python -m navcore.test_crowdnav_pp_live
"""

from __future__ import annotations

import time

import torch

from navcore.builder.environment_builder import EnvironmentBuilder
from navcore.gym_wrapper.crowd_sim_env import CrowdSimEnv
from navcore.gym_wrapper.observation_encoder import ObservationEncoder
from navcore.middleware.orca_middleware import DecentralizedORCAPlanner
from navcore.policies.crowdnav_pp.policy import CrowdNavPPPolicy, CrowdNavPPPolicyConfig
from navcore.step.step import Step
from navcore.visualization.visualizer import Visualizer


class CrowdNavPPLiveDemo:
    """Drives CrowdSimNextGen's robot with a live ``CrowdNavPPPolicy``.

    Owns everything an episode needs: the ``Environment``, the ORCA
    planner driving pedestrians, the observation encoder, the policy's
    recurrent hidden state, and the visualizer -- rebuilding all of it on
    every episode boundary (goal reached or collision), so the demo runs
    continuously rather than exiting after one episode.

    Attributes:
        policy: The (by default, freshly initialized/untrained)
            CrowdNav++ policy driving the robot.
        deterministic: Whether to use the action distribution's mean
            (``True``, smoother to watch) or sample from it (``False``,
            matches actual PPO rollout behavior).
    """

    def __init__(
        self,
        policy: CrowdNavPPPolicy | None = None,
        max_neighbors: int = 10,
        history_steps: int = 8,
        deterministic: bool = True,
        sleep_seconds: float = 0.03,
    ) -> None:
        self.policy = (
            policy if policy is not None else CrowdNavPPPolicy(CrowdNavPPPolicyConfig())
        )
        self.policy.eval()
        self.deterministic = deterministic
        self.sleep_seconds = sleep_seconds

        self.encoder = ObservationEncoder(
            max_neighbors=max_neighbors, history_steps=history_steps
        )

        self.env_builder = EnvironmentBuilder()
        self.env = self.env_builder.build_environment()
        self.step_driver = self._build_step_driver()

        self.hidden_state = self.policy.initial_hidden_state(nenv=1)
        self.visualizer = Visualizer()
        self.episode_count = 0
        self.tick_count = 0

    def _build_step_driver(self) -> Step:
        planner = DecentralizedORCAPlanner(
            config_file="orca.toml", obstacles=self.env.obstacles
        )
        return Step(planner=planner, env=self.env, robot_visible=False)

    def _select_robot_velocity(self):
        """Encode the current tick's observation and query the policy for
        the robot's velocity.

        Note: reuses ``CrowdSimEnv._decode_velocity_action`` for the
        action-to-velocity conversion (magnitude clipping to v_pref)
        rather than re-deriving that clipping logic here, per the
        project's "avoid duplicate logic across call sites" convention.
        It is a "private" (leading-underscore) staticmethod being reused
        outside its class -- pragmatic today, but promoting it to a
        public, standalone function (it has no dependency on
        ``CrowdSimEnv`` instance state) would be a small, worthwhile
        cleanup if a second caller ever needs it.
        """
        obs = self.encoder.encode(self.env)
        robot = torch.from_numpy(obs["robot"]).unsqueeze(0)
        neighbors = torch.from_numpy(obs["neighbors"]).unsqueeze(0)
        neighbor_mask = torch.from_numpy(obs["neighbor_mask"]).unsqueeze(0)
        neighbor_history = torch.from_numpy(obs["neighbor_history"]).unsqueeze(0)
        neighbor_history_mask = torch.from_numpy(
            obs["neighbor_history_mask"]
        ).unsqueeze(0)
        not_done_mask = torch.ones(1)

        with torch.no_grad():
            action, _, _, self.hidden_state = self.policy.act(
                robot,
                neighbors,
                neighbor_mask,
                neighbor_history,
                neighbor_history_mask,
                self.hidden_state,
                not_done_mask,
                deterministic=self.deterministic,
            )

        return CrowdSimEnv._decode_velocity_action(action.squeeze(0).numpy())

    def _reset_episode(self) -> None:
        self.episode_count += 1
        seed = self.env.info.random_seed + self.episode_count
        self.env = self.env_builder.reset(random_seed=seed)
        self.step_driver = self._build_step_driver()
        self.encoder.reset()
        self.hidden_state = self.policy.initial_hidden_state(nenv=1)

    def run(self, max_ticks: int | None = None) -> None:
        print(
            "Running live CrowdNav++ demo -- policy is UNTRAINED (random "
            "weights). This shows the observation -> policy -> action -> "
            "simulation pipeline wired end-to-end, not learned avoidance "
            "behavior."
        )
        while max_ticks is None or self.tick_count < max_ticks:
            self.tick_count += 1
            velocity = self._select_robot_velocity()
            result = self.step_driver.step(robot_velocity_override=velocity)
            self.visualizer.refresh(self.env)

            if result.robot_reached_goal or self.env.did_collision_happened():
                status = "reached goal" if result.robot_reached_goal else "collided"
                print(
                    f"Episode {self.episode_count}: robot {status} after "
                    f"{self.tick_count} ticks."
                )
                self._reset_episode()

            time.sleep(self.sleep_seconds)


if __name__ == "__main__":
    CrowdNavPPLiveDemo().run()
