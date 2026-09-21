"""Live visualization: CrowdNavPPPolicy driving the robot inside CrowdSimNextGen.

Loads a trained checkpoint and drives the robot with it, deterministically,
so its actual learned (or not-yet-learned) behavior can be inspected visually
rather than inferred from training-log scalars alone.

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

CHECKPOINT_PATH = (
    "./navcore/training/crowd_nav_pp/checkpoints/run3/crowdnav_pp_step6881280.pt"
)


class CrowdNavPPLiveDemo:
    """Drives CrowdSimNextGen's robot with a live ``CrowdNavPPPolicy``.

    Owns everything an episode needs: the ``Environment``, the ORCA
    planner driving pedestrians, the observation encoder, the policy's
    recurrent hidden state, and the visualizer -- rebuilding all of it on
    every episode boundary (goal reached or collision), so the demo runs
    continuously rather than exiting after one episode.

    Attributes:
        policy: The loaded, trained CrowdNav++ policy driving the robot.
        deterministic: Whether to use the action distribution's mean
            (``True``, smoother to watch) or sample from it (``False``,
            matches actual PPO rollout behavior).
    """

    def __init__(
        self,
        policy: CrowdNavPPPolicy | None = None,
        checkpoint_path: str = CHECKPOINT_PATH,
        max_neighbors: int = 10,
        history_steps: int = 8,
        deterministic: bool = True,
        sleep_seconds: float = 0.03,
    ) -> None:
        self.policy = (
            policy if policy is not None else CrowdNavPPPolicy(CrowdNavPPPolicyConfig())
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        print(self.policy.action_head.log_std.exp())
        self.policy.eval()
        self.deterministic = deterministic
        self.sleep_seconds = sleep_seconds

        self.encoder = ObservationEncoder(
            max_neighbors=max_neighbors, history_steps=history_steps
        )

        # include_static_obstacles=False -- must match CrowdSimEnvConfig's
        # default used during training (see crowd_sim_env.py). This
        # policy has never seen a static obstacle; evaluating it in an
        # arena full of tables would be a distribution-shift artifact,
        # not a measurement of what it actually learned.
        self.env_builder = EnvironmentBuilder(include_static_obstacles=True)
        self.env = self.env_builder.build_environment()
        self.step_driver = self._build_step_driver()

        self.hidden_state = self.policy.initial_hidden_state(nenv=1)
        self.visualizer = Visualizer()
        self.episode_count = 0
        self.tick_count = 0

    def _build_step_driver(self) -> Step:
        crowd_planner = DecentralizedORCAPlanner(config_file="orca.toml")
        return Step(
            crowd_planner=crowd_planner,
            env=self.env,
            robot_visible=False,
        )

    def _select_robot_velocity(self):
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
        self.encoder.reset(self.env)
        self.hidden_state = self.policy.initial_hidden_state(nenv=1)

    def _respawn_pedestrians(self, result) -> None:
        """Rebuild any pedestrian that reached its goal this tick, so the
        crowd stays dynamic for the whole episode instead of progressively
        freezing in place. Same pattern as test_sweep.py/GlobalPlanner.
        """
        for ped_id, reached in result.pedestrian_reached_goals.items():
            if reached:
                self.env = self.env_builder.rebuild_pedestrian(
                    env=self.env,
                    ped_id=ped_id,
                    random_seed=self.env.info.random_seed + self.tick_count + ped_id,
                )

    def run(self, max_ticks: int | None = None) -> None:
        print(
            f"Running live CrowdNav++ demo with checkpoint from "
            f"{CHECKPOINT_PATH} -- watching learned behavior."
        )
        while max_ticks is None or self.tick_count < max_ticks:
            self.tick_count += 1
            velocity = self._select_robot_velocity()
            result = self.step_driver.step(robot_velocity_override=velocity)
            self._respawn_pedestrians(result)
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
