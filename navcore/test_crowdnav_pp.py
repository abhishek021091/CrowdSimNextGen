"""Live visualization: CrowdNavPPPolicy driving the robot inside CrowdSimNextGen.

Loads a trained checkpoint and drives the robot with it, deterministically,
so its actual learned (or not-yet-learned) behavior can be inspected visually
rather than inferred from training-log scalars alone.

Run directly:

    python -m navcore.test_crowdnav_pp_live
"""

from __future__ import annotations

import random
import time

import numpy as np
import torch

from navcore.builder.environment_builder import EnvironmentBuilder
from navcore.gym_wrapper.crowd_sim_env import CrowdSimEnv
from navcore.gym_wrapper.observation_encoder import ObservationEncoder
from navcore.middleware.orca_middleware import DecentralizedORCAPlanner
from navcore.policies.crowdnav_pp.fusion_gate import FusionGateConfig
from navcore.policies.crowdnav_pp.policy import CrowdNavPPPolicy, CrowdNavPPPolicyConfig
from navcore.policies.crowdnav_pp.range_image_encoder import RangeImageEncoderConfig
from navcore.policies.crowdnav_pp.robot_obstacle_attention import (
    RobotObstacleAttentionConfig,
)
from navcore.step.step import Step
from navcore.training.crowd_nav_pp.crowd_nav_pp_trainer import PPOConfig
from navcore.visualization.visualizer import Visualizer

# CrowdNavPPTrainer.save_checkpoint (crowd_nav_pp_trainer.py) pickles
# CrowdNavPPPolicyConfig and PPOConfig directly into every checkpoint,
# including CrowdNavPPPolicyConfig's nested dataclass fields
# (RangeImageEncoderConfig, RobotObstacleAttentionConfig, FusionGateConfig --
# these are unconditional field(default_factory=...) values, present in the
# pickle regardless of whether use_range_image_obstacles is actually True;
# their presence is NOT evidence the obstacle branch was trained -- see
# this file's own load path below, which reads the flag itself rather than
# assuming). PyTorch >=2.6 defaults torch.load(weights_only=True), which
# validates every pickled global individually, so each nested dataclass
# needs its own allowlist entry. Extend this list if save_checkpoint, or
# any config it nests, ever grows another dataclass field.
torch.serialization.add_safe_globals(
    [
        PPOConfig,
        CrowdNavPPPolicyConfig,
        RangeImageEncoderConfig,
        RobotObstacleAttentionConfig,
        FusionGateConfig,
    ]
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


CHECKPOINT_PATH = (
    "./navcore/training/crowd_nav_pp/checkpoints/run8/crowdnav_pp_step1280000.pt"
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
            Built from the checkpoint's own saved ``policy_config`` when
            no ``policy`` is injected -- never guessed at by the caller.
            A checkpoint is self-describing (``save_checkpoint`` stores
            exactly the config it was trained with); asking the caller to
            separately specify an architecture flag that must happen to
            match is what caused this class's earlier
            ``use_range_image_obstacles`` parameter to silently diverge
            from the checkpoint and fail ``load_state_dict``.
        deterministic: Whether to use the action distribution's mean
            (``True``, smoother to watch) or sample from it (``False``,
            matches actual PPO rollout behavior).
        use_range_image_obstacles: Whether the range-image obstacle
            branch is active on ``self.policy``, read from the loaded
            policy's own config so ``_select_robot_velocity`` never needs
            to guess whether ``range_image`` should be built/passed.
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
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if policy is not None:
            self.policy = policy
        else:
            # Rebuild the exact architecture this checkpoint was trained
            # with, rather than constructing a fresh CrowdNavPPPolicyConfig
            # and hoping its flags happen to match -- see class docstring.
            self.policy = CrowdNavPPPolicy(checkpoint["policy_config"])

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy.eval()
        self.deterministic = deterministic
        self.sleep_seconds = sleep_seconds
        self.use_range_image_obstacles = self.policy.config.use_range_image_obstacles

        self.encoder = ObservationEncoder(
            max_neighbors=max_neighbors, history_steps=history_steps
        )

        # include_static_obstacles=True is deliberate here, not a leftover:
        # this arena must contain obstacles whenever the loaded policy's
        # obstacle branch is active, so there's something for it to attend
        # over. If self.use_range_image_obstacles is False for the loaded
        # checkpoint (as with run8), obstacles are still fine to include --
        # the robot just won't have any learned response to them beyond
        # whatever the human branch/ORCA already provide. Set this to False
        # only if you specifically want an obstacle-free arena.
        self.env_builder = EnvironmentBuilder(include_static_obstacles=True)
        self.env = self.env_builder.build_environment()
        self.encoder.reset(self.env)
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

        extra_kwargs = {}
        if self.use_range_image_obstacles:
            extra_kwargs["range_image"] = torch.from_numpy(
                obs["range_image"]
            ).unsqueeze(0)

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
                **extra_kwargs,
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

            collided = self.env.did_collision_happened()
            out_of_bounds = self.env.out_of_bounds()

            if result.robot_reached_goal or collided or out_of_bounds:
                if result.robot_reached_goal:
                    status = "reached goal"
                elif collided:
                    status = "collided"
                else:
                    status = "left the arena bounds"
                print(
                    f"Episode {self.episode_count}: robot {status} after "
                    f"{self.tick_count} ticks."
                )
                self._reset_episode()

            time.sleep(self.sleep_seconds)


if __name__ == "__main__":
    CrowdNavPPLiveDemo().run()
