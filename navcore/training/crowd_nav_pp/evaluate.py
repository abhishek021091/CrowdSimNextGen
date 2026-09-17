# navcore/training/crowd_nav_pp/evaluate.py
"""Deterministic evaluation harness for a trained CrowdNavPPPolicy.

Runs a fixed number of episodes with deterministic (mean) actions and a
fixed seed sequence, and reports outcome counts. Meant to replace
eyeballing single live-visualized episodes with a comparable, repeatable
number -- use this to compare checkpoints/configs against each other,
not to judge a single episode's behavior (use test_crowdnav_pp.py for
that).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import torch

from navcore.gym_wrapper.crowd_sim_env import ActionMode, CrowdSimEnv, CrowdSimEnvConfig
from navcore.gym_wrapper.goal_reaching_task import GoalReachingTask
from navcore.policies.crowdnav_pp.policy import CrowdNavPPPolicy, CrowdNavPPPolicyConfig


@dataclass(slots=True)
class EvalStats:
    episodes: int = 0
    reached_goal: int = 0
    collided: int = 0
    timed_out: int = 0
    episode_rewards: list[float] = field(default_factory=list)

    def report(self) -> dict[str, float]:
        n = max(self.episodes, 1)
        return {
            "episodes": self.episodes,
            "success_rate": self.reached_goal / n,
            "collision_rate": self.collided / n,
            "timeout_rate": self.timed_out / n,
            "mean_episode_reward": (
                sum(self.episode_rewards) / len(self.episode_rewards)
                if self.episode_rewards
                else float("nan")
            ),
        }


def evaluate(
    checkpoint_path: str,
    n_episodes: int,
    seed: int = 0,
    device: str = "cpu",
) -> EvalStats:
    env_config = CrowdSimEnvConfig(action_mode=ActionMode.VELOCITY)
    env = CrowdSimEnv(GoalReachingTask(), env_config)

    obs, _ = env.reset(seed=seed)
    policy = CrowdNavPPPolicy(
        CrowdNavPPPolicyConfig(
            robot_feature_dim=obs["robot"].shape[-1],
            neighbor_feature_dim=obs["neighbors"].shape[-1],
        )
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    stats = EvalStats()

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)  # fixed, reproducible sequence
        hidden = policy.initial_hidden_state(nenv=1, device=torch.device(device))
        not_done_mask = torch.zeros(1)  # first tick of episode: reset hidden state
        episode_reward = 0.0
        done = False

        while not done:
            batch = {
                k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
                for k, v in obs.items()
            }
            with torch.no_grad():
                action, _, _, hidden = policy.act(
                    batch["robot"],
                    batch["neighbors"],
                    batch["neighbor_mask"],
                    batch["neighbor_history"],
                    batch["neighbor_history_mask"],
                    hidden,
                    not_done_mask,
                    deterministic=True,
                )
            not_done_mask = torch.ones(1)  # every subsequent tick carries state
            obs, reward, terminated, truncated, info = env.step(
                action.squeeze(0).cpu().numpy()
            )
            episode_reward += float(reward)
            done = terminated or truncated

        stats.episodes += 1
        stats.episode_rewards.append(episode_reward)
        if info.get("robot_reached_goal"):
            stats.reached_goal += 1
        elif info.get("collision"):
            stats.collided += 1
        else:
            stats.timed_out += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained CrowdNavPPPolicy")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    stats = evaluate(args.checkpoint, args.n_episodes, args.seed, args.device)
    for key, value in stats.report().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
