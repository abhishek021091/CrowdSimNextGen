# navcore/training/crowd_nav_pp/evaluate.py
"""Deterministic evaluation harness for a trained CrowdNavPPPolicy.

Runs a fixed number of episodes with deterministic (mean) actions and a
fixed seed sequence, and reports outcome counts plus per-episode
diagnostics. Meant to replace eyeballing single live-visualized episodes
with a comparable, repeatable number -- use this to compare
checkpoints/configs against each other, not to judge a single episode's
behavior (use test_crowdnav_pp.py for that).

Outcome classification note:
    Step._compute_velocities checks goal-reach *before* this tick's
    movement is integrated (correct for its other callers, which loop
    until it becomes true -- see task_planners.py). CrowdSimEnv.step's
    `terminated` flag, by contrast, is decided from the *post-move*
    pose via GoalReachingTask.is_terminated. Using
    info["robot_reached_goal"] here would misclassify almost every real
    goal-reach as a timeout, since it reflects the previous tick's
    position. `terminated and not collision` is the correct condition,
    since GoalReachingTask.is_terminated is exactly
    `_reached_goal() or collided`.
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass, field

import torch

from navcore.gym_wrapper.crowd_sim_env import ActionMode, CrowdSimEnv, CrowdSimEnvConfig
from navcore.gym_wrapper.goal_reaching_task import GoalReachingTask
from navcore.policies.crowdnav_pp.policy import CrowdNavPPPolicy, CrowdNavPPPolicyConfig


@dataclass(slots=True)
class EpisodeResult:
    """Per-episode outcome and diagnostics.

    Attributes:
        outcome: One of "success", "collision", "timeout".
        steps: Ticks the episode ran before ending.
        total_reward: Sum of per-tick rewards.
        start_distance: Distance from the robot's start pose to its goal.
        final_distance: Distance from the robot's end pose to its goal.
        path_length: Total distance actually traveled by the robot
            (sum of per-tick displacement, not straight-line) -- lets
            you separate "took a long, wandering path" from "took a
            direct path but ran out of time."
        min_separation: Closest the robot ever got to any pedestrian,
            minus both radii -- negative means an actual overlap
            occurred sometime during the episode, even on episodes that
            didn't end in a flagged collision (e.g. a near-miss that
            self-resolved).
    """

    outcome: str
    steps: int
    total_reward: float
    start_distance: float
    final_distance: float
    path_length: float
    min_separation: float


@dataclass(slots=True)
class EvalStats:
    episodes: list[EpisodeResult] = field(default_factory=list)

    def report(self) -> dict[str, float | int]:
        n = max(len(self.episodes), 1)
        rewards = [e.total_reward for e in self.episodes]
        steps = [e.steps for e in self.episodes]
        final_distances = [e.final_distance for e in self.episodes]
        path_lengths = [e.path_length for e in self.episodes]
        min_seps = [e.min_separation for e in self.episodes]

        successes = [e for e in self.episodes if e.outcome == "success"]
        collisions = [e for e in self.episodes if e.outcome == "collision"]
        timeouts = [e for e in self.episodes if e.outcome == "timeout"]

        def _mean(values: list[float]) -> float:
            return statistics.mean(values) if values else float("nan")

        def _stdev(values: list[float]) -> float:
            return statistics.stdev(values) if len(values) > 1 else 0.0

        return {
            "episodes": len(self.episodes),
            "success_rate": len(successes) / n,
            "collision_rate": len(collisions) / n,
            "timeout_rate": len(timeouts) / n,
            "mean_episode_reward": _mean(rewards),
            "std_episode_reward": _stdev(rewards),
            "mean_steps": _mean(steps),
            # For timeouts/collisions specifically -- how close did the
            # robot actually get? A high mean here on timeouts means
            # "not making progress"; a low one means "arriving but not
            # quite crossing the success threshold" (see goal_reach
            # tolerance discussion).
            "mean_final_distance_timeouts": _mean([e.final_distance for e in timeouts]),
            "mean_final_distance_collisions": _mean(
                [e.final_distance for e in collisions]
            ),
            "mean_path_length": _mean(path_lengths),
            # Path efficiency: straight-line distance / actual distance
            # traveled, only meaningful for successes (a collision or
            # timeout didn't complete a path to compare against).
            "mean_path_efficiency_successes": _mean(
                [
                    e.start_distance / e.path_length
                    for e in successes
                    if e.path_length > 1e-6
                ]
            ),
            "min_separation_ever": min(min_seps) if min_seps else float("nan"),
            "mean_min_separation": _mean(min_seps),
        }

    def print_report(self) -> None:
        report = self.report()
        print(f"episodes:                          {report['episodes']}")
        print(f"success_rate:                       {report['success_rate']:.3f}")
        print(f"collision_rate:                     {report['collision_rate']:.3f}")
        print(f"timeout_rate:                        {report['timeout_rate']:.3f}")
        print(
            f"mean_episode_reward:                {report['mean_episode_reward']:.3f} "
            f"(+/- {report['std_episode_reward']:.3f})"
        )
        print(f"mean_steps:                         {report['mean_steps']:.1f}")
        print(
            f"mean_final_distance (timeouts):     "
            f"{report['mean_final_distance_timeouts']:.3f}"
        )
        print(
            f"mean_final_distance (collisions):   "
            f"{report['mean_final_distance_collisions']:.3f}"
        )
        print(f"mean_path_length:                   {report['mean_path_length']:.3f}")
        print(
            f"mean_path_efficiency (successes):   "
            f"{report['mean_path_efficiency_successes']:.3f}"
        )
        print(
            f"min_separation_ever:                {report['min_separation_ever']:.3f}"
        )
        print(
            f"mean_min_separation:                {report['mean_min_separation']:.3f}"
        )


def evaluate(
    checkpoint_path: str,
    n_episodes: int,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
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

        robot = env.env.robot
        assert robot.pose is not None and robot.goal is not None
        start_distance = math.hypot(
            robot.goal.gx - robot.pose.px, robot.goal.gy - robot.pose.py
        )
        prev_px, prev_py = robot.pose.px, robot.pose.py

        episode_reward = 0.0
        path_length = 0.0
        min_separation = float("inf")
        steps = 0
        done = False
        info: dict = {}

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
            steps += 1
            done = terminated or truncated

            robot = env.env.robot
            assert robot.pose is not None
            path_length += math.hypot(robot.pose.px - prev_px, robot.pose.py - prev_py)
            prev_px, prev_py = robot.pose.px, robot.pose.py

            for ped in env.env.crowd.values():
                if ped.pose is None:
                    continue
                separation = (
                    math.hypot(robot.pose.px - ped.pose.px, robot.pose.py - ped.pose.py)
                    - robot.radius
                    - ped.radius
                )
                min_separation = min(min_separation, separation)

        assert robot.pose is not None and robot.goal is not None
        final_distance = math.hypot(
            robot.goal.gx - robot.pose.px, robot.goal.gy - robot.pose.py
        )

        if terminated and not info.get("collision"):
            outcome = "success"
        elif info.get("collision"):
            outcome = "collision"
        else:
            outcome = "timeout"

        result = EpisodeResult(
            outcome=outcome,
            steps=steps,
            total_reward=episode_reward,
            start_distance=start_distance,
            final_distance=final_distance,
            path_length=path_length,
            min_separation=min_separation if min_separation != float("inf") else 0.0,
        )
        stats.episodes.append(result)

        if verbose:
            print(
                f"episode {episode:3d}: {outcome:9s}  steps={steps:4d}  "
                f"reward={episode_reward:8.2f}  final_dist={final_distance:.3f}  "
                f"path_len={path_length:.2f}  min_sep={result.min_separation:.3f}"
            )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained CrowdNavPPPolicy")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--verbose", action="store_true", help="Print a line per episode."
    )
    args = parser.parse_args()

    stats = evaluate(
        args.checkpoint, args.n_episodes, args.seed, args.device, verbose=args.verbose
    )
    stats.print_report()


if __name__ == "__main__":
    main()
