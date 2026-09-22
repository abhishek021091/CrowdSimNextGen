# navcore/training/crowd_nav_pp/evaluate.py
"""Deterministic evaluation harness for a trained CrowdNavPPPolicy.

Runs a fixed number of episodes with deterministic (mean) actions and a
fixed seed sequence, and reports outcome counts plus per-episode
diagnostics.

Hardcoded to this project's obstacle-encoder-enabled checkpoints
(use_obstacle_encoder=True, default ObstacleEncoderConfig) -- matches how
test_crowdnav_pp.py constructs CrowdNavPPPolicy for the same checkpoint
family. If you evaluate a checkpoint trained WITHOUT the obstacle
encoder, set USE_OBSTACLE_ENCODER = False below (or pass --no-obstacle-encoder).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from navcore.analysis.metrics_logger import TrainingMetricsLogger
from navcore.entities.agents.robot import Robot
from navcore.gym_wrapper.crowd_sim_env import ActionMode, CrowdSimEnv, CrowdSimEnvConfig
from navcore.gym_wrapper.goal_reaching_task import GoalReachingTask
from navcore.policies.crowdnav_pp.obstacle_encoder import (
    ObstacleEncoder,
    ObstacleEncoderConfig,
)
from navcore.policies.crowdnav_pp.policy import CrowdNavPPPolicy, CrowdNavPPPolicyConfig
from navcore.step.step import Step

_OUTCOMES: tuple[str, ...] = ("success", "collision", "out_of_bounds", "timeout")


def classify_outcome(info: dict) -> str:
    """Classify one finished episode from its terminal-tick info dict.

    Precedence matters: GoalReachingTask.is_terminated is
    reached_goal() or collided or out_of_bounds, so a tick can satisfy
    more than one condition. Collision is checked first (the task's
    actual failure mode), then out_of_bounds, then success -- checking
    "terminated and not collision" alone (the old version of this
    function) silently misclassified an out-of-bounds episode as a
    success.
    """
    if info.get("collision"):
        return "collision"
    if info.get("out_of_bounds"):
        return "out_of_bounds"
    if info.get("terminated"):
        return "success"
    return "timeout"


@dataclass(slots=True)
class EpisodeResult:
    outcome: str
    steps: int
    total_reward: float
    start_distance: float
    final_distance: float
    path_length: float
    time_to_goal: float | None
    min_separation: float
    mean_separation: float
    action_saturation_rate: float
    mean_commanded_speed: float
    mean_actual_speed: float
    seed: int


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        nan = float("nan")
        return {
            "mean": nan,
            "std": nan,
            "min": nan,
            "max": nan,
            "p10": nan,
            "p50": nan,
            "p90": nan,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


@dataclass(slots=True)
class EvalStats:
    episodes: list[EpisodeResult] = field(default_factory=list)

    def report(self) -> dict[str, Any]:
        n = len(self.episodes)
        if n == 0:
            raise ValueError("No episodes to report on.")

        def values(attr: str, where=lambda e: True) -> list[float]:
            return [getattr(e, attr) for e in self.episodes if where(e)]

        successes = [e for e in self.episodes if e.outcome == "success"]
        min_seps = values("min_separation")

        return {
            "episodes": n,
            "outcomes": {
                o: {
                    "count": sum(1 for e in self.episodes if e.outcome == o),
                    "rate": sum(1 for e in self.episodes if e.outcome == o) / n,
                }
                for o in _OUTCOMES
            },
            "reward": _stats(values("total_reward")),
            "reward_by_outcome": {
                o: _stats(values("total_reward", lambda e, o=o: e.outcome == o))
                for o in _OUTCOMES
            },
            "steps": _stats(values("steps")),
            "path_length": _stats(values("path_length")),
            "final_distance": _stats(values("final_distance")),
            "time_to_goal": _stats(
                [e.time_to_goal for e in successes if e.time_to_goal is not None]
            ),
            "path_efficiency_success": _stats(
                [
                    e.start_distance / e.path_length
                    for e in successes
                    if e.path_length > 1e-6
                ]
            ),
            "separation": {
                "min_ever": min(min_seps) if min_seps else float("nan"),
                "mean_of_episode_min": float(np.mean(min_seps))
                if min_seps
                else float("nan"),
                "mean_of_episode_mean": float(np.mean(values("mean_separation"))),
                "negative_separation_rate": sum(1 for s in min_seps if s < 0.0) / n,
            },
            "policy_behavior": {
                "mean_action_saturation_rate": float(
                    np.mean(values("action_saturation_rate"))
                ),
                "mean_commanded_speed": float(np.mean(values("mean_commanded_speed"))),
                "mean_actual_speed": float(np.mean(values("mean_actual_speed"))),
            },
        }

    def print_report(self) -> None:
        r = self.report()
        print(f"episodes: {r['episodes']}\n")

        print("outcomes:")
        for o in _OUTCOMES:
            d = r["outcomes"][o]
            print(f"  {o:<13} {d['count']:>5} ({d['rate']:>6.1%})")
        print()

        def row(label: str, block: dict[str, float]) -> None:
            print(
                f"  {label:<24} mean={block['mean']:>8.3f}  std={block['std']:>7.3f}  "
                f"p10={block['p10']:>8.3f}  p50={block['p50']:>8.3f}  p90={block['p90']:>8.3f}  "
                f"min={block['min']:>8.3f}  max={block['max']:>8.3f}"
            )

        print("reward:")
        row("overall", r["reward"])
        for o in _OUTCOMES:
            row(f"| {o}", r["reward_by_outcome"][o])
        print()

        print("episode shape:")
        row("steps", r["steps"])
        row("path_length (m)", r["path_length"])
        row("final_distance (m)", r["final_distance"])
        row("time_to_goal (s, success)", r["time_to_goal"])
        row("path_efficiency (success)", r["path_efficiency_success"])
        print()

        s = r["separation"]
        print("safety:")
        print(f"  min_separation_ever:           {s['min_ever']:.3f} m")
        print(f"  mean(episode min separation):  {s['mean_of_episode_min']:.3f} m")
        print(f"  mean(episode mean separation): {s['mean_of_episode_mean']:.3f} m")
        print(f"  negative_separation_rate:      {s['negative_separation_rate']:.1%}")
        print()

        b = r["policy_behavior"]
        print("policy behavior:")
        print(
            f"  mean action saturation rate:   {b['mean_action_saturation_rate']:.1%}"
        )
        print(f"  mean commanded speed:          {b['mean_commanded_speed']:.3f} m/s")
        print(f"  mean actual speed:             {b['mean_actual_speed']:.3f} m/s")


def evaluate(
    checkpoint_path: str,
    n_episodes: int,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
    use_obstacle_encoder: bool = True,
    episode_log_path: str | Path | None = None,
) -> EvalStats:
    env_config = CrowdSimEnvConfig(action_mode=ActionMode.VELOCITY)
    env = CrowdSimEnv(GoalReachingTask(), env_config)

    obs, _ = env.reset(seed=seed)

    obstacle_encoder = (
        ObstacleEncoder(ObstacleEncoderConfig()) if use_obstacle_encoder else None
    )
    policy = CrowdNavPPPolicy(
        CrowdNavPPPolicyConfig(
            robot_feature_dim=obs["robot"].shape[-1],
            neighbor_feature_dim=obs["neighbors"].shape[-1],
            use_obstacle_encoder=use_obstacle_encoder,
        ),
        obstacle_encoder=obstacle_encoder,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.to(device)
    policy.eval()

    v_max = float(Robot.config["kinematics"]["v_pref"])
    episode_logger = (
        TrainingMetricsLogger(Path(episode_log_path)) if episode_log_path else None
    )

    stats = EvalStats()

    for episode in range(n_episodes):
        episode_seed = seed + episode
        obs, _ = env.reset(seed=episode_seed)
        torch_device = torch.device(device)
        hidden = policy.initial_hidden_state(nenv=1, device=torch_device)
        not_done_mask = torch.zeros(1, device=torch_device)

        robot = env.env.robot
        assert robot.pose is not None and robot.goal is not None
        start_distance = math.hypot(
            robot.goal.gx - robot.pose.px, robot.goal.gy - robot.pose.py
        )
        prev_px, prev_py = robot.pose.px, robot.pose.py

        episode_reward = 0.0
        path_length = 0.0
        min_separation = float("inf")
        tick_min_separations: list[float] = []
        commanded_speeds: list[float] = []
        saturated_ticks = 0
        steps = 0
        done = False
        info: dict = {}

        while not done:
            batch = {
                k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
                for k, v in obs.items()
            }
            extra_kwargs = (
                {"ray_features": batch["ray_features"]} if use_obstacle_encoder else {}
            )

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
                    **extra_kwargs,
                )
            not_done_mask = torch.ones(1, device=torch_device)

            action_np = action.squeeze(0).cpu().numpy()
            commanded_speed = float(math.hypot(action_np[0], action_np[1]))
            commanded_speeds.append(commanded_speed)
            if commanded_speed > v_max + 1e-9:
                saturated_ticks += 1

            obs, reward, terminated, truncated, info = env.step(action_np)
            episode_reward += float(reward)
            steps += 1
            done = terminated or truncated

            robot = env.env.robot
            assert robot.pose is not None
            path_length += math.hypot(robot.pose.px - prev_px, robot.pose.py - prev_py)
            prev_px, prev_py = robot.pose.px, robot.pose.py

            tick_min = float("inf")
            for ped in env.env.crowd.values():
                if ped.pose is None:
                    continue
                separation = (
                    math.hypot(robot.pose.px - ped.pose.px, robot.pose.py - ped.pose.py)
                    - robot.radius
                    - ped.radius
                )
                tick_min = min(tick_min, separation)
                min_separation = min(min_separation, separation)
            if math.isfinite(tick_min):
                tick_min_separations.append(tick_min)

        assert robot.pose is not None and robot.goal is not None
        final_distance = math.hypot(
            robot.goal.gx - robot.pose.px, robot.goal.gy - robot.pose.py
        )
        outcome = classify_outcome(info)

        result = EpisodeResult(
            outcome=outcome,
            steps=steps,
            total_reward=episode_reward,
            start_distance=start_distance,
            final_distance=final_distance,
            path_length=path_length,
            time_to_goal=(steps * Step.dt) if outcome == "success" else None,
            min_separation=min_separation if math.isfinite(min_separation) else 0.0,
            mean_separation=float(np.mean(tick_min_separations))
            if tick_min_separations
            else 0.0,
            action_saturation_rate=saturated_ticks / steps if steps else 0.0,
            mean_commanded_speed=float(np.mean(commanded_speeds))
            if commanded_speeds
            else 0.0,
            mean_actual_speed=path_length / (steps * Step.dt) if steps else 0.0,
            seed=episode_seed,
        )
        stats.episodes.append(result)

        if episode_logger is not None:
            episode_logger.log(episode, asdict(result))

        if verbose:
            print(
                f"episode {episode:4d}: {result.outcome:12s}  steps={result.steps:4d}  "
                f"reward={result.total_reward:8.2f}  final_dist={result.final_distance:.3f}  "
                f"path_len={result.path_length:.2f}  min_sep={result.min_separation:.3f}"
            )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained CrowdNavPPPolicy")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-obstacle-encoder",
        action="store_true",
        help="Pass this if the checkpoint was trained WITHOUT the obstacle encoder.",
    )
    parser.add_argument("--episode-log", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    stats = evaluate(
        args.checkpoint,
        args.n_episodes,
        args.seed,
        args.device,
        verbose=args.verbose,
        use_obstacle_encoder=not args.no_obstacle_encoder,
        episode_log_path=args.episode_log,
    )
    stats.print_report()

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(stats.report(), f, indent=2)
        print(f"\nFull report written to {args.output_json}")


if __name__ == "__main__":
    main()
