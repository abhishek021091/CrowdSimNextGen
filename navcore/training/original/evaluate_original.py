# navcore/training/original/evaluate_original.py
"""Evaluate an original-port CrowdNav++ policy (41200.pt, official, adapter or
PPO checkpoints, any obstacle mode -- detected automatically).

    python -m navcore.training.original.evaluate_original            # 41200.pt
    python -m navcore.training.original.evaluate_original run/latest.pt --n-episodes 50
    python -m navcore.training.original.evaluate_original ckpt.pt --stochastic --video out.mp4

Reports success / collision / out-of-bounds / timeout rates, mean reward,
episode length, and FPS (policy-only inference FPS and end-to-end FPS).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

from navcore.training.original.checkpoint import (
    inspect_checkpoint,
    load_policy_checkpoint,
    resolve_pretrained,
)
from navcore.training.original.common import (
    EnvSettings,
    build_policy,
    make_env,
    pick_device,
    resolve_policy_config,
    resolve_static_obstacles,
    set_global_seed,
)

OUTCOMES = ("success", "collision", "out_of_bounds", "timeout")


def classify_outcome(info: dict) -> str:
    if info.get("collision"):
        return "collision"
    # if info.get("out_of_bounds"):
    #     return "out_of_bounds"
    if info.get("terminated"):
        return "success"
    return "timeout"


class VideoRecorder:
    """Headless (Agg) frame renderer -> mp4 (or gif if ffmpeg is unavailable).

    Uses the project's own sub-visualizers, in a robot-centred window, so it
    works without a display (the Qt-docking ``Visualizer`` does not).
    """

    def __init__(self, path: str, fps: int = 20, half_extent: float = 8.0) -> None:
        import imageio.v2 as imageio
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.path = str(path)
        self.half_extent = half_extent
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self.writer = imageio.get_writer(self.path, fps=fps)
        except Exception:  # noqa: BLE001 - no ffmpeg backend
            self.path = str(Path(self.path).with_suffix(".gif"))
            self.writer = imageio.get_writer(self.path, fps=fps)
            print(f"[video] mp4 writer unavailable, writing {self.path}")
        self.frames = 0

    def add(self, env, episode: int, step: int) -> None:
        from navcore.visualization.entities.crowd_visualizer import CrowdVisualizer
        from navcore.visualization.entities.obstacle_visualizer import (
            ObstacleVisualizer,
        )
        from navcore.visualization.entities.robot_visualizer import RobotVisualizer

        ax = self.ax
        ax.clear()
        pose = env.robot.pose
        cx, cy = pose.px, pose.py
        ax.set_xlim(cx - self.half_extent, cx + self.half_extent)
        ax.set_ylim(cy - self.half_extent, cy + self.half_extent)
        ax.set_aspect("equal")
        ObstacleVisualizer(env, ax).draw()
        CrowdVisualizer(env, ax).draw()
        RobotVisualizer(env, ax).draw()
        ax.set_title(f"episode {episode} step {step}")
        self.canvas.draw()
        self.writer.append_data(np.asarray(self.canvas.buffer_rgba())[..., :3].copy())
        self.frames += 1

    def close(self) -> None:
        self.writer.close()
        print(f"[video] wrote {self.frames} frames to {self.path}")


@torch.no_grad()
def evaluate(
    policy,
    env,
    n_episodes: int,
    *,
    seed: int = 0,
    deterministic: bool = True,
    device: torch.device = torch.device("cpu"),
    recorder: VideoRecorder | None = None,
    video_every: int = 1,
    video_episodes: int = 1,
    live_viz=None,
    verbose: bool = False,
) -> dict:
    was_training = policy.training
    policy.eval()
    if not deterministic:
        torch.manual_seed(seed)
    uses_rays = policy.config.uses_ray_features
    sync = (
        (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    )

    episodes = []
    policy_time = 0.0
    policy_calls = 0
    wall0 = time.perf_counter()
    total_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        hidden = policy.initial_hidden_state(nenv=1, device=device)
        not_done = torch.zeros(1, device=device)  # first tick resets hidden state
        ep_reward, steps, info = 0.0, 0, {}
        done = False
        record = recorder is not None and ep < video_episodes
        while not done:
            batch = {
                k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
                for k, v in obs.items()
            }
            sync()
            t = time.perf_counter()
            dist, _, hidden = policy.forward(
                batch["robot"],
                batch["neighbors"],
                batch["neighbor_mask"],
                batch["neighbor_history"],
                batch["neighbor_history_mask"],
                hidden,
                not_done,
                ray_features=batch["ray_features"] if uses_rays else None,
            )
            action = (
                dist.mean if deterministic else Normal(dist.mean, dist.stddev).sample()
            )
            sync()
            policy_time += time.perf_counter() - t
            policy_calls += 1
            not_done = torch.ones(1, device=device)

            obs, reward, terminated, truncated, info = env.step(
                action.squeeze(0).float().cpu().numpy()
            )
            ep_reward += float(reward)
            steps += 1
            done = terminated or truncated
            if record and steps % video_every == 0:
                recorder.add(env.env, ep, steps)
            if live_viz is not None:
                live_viz.refresh(env.env)
        total_steps += steps
        outcome = classify_outcome(info)
        episodes.append(
            {"outcome": outcome, "reward": ep_reward, "steps": steps, "seed": seed + ep}
        )
        if verbose:
            print(
                f"episode {ep:3d}: {outcome:13s} steps={steps:4d} reward={ep_reward:8.2f}",
                flush=True,
            )

    wall = time.perf_counter() - wall0
    n = max(len(episodes), 1)
    rewards = [e["reward"] for e in episodes]
    lengths = [e["steps"] for e in episodes]
    report = {
        "episodes": len(episodes),
        "deterministic": deterministic,
        **{f"{o}_rate": sum(e["outcome"] == o for e in episodes) / n for o in OUTCOMES},
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "std_reward": float(np.std(rewards)) if rewards else float("nan"),
        "mean_length": float(np.mean(lengths)) if lengths else float("nan"),
        "inference_fps": policy_calls / policy_time
        if policy_time > 0
        else float("nan"),
        "end_to_end_fps": total_steps / wall if wall > 0 else float("nan"),
        "per_episode": episodes,
    }
    if was_training:
        policy.train()
    return report


def print_report(r: dict) -> None:
    print(
        f"\nepisodes: {r['episodes']}  ({'deterministic' if r['deterministic'] else 'stochastic'})"
    )
    for o in OUTCOMES:
        print(f"  {o + ' rate':<20} {r[o + '_rate']:.1%}")
    print(f"  {'mean reward':<20} {r['mean_reward']:.2f} (std {r['std_reward']:.2f})")
    print(f"  {'mean episode length':<20} {r['mean_length']:.1f} steps")
    print(f"  {'inference FPS':<20} {r['inference_fps']:.0f} (policy forward only)")
    print(f"  {'end-to-end FPS':<20} {r['end_to_end_fps']:.0f} (policy + simulator)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Evaluate an original-port CrowdNav++ policy"
    )
    p.add_argument(
        "checkpoint", nargs="?", default=None, help="default: 41200.pt (searched)"
    )
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--obstacle-mode",
        default="auto",
        choices=["auto", "none", "point_tokens", "encoder"],
    )
    p.add_argument("--max-neighbors", type=int, default=None)
    p.add_argument("--static-obstacles", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--obstacle-num-rays", type=int, default=60)
    p.add_argument("--obstacle-max-range", type=float, default=5.0)
    p.add_argument("--history-steps", type=int, default=8)
    p.add_argument("--max-episode-steps", type=int, default=1500)
    p.add_argument(
        "--render", action="store_true", help="live window (needs a Qt display)"
    )
    p.add_argument("--video", default=None, help="write mp4/gif here (headless OK)")
    p.add_argument("--video-fps", type=int, default=20)
    p.add_argument("--video-every", type=int, default=1)
    p.add_argument("--video-episodes", type=int, default=1)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output-json", default=None)
    args = p.parse_args(argv)

    set_global_seed(args.seed)
    device = pick_device(args.device)
    ckpt = resolve_pretrained(args.checkpoint)
    _, info = inspect_checkpoint(ckpt)
    cfg = resolve_policy_config(
        obstacle_mode=args.obstacle_mode,
        max_neighbors=args.max_neighbors,
        obstacle_max_range=args.obstacle_max_range,
        checkpoint_info=info,
    )
    policy = build_policy(cfg, device)
    info = load_policy_checkpoint(policy, ckpt)
    print(f"loaded: {info.summary()}")

    settings = EnvSettings(
        max_neighbors=cfg.max_neighbors,
        history_steps=args.history_steps,
        max_episode_steps=args.max_episode_steps,
        static_obstacles=resolve_static_obstacles(args.static_obstacles, cfg),
        obstacle_num_rays=args.obstacle_num_rays,
        obstacle_max_range=cfg.obstacle_max_range,
    )
    env = make_env(settings)
    recorder = VideoRecorder(args.video, args.video_fps) if args.video else None
    live = None
    if args.render:
        try:
            from navcore.visualization.visualizer import Visualizer

            live = Visualizer()
        except Exception as e:  # noqa: BLE001
            print(
                f"[render] live viewer unavailable ({type(e).__name__}: {e}); continuing headless"
            )
    try:
        report = evaluate(
            policy,
            env,
            args.n_episodes,
            seed=args.seed,
            deterministic=not args.stochastic,
            device=device,
            recorder=recorder,
            video_every=args.video_every,
            video_episodes=args.video_episodes,
            live_viz=live,
            verbose=args.verbose,
        )
    finally:
        if recorder:
            recorder.close()
    print_report(report)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2))
        print(f"report written to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
