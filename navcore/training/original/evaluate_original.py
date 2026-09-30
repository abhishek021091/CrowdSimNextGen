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
import math
import time
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import Point
from torch.distributions import Normal

from navcore.entities.agents.robot import Robot
from navcore.entities.obstacles.geometry_conversion import (
    arena_boundary_ring,
    obstacle_to_shapely_polygon,
)
from navcore.step.step import Step
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
    resolve_ray_spec,  # new
    resolve_static_obstacles,
    set_global_seed,
    verify_ray_wiring,  # new
)

OUTCOMES = ("success", "collision", "out_of_bounds", "timeout")


def classify_outcome(info: dict) -> str:
    if info.get("collision"):
        return "collision"
    if info.get("out_of_bounds"):
        return "out_of_bounds"
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


DEGENERATE_STEPS = 2  # episodes this short ended at spawn, not through behaviour


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        nan = float("nan")
        return dict.fromkeys(("mean", "std", "min", "p10", "p50", "p90", "max"), nan)
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()) if len(a) > 1 else 0.0,
        "min": float(a.min()),
        "p10": float(np.percentile(a, 10)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
    }


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a rate; with n=50 the error bar is large."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _tick_geometry(base, polys, ring) -> tuple[float, float, bool]:
    """(min pedestrian surface separation, min obstacle/wall clearance, hit_ped)."""
    robot = base.robot
    px, py = robot.pose.px, robot.pose.py
    min_ped = float("inf")
    for ped in base.crowd.values():
        if ped.pose is None:
            continue
        sep = math.hypot(px - ped.pose.px, py - ped.pose.py) - robot.radius - ped.radius
        min_ped = min(min_ped, sep)
    pt = Point(px, py)
    clear = pt.distance(ring) - robot.radius
    for poly in polys:
        clear = min(clear, pt.distance(poly) - robot.radius)
    # Same criterion CollisionChecker uses for pedestrians.
    return min_ped, clear, min_ped < base.info.pedestrian_safety_distance


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
    v_max = float(Robot.config["kinematics"]["v_pref"])

    episodes: list[dict] = []
    policy_time, policy_calls, total_steps = 0.0, 0, 0
    wall0 = time.perf_counter()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        hidden = policy.initial_hidden_state(nenv=1, device=device)
        not_done = torch.zeros(1, device=device)

        base = env.env
        polys = [
            obstacle_to_shapely_polygon(o)
            for k, o in base.obstacles.items()
            if k != "boundary"
        ]
        ring = arena_boundary_ring(base)
        robot = base.robot
        start_dist = math.hypot(
            robot.goal.gx - robot.pose.px, robot.goal.gy - robot.pose.py
        )
        start_ped_sep, start_clear, _ = _tick_geometry(base, polys, ring)
        prev = (robot.pose.px, robot.pose.py)

        ep_reward, steps, info, done = 0.0, 0, {}, False
        path_len = 0.0
        min_ped, min_clear = float("inf"), float("inf")
        tick_ped_mins: list[float] = []
        cmd_speeds: list[float] = []
        saturated = 0
        hit_ped = False
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

            action_np = action.squeeze(0).float().cpu().numpy()
            speed = math.hypot(float(action_np[0]), float(action_np[1]))
            cmd_speeds.append(speed)
            saturated += speed > v_max + 1e-9

            obs, reward, terminated, truncated, info = env.step(action_np)
            ep_reward += float(reward)
            steps += 1
            done = terminated or truncated

            base = env.env
            pose = base.robot.pose
            path_len += math.hypot(pose.px - prev[0], pose.py - prev[1])
            prev = (pose.px, pose.py)
            tick_ped, tick_clear, hit_ped = _tick_geometry(base, polys, ring)
            if math.isfinite(tick_ped):
                min_ped = min(min_ped, tick_ped)
                tick_ped_mins.append(tick_ped)
            min_clear = min(min_clear, tick_clear)

            if record and steps % video_every == 0:
                recorder.add(base, ep, steps)
            if live_viz is not None:
                live_viz.refresh(base)

        total_steps += steps
        outcome = classify_outcome(info)
        robot = env.env.robot
        final_dist = math.hypot(
            robot.goal.gx - robot.pose.px, robot.goal.gy - robot.pose.py
        )
        cause = None
        if outcome == "collision":
            cause = "pedestrian" if hit_ped else "obstacle_or_wall"

        rec = {
            "outcome": outcome,
            "collision_cause": cause,
            "degenerate": steps <= DEGENERATE_STEPS,
            "reward": ep_reward,
            "steps": steps,
            "seed": seed + ep,
            "start_distance": start_dist,
            "final_distance": final_dist,
            "path_length": path_len,
            "path_efficiency": (start_dist / path_len) if path_len > 1e-6 else None,
            "time_to_goal": steps * Step.dt if outcome == "success" else None,
            "start_ped_separation": start_ped_sep
            if math.isfinite(start_ped_sep)
            else None,
            "start_obstacle_clearance": start_clear,
            "min_ped_separation": min_ped if math.isfinite(min_ped) else None,
            "mean_ped_separation": float(np.mean(tick_ped_mins))
            if tick_ped_mins
            else None,
            "min_obstacle_clearance": min_clear,
            "mean_commanded_speed": float(np.mean(cmd_speeds)),
            "mean_actual_speed": path_len / (steps * Step.dt),
            "saturation_rate": saturated / steps,
        }
        episodes.append(rec)
        if verbose:
            print(
                f"ep {ep:3d} seed={rec['seed']:<4d} {outcome:9s}"
                f"{'/' + cause if cause else '':<18s} steps={steps:4d} "
                f"R={ep_reward:7.2f} dist {start_dist:5.1f}->{final_dist:5.1f} "
                f"path={path_len:6.1f} min_ped={min_ped:6.2f} "
                f"min_obs={min_clear:6.2f}{'  [degenerate]' if rec['degenerate'] else ''}",
                flush=True,
            )

    wall = time.perf_counter() - wall0
    report = _build_report(
        episodes, deterministic, policy_calls, policy_time, total_steps, wall
    )
    if was_training:
        policy.train()
    return report


def _build_report(
    episodes, deterministic, policy_calls, policy_time, total_steps, wall
):
    n = max(len(episodes), 1)
    nan = float("nan")

    def vals(key, where=lambda e: True):
        return [e[key] for e in episodes if where(e) and e[key] is not None]

    def rate(group):
        g = [e for e in group]
        m = max(len(g), 1)
        return {o: sum(e["outcome"] == o for e in g) / m for o in OUTCOMES}

    valid = [e for e in episodes if not e["degenerate"]]
    successes = [e for e in episodes if e["outcome"] == "success"]
    lo, hi = _wilson(len(successes), len(episodes))
    causes = {
        c: sum(e["collision_cause"] == c for e in episodes)
        for c in ("pedestrian", "obstacle_or_wall")
    }
    valid_rates = rate(valid)

    return {
        # -- top-level scalars (consumed by train_original's eval hook) --
        "episodes": len(episodes),
        "deterministic": deterministic,
        **{f"{o}_rate": sum(e["outcome"] == o for e in episodes) / n for o in OUTCOMES},
        "mean_reward": float(np.mean(vals("reward"))) if episodes else nan,
        "std_reward": float(np.std(vals("reward"))) if episodes else nan,
        "mean_length": float(np.mean(vals("steps"))) if episodes else nan,
        "inference_fps": policy_calls / policy_time if policy_time > 0 else nan,
        "end_to_end_fps": total_steps / wall if wall > 0 else nan,
        # -- new --
        "success_rate_ci95": [lo, hi],
        "degenerate_episodes": len(episodes) - len(valid),
        "rates_excluding_degenerate": valid_rates,
        "collision_causes": causes,
        "reward": _stats(vals("reward")),
        "reward_by_outcome": {
            o: _stats(vals("reward", lambda e, o=o: e["outcome"] == o))
            for o in OUTCOMES
        },
        "steps": _stats(vals("steps")),
        "path_length": _stats(vals("path_length")),
        "start_distance": _stats(vals("start_distance")),
        "final_distance_by_outcome": {
            o: _stats(vals("final_distance", lambda e, o=o: e["outcome"] == o))
            for o in ("collision", "timeout")
        },
        "time_to_goal_s": _stats(vals("time_to_goal")),
        "path_efficiency_success": _stats(
            vals("path_efficiency", lambda e: e["outcome"] == "success")
        ),
        "separation": {
            "ped_min_ever": min(vals("min_ped_separation"), default=nan),
            "ped_episode_min": _stats(vals("min_ped_separation")),
            "ped_episode_mean": _stats(vals("mean_ped_separation")),
            "obstacle_clearance_episode_min": _stats(vals("min_obstacle_clearance")),
            "close_call_rate_ped_lt_0.2m": sum(
                (e["min_ped_separation"] or 9) < 0.2 and e["outcome"] != "collision"
                for e in episodes
            )
            / n,
        },
        "policy_behavior": {
            "mean_commanded_speed": float(np.mean(vals("mean_commanded_speed"))),
            "mean_actual_speed": float(np.mean(vals("mean_actual_speed"))),
            "mean_saturation_rate": float(np.mean(vals("saturation_rate"))),
        },
        "per_episode": episodes,
    }


def print_report(r: dict) -> None:
    def row(label, b):
        print(
            f"  {label:<26} mean={b['mean']:8.2f} p10={b['p10']:8.2f} "
            f"p50={b['p50']:8.2f} p90={b['p90']:8.2f} min={b['min']:8.2f} max={b['max']:8.2f}"
        )

    mode = "deterministic" if r["deterministic"] else "stochastic"
    print(f"\nepisodes: {r['episodes']} ({mode})")
    lo, hi = r["success_rate_ci95"]
    print("\noutcomes:")
    for o in OUTCOMES:
        extra = f"  (95% CI {lo:.0%}-{hi:.0%})" if o == "success" else ""
        print(f"  {o:<14} {r[o + '_rate']:6.1%}{extra}")
    c = r["collision_causes"]
    print(
        f"  collisions by cause: pedestrian={c['pedestrian']}  obstacle/wall={c['obstacle_or_wall']}"
    )
    if r["degenerate_episodes"]:
        v = r["rates_excluding_degenerate"]
        print(
            f"\n  !! {r['degenerate_episodes']} degenerate episode(s) (<= {DEGENERATE_STEPS} steps: "
            f"spawn collision or spawn-at-goal). Excluding them:\n"
            f"     success={v['success']:.1%} collision={v['collision']:.1%} "
            f"timeout={v['timeout']:.1%}"
        )

    print("\nreward:")
    row("overall", r["reward"])
    for o in OUTCOMES:
        if r[o + "_rate"] > 0:
            row(f"| {o}", r["reward_by_outcome"][o])

    print("\nepisode shape:")
    row("steps", r["steps"])
    row("start_distance (m)", r["start_distance"])
    row("path_length (m)", r["path_length"])
    row("time_to_goal (s, success)", r["time_to_goal_s"])
    row("path_efficiency (success)", r["path_efficiency_success"])
    for o, b in r["final_distance_by_outcome"].items():
        if r[o + "_rate"] > 0:
            row(f"final_dist on {o} (m)", b)

    s = r["separation"]
    print("\nsafety:")
    print(f"  min pedestrian separation ever : {s['ped_min_ever']:.3f} m")
    row("ped sep, episode min", s["ped_episode_min"])
    row("ped sep, episode mean", s["ped_episode_mean"])
    row("obstacle clearance, min", s["obstacle_clearance_episode_min"])
    print(
        f"  close calls (<0.2 m, no collision): {s['close_call_rate_ped_lt_0.2m']:.1%}"
    )

    b = r["policy_behavior"]
    print("\npolicy behaviour:")
    print(
        f"  commanded speed {b['mean_commanded_speed']:.3f} m/s | "
        f"actual {b['mean_actual_speed']:.3f} m/s | saturation {b['mean_saturation_rate']:.1%}"
    )
    print(
        f"\nspeed: inference {r['inference_fps']:.0f} FPS (policy only), "
        f"{r['end_to_end_fps']:.0f} FPS end-to-end"
    )


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
    p.add_argument(
        "--obstacle-num-rays",
        type=int,
        default=None,
        help="Override env.toml [obstacle_sensor].num_rays",
    )
    p.add_argument(
        "--obstacle-max-range",
        type=float,
        default=None,
        help="Override env.toml [obstacle_sensor].max_range",
    )
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

    ray_spec = resolve_ray_spec(args.obstacle_num_rays, args.obstacle_max_range)

    cfg = resolve_policy_config(
        obstacle_mode=args.obstacle_mode,
        max_neighbors=args.max_neighbors,
        ray_spec=ray_spec,
        checkpoint_info=info,
    )
    policy = build_policy(cfg, device)

    settings = EnvSettings(
        max_neighbors=cfg.max_neighbors,
        history_steps=args.history_steps,
        max_episode_steps=args.max_episode_steps,
        static_obstacles=resolve_static_obstacles(args.static_obstacles, cfg),
        ray_spec=ray_spec,
    )
    env = make_env(settings)

    verify_ray_wiring(env, policy)  # fail here, with component names, not mid-episode

    info = load_policy_checkpoint(policy, ckpt)
    print(f"loaded: {info.summary()}")
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
