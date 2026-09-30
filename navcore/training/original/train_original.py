"""CLI entry point for training the original-port CrowdNav++ policy with PPO."""

from __future__ import annotations

import argparse
import dataclasses
import warnings
from pathlib import Path

from navcore.training.original.checkpoint import (
    describe_optimizer_state,
    inspect_checkpoint,
    load_policy_checkpoint,
)
from navcore.training.original.common import (
    EnvSettings,
    build_policy,
    make_env,
    make_vec_env,
    pick_device,
    resolve_policy_config,
    resolve_ray_spec,  # new
    resolve_static_obstacles,
    set_global_seed,
    verify_ray_wiring,  # new
)
from navcore.training.original.original_trainer import (  # was: original_ppo_trainer
    OriginalPPOTrainer,
    PPOConfig,
)


def resolve_metrics_path(raw: str | None) -> Path | None:
    """A path without suffix (or an existing dir) is a directory -> metrics.jsonl inside."""
    if raw is None:
        return None
    p = Path(raw)
    if p.is_dir() or p.suffix == "":
        p = p / "metrics.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train original CrowdNav++ with PPO")
    # training
    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--num-minibatch", type=int, default=1)
    # PPO
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--clip-range-vf", type=float, default=0.2)
    p.add_argument("--no-vf-clip", action="store_true")
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--no-grad-clip", action="store_true")
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument(
        "--lr-schedule", choices=["constant", "linear", "cosine"], default="linear"
    )
    p.add_argument("--lr-min-factor", type=float, default=0.0)
    p.add_argument("--no-normalize-advantage", action="store_true")
    p.add_argument("--amp", action="store_true")
    # environment / policy (None => take from checkpoint, else default)
    p.add_argument("--max-neighbors", type=int, default=None)
    p.add_argument("--history-steps", type=int, default=8)
    p.add_argument("--max-episode-steps", type=int, default=1500)
    p.add_argument(
        "--obstacle-mode",
        choices=["auto", "none", "point_tokens", "encoder"],
        default="auto",
    )
    p.add_argument("--static-obstacles", choices=["auto", "on", "off"], default="auto")
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
    p.add_argument("--obstacle-hit-radius", type=float, default=None)
    p.add_argument("--use-gst-prediction", action="store_true")
    p.add_argument("--gst-checkpoint", type=str, default=None)
    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--checkpoint-dir", default="runs/original")
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Initial weights (official/adapter/PPO); counters and optimizer stay fresh.",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="PPO checkpoint: restores weights, optimizer, counters.",
    )
    p.add_argument(
        "--metrics-path",
        default=None,
        help="JSONL file, or a directory (-> metrics.jsonl).",
    )
    p.add_argument("--tensorboard-dir", default=None)
    p.add_argument(
        "--eval-every", type=int, default=0, help="Run eval every N updates (0 = off)."
    )
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--render", action="store_true", help="Render env 0 only.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = pick_device(args.device)

    if args.resume and args.checkpoint:
        warnings.warn(
            "--resume and --checkpoint both given; --resume wins, --checkpoint ignored."
        )
        args.checkpoint = None

    # Policy config source: resume checkpoint (restores its policy_config), else pretrained.
    info = None
    source = args.resume or args.checkpoint
    if source:
        _, info = inspect_checkpoint(source)

    ray_spec = resolve_ray_spec(args.obstacle_num_rays, args.obstacle_max_range)

    policy_cfg = resolve_policy_config(
        obstacle_mode=args.obstacle_mode,
        max_neighbors=args.max_neighbors,
        ray_spec=ray_spec,
        checkpoint_info=info,
        obstacle_hit_radius=args.obstacle_hit_radius,
    )
    if args.use_gst_prediction:
        policy_cfg = dataclasses.replace(policy_cfg, use_gst_prediction=True)

    gst_predictor = None
    if policy_cfg.use_gst_prediction:
        if not args.gst_checkpoint:
            parser.error(
                "GST prediction is enabled but --gst-checkpoint was not given."
            )
        from navcore.training.gst_predictor.gst_predictor_trainer import (
            GSTPredictorTrainer,
        )

        gst_predictor = GSTPredictorTrainer.load_predictor(
            args.gst_checkpoint, device=str(device)
        )

    settings = EnvSettings(
        max_neighbors=policy_cfg.max_neighbors,
        history_steps=args.history_steps,
        max_episode_steps=args.max_episode_steps,
        static_obstacles=resolve_static_obstacles(args.static_obstacles, policy_cfg),
        ray_spec=ray_spec,
    )
    if settings.static_obstacles and not policy_cfg.uses_ray_features:
        warnings.warn(
            "Static obstacles are ON but the policy has obstacle_mode=NONE: it cannot see them."
        )

    env = make_vec_env(settings, args.n_envs, render=args.render)
    policy = build_policy(policy_cfg, device, gst_predictor=gst_predictor)
    verify_ray_wiring(env, policy)

    if args.checkpoint:
        info = load_policy_checkpoint(policy, args.checkpoint)  # prints report
        print(describe_optimizer_state(info, restored=None))

    ppo_cfg = PPOConfig(
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        num_minibatch=args.num_minibatch,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        clip_range_vf=None if args.no_vf_clip else args.clip_range_vf,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=None if args.no_grad_clip else args.max_grad_norm,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        lr_min_factor=args.lr_min_factor,
        normalize_advantage=not args.no_normalize_advantage,
        target_kl=args.target_kl,
        amp=args.amp,
        device=str(device),
    )

    writer = None
    if args.tensorboard_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise SystemExit(
                "--tensorboard-dir needs `pip install tensorboard`."
            ) from e
        Path(args.tensorboard_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_dir)

    trainer = OriginalPPOTrainer(
        env,
        policy,
        ppo_cfg,
        writer=writer,
        metrics_path=resolve_metrics_path(args.metrics_path),
        render=args.render,
        seed=args.seed,
    )
    if args.resume:
        trainer.load_training_state(args.resume)  # prints report + optimizer note

    eval_fn = None
    if args.eval_every > 0:
        from navcore.training.original.evaluate_original import evaluate

        eval_env = make_env(settings)

        def eval_fn():
            r = evaluate(
                policy,
                eval_env,
                args.eval_episodes,
                seed=args.seed + 10_000,
                device=device,
            )
            return {k: v for k, v in r.items() if isinstance(v, float)}

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    trainer.train(
        total_timesteps=args.total_timesteps,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        eval_fn=eval_fn,
        eval_every=args.eval_every,
    )


if __name__ == "__main__":
    main()
