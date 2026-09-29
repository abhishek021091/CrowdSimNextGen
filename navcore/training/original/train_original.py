"""
CLI entry point for training the original CrowdNav++ policy.

Examples
--------

# Train from official pretrained weights
python -m navcore.training.original.train_original \
    --checkpoint /path/to/41200.pt \
    --total-timesteps 5000000 \
    --n-envs 16 \
    --device cuda

# Resume PPO training
python -m navcore.training.original.train_original \
    --resume runs/original/latest.pt
"""

from __future__ import annotations

import argparse

from navcore.training.original.common import (
    EnvSettings,
    build_policy,
    make_vec_env,
    pick_device,
    resolve_policy_config,
    resolve_static_obstacles,
    set_global_seed,
)
from navcore.training.original.original_ppo_trainer import (
    OriginalPPOTrainer,
    PPOConfig,
)
from navcore.training.original.checkpoint import (
    inspect_checkpoint,
    load_policy_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Original CrowdNav++ with PPO")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--num-minibatch", type=int, default=1)

    # ------------------------------------------------------------------
    # PPO Hyperparameters
    # ------------------------------------------------------------------

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--clip-range-vf", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)

    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--no-grad-clip", action="store_true")

    parser.add_argument("--target-kl", type=float, default=None)

    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "linear", "cosine"],
        default="linear",
    )
    parser.add_argument("--lr-min-factor", type=float, default=0.0)

    parser.add_argument("--normalize-advantage", action="store_true", default=True)

    parser.add_argument("--amp", action="store_true")

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    parser.add_argument("--max-neighbors", type=int, default=10)
    parser.add_argument("--history-steps", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=1500)

    parser.add_argument(
        "--obstacle-mode",
        choices=["auto", "none", "point_tokens", "encoder"],
        default="auto",
    )

    parser.add_argument(
        "--static-obstacles",
        choices=["auto", "on", "off"],
        default="auto",
    )

    parser.add_argument("--obstacle-num-rays", type=int, default=60)
    parser.add_argument("--obstacle-max-range", type=float, default=5.0)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")

    parser.add_argument(
        "--checkpoint-dir",
        default="runs/original",
    )

    parser.add_argument("--checkpoint-every", type=int, default=200)

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Load pretrained weights before training.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume a PPO checkpoint.",
    )

    parser.add_argument("--metrics-path", default=None)

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    set_global_seed(args.seed)

    device = pick_device(args.device)

    checkpoint_info = None
    if args.checkpoint:
        _, checkpoint_info = inspect_checkpoint(args.checkpoint)

    policy_cfg = resolve_policy_config(
        obstacle_mode=args.obstacle_mode,
        max_neighbors=args.max_neighbors,
        obstacle_max_range=args.obstacle_max_range,
        checkpoint_info=checkpoint_info,
    )

    settings = EnvSettings(
        max_neighbors=policy_cfg.max_neighbors,
        history_steps=args.history_steps,
        max_episode_steps=args.max_episode_steps,
        static_obstacles=resolve_static_obstacles(
            args.static_obstacles,
            policy_cfg,
        ),
        obstacle_num_rays=args.obstacle_num_rays,
        obstacle_max_range=policy_cfg.obstacle_max_range,
    )

    env = make_vec_env(settings, args.n_envs)

    policy = build_policy(policy_cfg, device)

    # ------------------------------------------------------------------
    # Load pretrained weights
    # ------------------------------------------------------------------

    if args.checkpoint:
        info = load_policy_checkpoint(policy, args.checkpoint)
        print(f"Loaded pretrained checkpoint:\n{info.summary()}")

    # ------------------------------------------------------------------
    # PPO config
    # ------------------------------------------------------------------

    ppo_cfg = PPOConfig(
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        num_minibatch=args.num_minibatch,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        clip_range_vf=args.clip_range_vf,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=None if args.no_grad_clip else args.max_grad_norm,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        lr_min_factor=args.lr_min_factor,
        normalize_advantage=args.normalize_advantage,
        target_kl=args.target_kl,
        amp=args.amp,
        device=str(device),
    )

    trainer = OriginalPPOTrainer(
        env=env,
        policy=policy,
        config=ppo_cfg,
        metrics_path=args.metrics_path,
    )

    # ------------------------------------------------------------------
    # Resume training
    # ------------------------------------------------------------------

    if args.resume:
        info = trainer.load_training_state(args.resume)
        print(f"Resumed:\n{info.summary()}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    trainer.train(
        total_timesteps=args.total_timesteps,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
