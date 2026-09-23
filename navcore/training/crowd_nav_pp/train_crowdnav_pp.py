# navcore/training/crowd_nav_pp/train_crowdnav_pp.py
"""CLI entry point: train CrowdNavPPPolicy with PPO against n_envs
parallel CrowdSimEnv copies.

Run directly:

    python -m navcore.training.train_crowdnav_pp --n-envs 16
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import tomllib
import torch

import navcore.configs
from navcore.gym_wrapper.crowd_sim_env import ActionMode, CrowdSimEnv, CrowdSimEnvConfig
from navcore.gym_wrapper.goal_reaching_task import GoalReachingTask
from navcore.policies.crowdnav_pp.policy import CrowdNavPPPolicy, CrowdNavPPPolicyConfig
from navcore.training.crowd_nav_pp.crowd_nav_pp_trainer import (
    CrowdNavPPTrainer,
    PPOConfig,
)
from navcore.training.crowd_nav_pp.vec_env import VecCrowdSimEnv
from navcore.training.gst_predictor.gst_predictor_trainer import GSTPredictorTrainer


def _default_seed_from_toml() -> int:
    """Fall back to env.toml's [random] seed when --seed isn't given."""
    env_path = Path(navcore.configs.__file__).parent / "env.toml"
    with open(env_path, "rb") as f:
        config = tomllib.load(f)
    return int(config["random"]["seed"])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CrowdNav++ via PPO")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.15)
    parser.add_argument("--clip-range-vf", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.00)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--no-grad-clip", action="store_true", default=False)
    parser.add_argument("--max-neighbors", type=int, default=10)
    parser.add_argument("--history-steps", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=1500)
    parser.add_argument("--use-obstacle-encoder", action="store_true", default=False)
    parser.add_argument("--obstacle-num-rays", type=int, default=60)
    parser.add_argument("--obstacle-max-range", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=_default_seed_from_toml())
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./navcore/training/crowd_nav_pp/checkpoints",
    )
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--use-gst-prediction", action="store_true", default=False)
    parser.add_argument("--gst-checkpoint", type=str, default=None)
    parser.add_argument("--metrics-path", type=str, default=None)
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Enable visualization for the first environment.",
    )
    args = parser.parse_args()
    log_run_config(args, args.checkpoint_dir)
    set_seed(args.seed)

    if args.use_gst_prediction and not args.gst_checkpoint:
        parser.error("--use-gst-prediction requires --gst-checkpoint")

    env_config = CrowdSimEnvConfig(
        action_mode=ActionMode.VELOCITY,
        max_neighbors=args.max_neighbors,
        history_steps=args.history_steps,
        max_episode_steps=args.max_episode_steps,
        obstacle_num_rays=args.obstacle_num_rays,
        obstacle_max_range=args.obstacle_max_range,
    )

    # Each parallel slot needs its own CrowdSimEnv + GoalReachingTask instance
    # (GoalReachingTask carries per-episode state -- see its reset()) -- so
    # build fresh factories, not shared objects.
    env_fns = [
        (
            lambda i=i: CrowdSimEnv(
                GoalReachingTask(),
                env_config,
                render_mode="human" if args.render and i == 0 else None,
            )
        )
        for i in range(args.n_envs)
    ]
    env = VecCrowdSimEnv(env_fns)
    gst_predictor = None
    if args.use_gst_prediction:
        gst_predictor = GSTPredictorTrainer.load_predictor(
            args.gst_checkpoint, device=args.device
        )

    # Defaults already match ObservationEncoder's actual feature widths
    # (robot_feature_dim=8, neighbor_feature_dim=5) -- see policy.py's
    # _NEIGHBOR_MOTION_SLICE comment for the same real coupling point.
    policy = CrowdNavPPPolicy(
        CrowdNavPPPolicyConfig(
            use_gst_prediction=args.use_gst_prediction,
            # use_obstacle_encoder=args.use_obstacle_encoder,
        ),
        gst_predictor=gst_predictor,
        # obstacle_encoder=obstacle_encoder,
    )

    ppo_config = PPOConfig(
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        clip_range_vf=args.clip_range_vf,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=None if args.no_grad_clip else args.max_grad_norm,
        device=args.device,
    )

    trainer = CrowdNavPPTrainer(
        env,
        policy,
        ppo_config,
        metrics_path=args.metrics_path,
        render=args.render,
        seed=args.seed,
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(
        total_timesteps=args.total_timesteps,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
    )


def log_run_config(args, checkpoint_dir: str) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    record = {
        "command": " ".join(sys.argv),
        "args": vars(args),
    }
    config_path = os.path.join(checkpoint_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"Run config saved to {config_path}")


if __name__ == "__main__":
    main()
