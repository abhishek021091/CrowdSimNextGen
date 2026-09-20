# navcore/training/crowd_nav_pp/train_crowdnav_pp.py
"""CLI entry point: train CrowdNavPPPolicy with PPO against n_envs
parallel CrowdSimEnv copies.

Run directly:

    python -m navcore.training.train_crowdnav_pp --n-envs 16
"""

from __future__ import annotations

import argparse

import torch

from navcore.gym_wrapper.crowd_sim_env import ActionMode, CrowdSimEnv, CrowdSimEnvConfig
from navcore.gym_wrapper.goal_reaching_task import GoalReachingTask
from navcore.policies.crowdnav_pp.policy import CrowdNavPPPolicy, CrowdNavPPPolicyConfig
from navcore.training.crowd_nav_pp.ppo_trainer import CrowdNavPPTrainer, PPOConfig
from navcore.training.crowd_nav_pp.vec_env import VecCrowdSimEnv
from navcore.training.gst_predictor.gst_predictor_trainer import GSTPredictorTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CrowdNav++ via PPO")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--clip-range-vf", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--max-neighbors", type=int, default=10)
    parser.add_argument("--history-steps", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=500)
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
    args = parser.parse_args()

    if args.use_gst_prediction and not args.gst_checkpoint:
        parser.error("--use-gst-prediction requires --gst-checkpoint")

    env_config = CrowdSimEnvConfig(
        action_mode=ActionMode.VELOCITY,
        max_neighbors=args.max_neighbors,
        history_steps=args.history_steps,
        max_episode_steps=args.max_episode_steps,
    )

    # Each parallel slot needs its own CrowdSimEnv + GoalReachingTask instance
    # (GoalReachingTask carries per-episode state -- see its reset()) -- so
    # build fresh factories, not shared objects.
    env_fns = [
        (lambda: CrowdSimEnv(GoalReachingTask(), env_config))
        for _ in range(args.n_envs)
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
        CrowdNavPPPolicyConfig(use_gst_prediction=args.use_gst_prediction),
        gst_predictor=gst_predictor,
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
        max_grad_norm=args.max_grad_norm,
        device=args.device,
    )

    trainer = CrowdNavPPTrainer(env, policy, ppo_config, metrics_path=args.metrics_path)
    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(
        total_timesteps=args.total_timesteps,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
