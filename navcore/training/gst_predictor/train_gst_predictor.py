# navcore/training/train_gst_predictor.py
"""CLI entry point: pretrain a GSTPredictor on ORCA-only crowd trajectories.

Run directly:

    python -m navcore.training.train_gst_predictor
"""

from __future__ import annotations

import argparse

import torch

from navcore.policies.gst_predictor.gst_predictor import GSTPredictorConfig
from navcore.training.gst_predictor.gst_predictor_trainer import (
    GSTPredictorTrainer,
    GSTTrainingConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain GST trajectory predictor")
    parser.add_argument("--obs-length", type=int, default=8)
    parser.add_argument("--pred-length", type=int, default=5)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--episodes-per-collection", type=int, default=20)
    parser.add_argument("--ticks-per-episode", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./navcore/training/gst_predictor/checkpoints/gst_predictor.pt",
    )
    args = parser.parse_args()

    predictor_config = GSTPredictorConfig(
        obs_length=args.obs_length, pred_length=args.pred_length
    )
    training_config = GSTTrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        episodes_per_collection=args.episodes_per_collection,
        ticks_per_episode=args.ticks_per_episode,
        device=args.device,
    )

    trainer = GSTPredictorTrainer(predictor_config, training_config)
    trainer.train()
    trainer.save(args.output)
    print(f"Saved pretrained GST predictor to {args.output}")


if __name__ == "__main__":
    main()
