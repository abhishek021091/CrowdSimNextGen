# navcore/training/gst_predictor_trainer.py
"""GSTPredictorTrainer: supervised pretraining loop for GSTPredictor.

Matches the reference paper's two-stage workflow: trained once, offline,
against ground-truth pedestrian trajectories; frozen and loaded read-only
into CrowdNavPPPolicy afterward (see load_predictor) -- never jointly
trained with PPO.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from navcore.policies.gst_predictor.gst_predictor import (
    GSTPredictor,
    GSTPredictorConfig,
)
from navcore.policies.gst_predictor.prediction_head import gaussian_nll_loss
from navcore.training.gst_predictor.gst_data_collection import (
    GSTDataCollector,
    samples_to_batch,
)


@dataclass(slots=True)
class GSTTrainingConfig:
    learning_rate: float = 1e-3
    batch_size: int = 64
    n_epochs: int = 20
    episodes_per_collection: int = 20
    ticks_per_episode: int = 200
    max_agents: int = 20
    device: str = "cpu"


class GSTPredictorTrainer:
    def __init__(
        self,
        predictor_config: GSTPredictorConfig | None = None,
        training_config: GSTTrainingConfig | None = None,
        collector: GSTDataCollector | None = None,
    ) -> None:
        self.predictor_config = predictor_config or GSTPredictorConfig()
        self.config = training_config or GSTTrainingConfig()
        self.device = torch.device(self.config.device)

        self.predictor = GSTPredictor(self.predictor_config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=self.config.learning_rate
        )
        self.collector = collector or GSTDataCollector(
            max_agents=self.config.max_agents,
            obs_length=self.predictor_config.obs_length,
            pred_length=self.predictor_config.pred_length,
        )

    def _iterate_batches(self, samples):
        n = len(samples)
        indices = torch.randperm(n)
        for start in range(0, n, self.config.batch_size):
            batch_idx = indices[start : start + self.config.batch_size]
            yield [samples[i] for i in batch_idx.tolist()]

    def train_epoch(self, samples) -> float:
        self.predictor.train()
        total_loss = 0.0
        n_batches = 0

        for batch_samples in self._iterate_batches(samples):
            batch = samples_to_batch(batch_samples, self.device)
            mean, log_var = self.predictor(
                batch["history_positions"],
                batch["history_velocity"],
                batch["history_mask"],
            )
            loss = gaussian_nll_loss(
                mean, log_var, batch["future_displacement"], batch["future_mask"]
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self, log_every: int = 1) -> None:
        for epoch in range(self.config.n_epochs):
            samples = self.collector.collect(
                n_episodes=self.config.episodes_per_collection,
                ticks_per_episode=self.config.ticks_per_episode,
            )
            mean_loss = self.train_epoch(samples)
            if epoch % log_every == 0:
                print(f"epoch={epoch} samples={len(samples)} mean_nll={mean_loss:.4f}")

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "predictor_state_dict": self.predictor.state_dict(),
                "predictor_config": self.predictor_config,
            },
            path,
        )

    @staticmethod
    def load_predictor(path: str, device: str = "cpu") -> GSTPredictor:
        """Load a pretrained GSTPredictor, frozen for use inside CrowdNavPPPolicy."""
        checkpoint = torch.load(path, map_location=device)
        predictor = GSTPredictor(checkpoint["predictor_config"])
        predictor.load_state_dict(checkpoint["predictor_state_dict"])
        predictor.to(device)
        predictor.eval()
        for p in predictor.parameters():
            p.requires_grad_(False)
        return predictor
