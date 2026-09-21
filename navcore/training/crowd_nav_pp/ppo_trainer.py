# navcore/training/crowd_nav_pp/ppo_trainer.py
"""CrowdNavPPTrainer: PPO training loop for CrowdNavPPPolicy against a
vectorized CrowdSimEnv (VecCrowdSimEnv, n_envs parallel episodes per rollout).

See rollout_buffer.py's module docstring for the Slice 2 scope (n_envs
parallel trajectories, whole-per-env-trajectory recompute per PPO epoch, no
chunked BPTT). This module owns everything else PPO needs on top of that:
action-mode wiring (must be ActionMode.VELOCITY -- CrowdNav++ predicts
velocity directly; see CrowdSimEnv's own module docstring for why WAYPOINT
mode wouldn't exercise the policy's collision-avoidance behavior at all),
vectorized rollout collection, the clipped-surrogate update, and
checkpointing.

Action log-prob convention:
    CrowdSimEnv._decode_velocity_action clips a raw (vx, vy) sample's
    *magnitude* to v_pref before applying it to the simulator, but
    log-probs stored here are computed on the raw, pre-clip sample --
    the standard PPO-on-continuous-control convention. Re-deriving a
    log-prob for the clipped action would require a change-of-variables
    correction this project has no need for; every existing recurrent-
    PPO implementation this policy was ported from (see policy.py's
    module docstring) takes the same shortcut.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from navcore.analysis.metrics_logger import TrainingMetricsLogger
from navcore.policies.crowdnav_pp.policy import CrowdNavPPPolicy
from navcore.training.crowd_nav_pp.rollout_buffer import RecurrentRolloutBuffer
from navcore.training.crowd_nav_pp.vec_env import VecCrowdSimEnv


@dataclass(slots=True)
class PPOConfig:
    """PPO hyperparameters.

    Attributes:
        n_steps: Ticks collected per env, per rollout, before each update
            phase (total rollout batch size is n_steps * n_envs).
        n_epochs: Full-trajectory recompute passes per rollout (see
            module docstring -- there is no time-axis minibatching here,
            only repeated whole-sequence passes).
        gamma: Discount factor.
        gae_lambda: GAE's bias/variance trade-off parameter.
        clip_range: PPO's surrogate-objective clip epsilon.
        clip_range_vf: Optional value-function clip epsilon. None
            disables value clipping (plain MSE against returns).
        ent_coef: Entropy-bonus weight, encouraging exploration.
        vf_coef: Value-loss weight in the combined objective.
        max_grad_norm: Global gradient-norm clip applied before each
            optimizer step.
        learning_rate: Adam learning rate.
        normalize_advantage: Whether to standardize advantages
            (zero mean, unit std, over the whole T*n_envs batch) before
            computing the surrogate loss.
        device: Torch device string ("cpu" or "cuda").
    """

    n_steps: int = 512
    n_epochs: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float | None = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    normalize_advantage: bool = True
    device: str = "cpu"


def _to_tensor_batch(
    obs: dict[str, np.ndarray], device: torch.device
) -> dict[str, Tensor]:
    """Convert one step's already n_envs-batched observation to tensors."""
    return {
        k: torch.as_tensor(v, dtype=torch.float32, device=device)
        for k, v in obs.items()
    }


class CrowdNavPPTrainer:
    """Drives PPO training of a CrowdNavPPPolicy against a VecCrowdSimEnv.

    Attributes:
        env: The vectorized training environment (n_envs parallel
            CrowdSimEnv copies). Must be configured with
            ActionMode.VELOCITY -- see module docstring.
        n_envs: env.n_envs, cached for convenience.
        policy: The recurrent actor-critic being trained.
        config: PPO hyperparameters.
    """

    def __init__(
        self,
        env: VecCrowdSimEnv,
        policy: CrowdNavPPPolicy,
        config: PPOConfig | None = None,
        metrics_path: str | Path | None = None,
    ) -> None:
        from navcore.gym_wrapper.crowd_sim_env import ActionMode

        if env.config.action_mode is not ActionMode.VELOCITY:
            obs_space = env.observation_space
            expected_robot_dim = obs_space["robot"].shape[-1]
            expected_neighbor_dim = obs_space["neighbors"].shape[-1]
            if policy.config.robot_feature_dim != expected_robot_dim:
                raise ValueError(
                    f"policy.config.robot_feature_dim={policy.config.robot_feature_dim} "
                    f"does not match env's robot observation width "
                    f"{expected_robot_dim} -- ObservationEncoder and "
                    f"CrowdNavPPPolicyConfig have drifted apart."
                )
            if policy.config.neighbor_feature_dim != expected_neighbor_dim:
                raise ValueError(
                    f"policy.config.neighbor_feature_dim="
                    f"{policy.config.neighbor_feature_dim} does not match env's "
                    f"neighbor observation width {expected_neighbor_dim}."
                )

        self.env = env
        self.n_envs = env.n_envs
        self.policy = policy
        self.config = config or PPOConfig()
        self.device = torch.device(self.config.device)

        self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate
        )

        self.buffer = RecurrentRolloutBuffer(device=self.device)

        # Dependency-free JSONL logger -- training loop stays headless.
        # Plot offline via navcore.analysis.training_curves.TrainingCurvePlotter.
        self.metrics_logger = (
            TrainingMetricsLogger(Path(metrics_path))
            if metrics_path is not None
            else None
        )
        self._obs: dict[str, np.ndarray] | None = None
        self._hidden_state: Tensor = self.policy.initial_hidden_state(
            nenv=self.n_envs, device=self.device
        )
        # Forces not_done_mask=0.0 (hidden-state reset) on every env's very
        # first tick ever collected, same as a real episode boundary.
        self._prev_done: np.ndarray = np.ones(self.n_envs, dtype=bool)

        self.total_steps = 0
        self.total_updates = 0

    # -- rollout collection ---------------------------------------------------

    def collect_rollout(self) -> dict[str, float]:
        self.policy.eval()
        self.buffer.start(self._hidden_state)

        if self._obs is None:
            self._obs = self.env.reset()

        episode_rewards: list[float] = []
        episode_reward = np.zeros(self.n_envs, dtype=np.float32)
        use_obstacle_encoder = self.policy.config.use_obstacle_encoder

        for _ in range(self.config.n_steps):
            not_done_mask = np.where(self._prev_done, 0.0, 1.0).astype(np.float32)
            obs_t = _to_tensor_batch(self._obs, self.device)
            not_done_mask_t = torch.as_tensor(
                not_done_mask, dtype=torch.float32, device=self.device
            )

            extra_kwargs = {}
            if use_obstacle_encoder:
                extra_kwargs["ray_features"] = obs_t["ray_features"]

            with torch.no_grad():
                action_t, log_prob_t, value_t, new_hidden = self.policy.act(
                    obs_t["robot"],
                    obs_t["neighbors"],
                    obs_t["neighbor_mask"],
                    obs_t["neighbor_history"],
                    obs_t["neighbor_history_mask"],
                    self._hidden_state,
                    not_done_mask_t,
                    deterministic=False,
                    **extra_kwargs,
                )

            action_np = action_t.cpu().numpy().astype(np.float32)
            next_obs, reward, done, _infos = self.env.step(action_np)

            self.buffer.add(
                obs=self._obs,
                not_done_mask=not_done_mask,
                action=action_np,
                log_prob=log_prob_t.cpu().numpy().astype(np.float32),
                value=value_t.squeeze(-1).cpu().numpy().astype(np.float32),
                reward=reward.astype(np.float32),
                done=done,
            )

            episode_reward += reward
            self.total_steps += self.n_envs
            self._hidden_state = new_hidden
            self._prev_done = done

            for env_idx in np.nonzero(done)[0]:
                episode_rewards.append(float(episode_reward[env_idx]))
                episode_reward[env_idx] = 0.0
                # env's own self._hidden_state reset happens via
                # not_done_mask on the *next* tick, not eagerly here --
                # same two-phase deferral as the single-env version.

            self._obs = next_obs

        bootstrap_not_done_mask = np.where(self._prev_done, 0.0, 1.0).astype(np.float32)
        with torch.no_grad():
            obs_t = _to_tensor_batch(self._obs, self.device)
            mask_t = torch.as_tensor(
                bootstrap_not_done_mask, dtype=torch.float32, device=self.device
            )
            extra_kwargs = {}
            if use_obstacle_encoder:
                extra_kwargs["ray_features"] = obs_t["ray_features"]
            _, last_value_t, _ = self.policy.forward(
                obs_t["robot"],
                obs_t["neighbors"],
                obs_t["neighbor_mask"],
                obs_t["neighbor_history"],
                obs_t["neighbor_history_mask"],
                self._hidden_state,
                mask_t,
                **extra_kwargs,
            )

        self.buffer.compute_returns_and_advantages(
            last_value=last_value_t.squeeze(-1).cpu().numpy().astype(np.float32),
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        return {
            "episodes_completed": len(episode_rewards),
            "mean_episode_reward": (
                float(np.mean(episode_rewards)) if episode_rewards else float("nan")
            ),
        }

    # -- PPO update -------------------------------------------------------------

    def _recompute_sequence(self) -> tuple[Tensor, Tensor, Tensor]:
        obs = self.buffer.observations()
        not_done_masks = self.buffer.not_done_masks()
        actions = self.buffer.actions()

        T = len(self.buffer)
        hidden = self.buffer.initial_hidden_state
        assert hidden is not None

        use_obstacle_encoder = self.policy.config.use_obstacle_encoder
        log_probs: list[Tensor] = []
        values: list[Tensor] = []
        entropies: list[Tensor] = []

        for t in range(T):
            extra_kwargs = {}
            if use_obstacle_encoder:
                extra_kwargs["ray_features"] = obs["ray_features"][t]
            distribution, value, hidden = self.policy.forward(
                obs["robot"][t],
                obs["neighbors"][t],
                obs["neighbor_mask"][t],
                obs["neighbor_history"][t],
                obs["neighbor_history_mask"][t],
                hidden,
                not_done_masks[t],
                **extra_kwargs,
            )
            log_probs.append(distribution.log_prob(actions[t]))
            values.append(value.squeeze(-1))
            entropies.append(distribution.entropy())

        return torch.stack(log_probs), torch.stack(values), torch.stack(entropies)

    def update(self) -> dict[str, float]:
        """Run config.n_epochs clipped-surrogate PPO passes over the
        current buffer.

        Returns:
            Diagnostics averaged across epochs: policy loss, value loss,
            mean entropy, approximate KL divergence from the collecting
            policy, and the fraction of steps whose probability ratio
            was clipped -- all averaged over every (t, env) entry in
            the buffer, not just per-env.
        """
        self.policy.train()

        advantages = self.buffer.advantages
        returns = self.buffer.returns
        old_log_probs = self.buffer.old_log_probs()
        old_values = self.buffer.old_values()

        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }

        for _ in range(self.config.n_epochs):
            new_log_probs, new_values, entropies = self._recompute_sequence()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(
                    ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range
                )
                * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            if self.config.clip_range_vf is not None:
                values_clipped = old_values + torch.clamp(
                    new_values - old_values,
                    -self.config.clip_range_vf,
                    self.config.clip_range_vf,
                )
                value_loss = torch.max(
                    (new_values - returns) ** 2, (values_clipped - returns) ** 2
                ).mean()
            else:
                value_loss = ((new_values - returns) ** 2).mean()

            entropy_loss = entropies.mean()
            loss = (
                policy_loss
                + self.config.vf_coef * value_loss
                - self.config.ent_coef * entropy_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()

            with torch.no_grad():
                approx_kl = (old_log_probs - new_log_probs).mean().item()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > self.config.clip_range).float().mean()
                ).item()

            totals["policy_loss"] += policy_loss.item()
            totals["value_loss"] += value_loss.item()
            totals["entropy"] += entropy_loss.item()
            totals["approx_kl"] += approx_kl
            totals["clip_fraction"] += clip_fraction

        n = self.config.n_epochs
        stats = {k: v / n for k, v in totals.items()}
        self.total_updates += 1
        return stats

    # -- top-level loop + checkpointing ----------------------------------------

    def train(
        self,
        total_timesteps: int,
        log_every: int = 1,
        checkpoint_every: int | None = None,
        checkpoint_dir: str | None = None,
    ) -> None:
        """Alternate collect/update until total_timesteps ticks are collected."""
        while self.total_steps < total_timesteps:
            rollout_stats = self.collect_rollout()
            update_stats = self.update()
            if self.metrics_logger is not None:
                self.metrics_logger.log(
                    self.total_steps, {**rollout_stats, **update_stats}
                )

            if self.total_updates % log_every == 0:
                print(
                    f"update={self.total_updates} steps={self.total_steps} "
                    f"episodes={rollout_stats['episodes_completed']} "
                    f"mean_ep_reward={rollout_stats['mean_episode_reward']:.2f} "
                    f"policy_loss={update_stats['policy_loss']:.4f} "
                    f"value_loss={update_stats['value_loss']:.4f} "
                    f"entropy={update_stats['entropy']:.4f} "
                    f"approx_kl={update_stats['approx_kl']:.4f} "
                    f"clip_frac={update_stats['clip_fraction']:.3f}"
                )

            if (
                checkpoint_every
                and checkpoint_dir
                and self.total_updates % checkpoint_every == 0
            ):
                self.save_checkpoint(
                    f"{checkpoint_dir}/crowdnav_pp_step{self.total_steps}.pt"
                )

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "total_updates": self.total_updates,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint["total_steps"]
        self.total_updates = checkpoint["total_updates"]
