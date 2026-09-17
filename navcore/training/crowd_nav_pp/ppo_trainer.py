# navcore/training/crowd_nav_pp/ppo_trainer.py
"""CrowdNavPPTrainer: PPO training loop for CrowdNavPPPolicy against CrowdSimEnv.

See rollout_buffer.py's module docstring for the Slice 1 scope decision
(single environment, whole-trajectory recompute per PPO epoch, no
chunked BPTT). This module owns everything else PPO needs on top of
that: action-mode wiring (must be ActionMode.VELOCITY -- CrowdNav++
predicts velocity directly; see CrowdSimEnv's own module docstring for
why WAYPOINT mode wouldn't exercise the policy's collision-avoidance
behavior at all), rollout collection, the clipped-surrogate update, and
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

from dataclasses import dataclass

import numpy as np
import torch
import os
from torch import Tensor

from navcore.gym_wrapper.crowd_sim_env import ActionMode, CrowdSimEnv
from navcore.policies.crowdnav_pp.policy import CrowdNavPPPolicy
from navcore.training.crowd_nav_pp.rollout_buffer import RecurrentRolloutBuffer


@dataclass(slots=True)
class PPOConfig:
    """PPO hyperparameters.

    Attributes:
        n_steps: Ticks collected per rollout, before each update phase.
        n_epochs: Full-trajectory recompute passes per rollout (see
            module docstring -- there is no time-axis minibatching here,
            only repeated whole-sequence passes).
        gamma: Discount factor.
        gae_lambda: GAE's bias/variance trade-off parameter.
        clip_range: PPO's surrogate-objective clip epsilon.
        clip_range_vf: Optional value-function clip epsilon. ``None``
            disables value clipping (plain MSE against returns).
        ent_coef: Entropy-bonus weight, encouraging exploration.
        vf_coef: Value-loss weight in the combined objective.
        max_grad_norm: Global gradient-norm clip applied before each
            optimizer step.
        learning_rate: Adam learning rate.
        normalize_advantage: Whether to standardize advantages
            (zero mean, unit std) before computing the surrogate loss.
        device: Torch device string ("cpu" or "cuda").
    """

    n_steps: int = 512
    n_epochs: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float | None = None
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    normalize_advantage: bool = True
    device: str = "cpu"


def _to_batch(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, Tensor]:
    """Add a leading nenv=1 axis to one step's unbatched observation dict."""
    return {
        k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
        for k, v in obs.items()
    }


class CrowdNavPPTrainer:
    """Drives PPO training of a ``CrowdNavPPPolicy`` against one ``CrowdSimEnv``.

    Attributes:
        env: The training environment. Must be configured with
            ``ActionMode.VELOCITY`` -- see module docstring.
        policy: The recurrent actor-critic being trained.
        config: PPO hyperparameters.
    """

    def __init__(
        self,
        env: CrowdSimEnv,
        policy: CrowdNavPPPolicy,
        config: PPOConfig | None = None,
    ) -> None:
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
        self.policy = policy
        self.config = config or PPOConfig()
        self.device = torch.device(self.config.device)

        self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate
        )

        self.buffer = RecurrentRolloutBuffer(device=self.device)

        self._obs: dict[str, np.ndarray] | None = None
        self._hidden_state: Tensor = self.policy.initial_hidden_state(
            nenv=1, device=self.device
        )
        # Forces not_done_mask=0.0 (hidden-state reset) on the very
        # first tick ever collected, same as a real episode boundary.
        self._prev_done = True

        self.total_steps = 0
        self.total_updates = 0

    # -- rollout collection ---------------------------------------------------

    def collect_rollout(self) -> dict[str, float]:
        """Collect ``config.n_steps`` ticks, then compute GAE targets.

        Resumes from wherever the previous rollout (or, on the first
        call, a fresh ``env.reset()``) left off -- the recurrent hidden
        state and in-progress episode both carry across rollout
        boundaries; only PPO's data window resets each call.

        Returns:
            A small dict of collection-time diagnostics (episodes
            completed this rollout, their mean total reward).
        """
        self.policy.eval()
        self.buffer.start(self._hidden_state)

        if self._obs is None:
            self._obs, _ = self.env.reset()

        episode_rewards: list[float] = []
        episode_reward = 0.0

        for _ in range(self.config.n_steps):
            not_done_mask = 0.0 if self._prev_done else 1.0
            obs_t = _to_batch(self._obs, self.device)
            not_done_mask_t = torch.tensor(
                [not_done_mask], dtype=torch.float32, device=self.device
            )

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
                )

            action_np = action_t.squeeze(0).cpu().numpy().astype(np.float32)
            next_obs, reward, terminated, truncated, _info = self.env.step(action_np)
            done = terminated or truncated

            self.buffer.add(
                obs=self._obs,
                not_done_mask=not_done_mask,
                action=action_np,
                log_prob=float(log_prob_t.item()),
                value=float(value_t.item()),
                reward=float(reward),
                done=done,
            )

            episode_reward += float(reward)
            self.total_steps += 1
            self._hidden_state = new_hidden
            self._prev_done = done

            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                next_obs, _ = self.env.reset()
                # self._hidden_state is intentionally left as-is here --
                # it gets zeroed inside policy.forward via not_done_mask
                # on the *next* tick, not eagerly here. Two-phase
                # ordering: this loop only ever reads/writes its own
                # local state, never reaches into the policy's internals.

            self._obs = next_obs

        bootstrap_not_done_mask = 0.0 if self._prev_done else 1.0
        with torch.no_grad():
            obs_t = _to_batch(self._obs, self.device)
            mask_t = torch.tensor(
                [bootstrap_not_done_mask], dtype=torch.float32, device=self.device
            )
            _, last_value_t, _ = self.policy.forward(
                obs_t["robot"],
                obs_t["neighbors"],
                obs_t["neighbor_mask"],
                obs_t["neighbor_history"],
                obs_t["neighbor_history_mask"],
                self._hidden_state,
                mask_t,
            )

        self.buffer.compute_returns_and_advantages(
            last_value=float(last_value_t.item()),
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
        """Re-run the policy sequentially over the whole buffered rollout.

        Starts from the buffer's stored (detached) seed hidden state and
        threads each step's stored ``not_done_mask`` through, so hidden-
        state reset points exactly match what was seen at collection
        time -- only the network's *weights* differ between this pass
        and the original rollout. Gradients flow through every step of
        this loop; see module docstring for why this is a whole-sequence
        pass rather than a chunked one.

        Returns:
            ``(log_probs, values, entropies)``, each shaped ``[T]``.
        """
        obs = self.buffer.observations()
        not_done_masks = self.buffer.not_done_masks()
        actions = self.buffer.actions()

        T = len(self.buffer)
        hidden = self.buffer.initial_hidden_state
        assert hidden is not None

        log_probs: list[Tensor] = []
        values: list[Tensor] = []
        entropies: list[Tensor] = []

        for t in range(T):
            distribution, value, hidden = self.policy.forward(
                obs["robot"][t : t + 1],
                obs["neighbors"][t : t + 1],
                obs["neighbor_mask"][t : t + 1],
                obs["neighbor_history"][t : t + 1],
                obs["neighbor_history_mask"][t : t + 1],
                hidden,
                not_done_masks[t : t + 1],
            )
            log_probs.append(distribution.log_prob(actions[t : t + 1]).squeeze(0))
            values.append(value.squeeze(0).squeeze(-1))
            entropies.append(distribution.entropy().squeeze(0))

        return torch.stack(log_probs), torch.stack(values), torch.stack(entropies)

    def update(self) -> dict[str, float]:
        """Run ``config.n_epochs`` clipped-surrogate PPO passes over the
        current buffer.

        Returns:
            Diagnostics averaged across epochs: policy loss, value loss,
            mean entropy, approximate KL divergence from the collecting
            policy, and the fraction of steps whose probability ratio
            was clipped.
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
        """Alternate collect/update until ``total_timesteps`` ticks are collected."""
        while self.total_steps < total_timesteps:
            rollout_stats = self.collect_rollout()
            update_stats = self.update()

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
