# navcore/training/rollout_buffer.py
"""RecurrentRolloutBuffer: single-environment, GAE-based rollout storage
for CrowdNavPPPolicy's PPO training loop.

Scope note (Slice 1):
    Buffers exactly one trajectory from one CrowdSimEnv instance,
    spanning n_steps ticks (which may cross zero or more episode
    boundaries -- not_done_mask/dones record exactly where). PPO's
    clipped-surrogate update recomputes the policy's forward pass
    sequentially over the *entire* buffered trajectory every epoch (see
    ppo_trainer.py) rather than splitting it into independently-seeded
    truncated-BPTT chunks across parallel environments -- this matches
    RecurrentNodeUpdate's own documented scope (single-timestep GRU
    update only; the batched multi-timestep path is explicitly deferred
    there until a real training loop exists to design it against). This
    *is* that training loop, and it deliberately stays on the
    single-timestep path rather than silently inventing the deferred
    batched path here. Multi-env vectorization + chunked BPTT is a
    natural follow-up once this path is validated -- flagged, not
    implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

_OBS_KEYS = (
    "robot",
    "neighbors",
    "neighbor_mask",
    "neighbor_history",
    "neighbor_history_mask",
)


@dataclass(slots=True)
class RecurrentRolloutBuffer:
    """Accumulates one rollout's transitions, then computes GAE targets.

    Attributes:
        device: Where returned tensors live.
        initial_hidden_state: The GRU hidden state that existed
            immediately before this buffer's first stored step.
            ``ppo_trainer.py``'s recompute pass starts from this exact
            (detached) value every epoch, so a whole rollout's worth of
            hidden-state history never needs to be stored -- only the
            seed, plus each step's ``not_done_mask`` to reproduce the
            same reset points during recompute.
    """

    device: torch.device
    initial_hidden_state: Tensor | None = None

    _obs: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {k: [] for k in _OBS_KEYS}
    )
    _actions: list[np.ndarray] = field(default_factory=list)
    _log_probs: list[float] = field(default_factory=list)
    _values: list[float] = field(default_factory=list)
    _rewards: list[float] = field(default_factory=list)
    _dones: list[bool] = field(default_factory=list)
    _not_done_masks: list[float] = field(default_factory=list)

    _advantages: Tensor | None = None
    _returns: Tensor | None = None

    def __len__(self) -> int:
        return len(self._rewards)

    def start(self, initial_hidden_state: Tensor) -> None:
        """Reset buffer contents and record this rollout's seed hidden state."""
        for k in _OBS_KEYS:
            self._obs[k] = []
        self._actions = []
        self._log_probs = []
        self._values = []
        self._rewards = []
        self._dones = []
        self._not_done_masks = []
        self._advantages = None
        self._returns = None
        self.initial_hidden_state = initial_hidden_state.detach().clone()

    def add(
        self,
        obs: dict[str, np.ndarray],
        not_done_mask: float,
        action: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        """Record one collected transition.

        Args:
            obs: This step's ``ObservationEncoder.encode()`` output --
                unbatched arrays, exactly as produced (no leading nenv
                axis; that's added on demand when tensors are pulled
                back out for a forward pass).
            not_done_mask: The mask actually fed to the policy for this
                step (``1.0 - previous step's done``). Stored, not
                recomputed, so recompute passes during PPO update
                reproduce identical hidden-state reset points.
            action: The raw (pre-env-clip) sampled action.
            log_prob: The policy's log-prob of ``action`` at collection
                time -- PPO's ratio denominator.
            value: The policy's value estimate at collection time.
            reward: This step's scalar reward.
            done: Whether the episode ended (terminated or truncated)
                on this step.
        """
        for k in _OBS_KEYS:
            self._obs[k].append(obs[k])
        self._not_done_masks.append(not_done_mask)
        self._actions.append(action)
        self._log_probs.append(log_prob)
        self._values.append(value)
        self._rewards.append(reward)
        self._dones.append(done)

    # -- tensor views over stored steps --------------------------------------

    def observations(self) -> dict[str, Tensor]:
        """Return every stored step's observation, stacked as ``[T, ...]``."""
        return {
            k: torch.as_tensor(np.stack(v), dtype=torch.float32, device=self.device)
            for k, v in self._obs.items()
        }

    def actions(self) -> Tensor:
        return torch.as_tensor(
            np.stack(self._actions), dtype=torch.float32, device=self.device
        )

    def old_log_probs(self) -> Tensor:
        return torch.as_tensor(self._log_probs, dtype=torch.float32, device=self.device)

    def old_values(self) -> Tensor:
        return torch.as_tensor(self._values, dtype=torch.float32, device=self.device)

    def not_done_masks(self) -> Tensor:
        return torch.as_tensor(
            self._not_done_masks, dtype=torch.float32, device=self.device
        )

    # -- GAE ------------------------------------------------------------------

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> None:
        """Compute per-step advantage/return targets via GAE(lambda).

        Must be called once, right after the rollout is fully collected.
        The result is fixed for every PPO epoch over this rollout -- only
        the policy's *recomputed* log-probs/values change across epochs,
        never these targets (standard PPO: advantages are a property of
        the data-collecting policy, not the updating one).

        Args:
            last_value: Bootstrap value estimate for the state one tick
                past the buffer's last stored step (see
                ``ppo_trainer.CrowdNavPPTrainer.collect_rollout``).
            gamma: Discount factor.
            gae_lambda: GAE's bias/variance trade-off parameter.
        """
        T = len(self)
        rewards = self._rewards
        values = self._values + [last_value]
        dones = self._dones

        advantages = [0.0] * T
        gae = 0.0
        for t in reversed(range(T)):
            next_non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        advantages_t = torch.as_tensor(
            advantages, dtype=torch.float32, device=self.device
        )
        returns_t = advantages_t + torch.as_tensor(
            values[:T], dtype=torch.float32, device=self.device
        )
        self._advantages = advantages_t
        self._returns = returns_t

    @property
    def advantages(self) -> Tensor:
        assert self._advantages is not None, (
            "call compute_returns_and_advantages() before reading advantages."
        )
        return self._advantages

    @property
    def returns(self) -> Tensor:
        assert self._returns is not None, (
            "call compute_returns_and_advantages() before reading returns."
        )
        return self._returns
