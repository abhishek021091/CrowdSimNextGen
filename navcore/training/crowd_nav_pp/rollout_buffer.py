# navcore/training/crowd_nav_pp/rollout_buffer.py
"""RecurrentRolloutBuffer: multi-environment, GAE-based rollout storage
for CrowdNavPPPolicy's PPO training loop.

Scope note (Slice 2 -- supersedes Slice 1's single-env note):
    Buffers n_envs trajectories collected in lockstep from a VecCrowdSimEnv,
    spanning n_steps ticks each. Every per-step field (obs, actions,
    log_probs, values, rewards, dones, not_done_masks) carries a leading
    n_envs axis; GAE is computed independently per env-column (the
    recursion in compute_returns_and_advantages is elementwise over numpy
    arrays, so it naturally decorrelates envs without special-casing).
    PPO's clipped-surrogate update still recomputes the policy's forward
    pass sequentially over the whole buffered T-step window every epoch
    (see ppo_trainer.py) -- this is unchanged from Slice 1 and still
    matches RecurrentNodeUpdate's documented single-timestep-only scope;
    only the *batch* dimension at each timestep grew from 1 to n_envs.
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
    """Accumulates one rollout's transitions across n_envs, then computes GAE targets.

    Attributes:
        device: Where returned tensors live.
        n_envs: Number of parallel environments this buffer's fields are
            batched over. Set by start(), from initial_hidden_state's
            leading dim.
        initial_hidden_state: The GRU hidden state ([n_envs, hidden_size])
            that existed immediately before this buffer's first stored
            step. ppo_trainer.py's recompute pass starts from this exact
            (detached) value every epoch, so a whole rollout's worth of
            hidden-state history never needs to be stored -- only the
            seed, plus each step's not_done_mask to reproduce the same
            reset points during recompute.
    """

    device: torch.device
    n_envs: int = 1
    initial_hidden_state: Tensor | None = None

    _obs: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {k: [] for k in _OBS_KEYS}
    )
    _actions: list[np.ndarray] = field(default_factory=list)
    _log_probs: list[np.ndarray] = field(default_factory=list)
    _values: list[np.ndarray] = field(default_factory=list)
    _rewards: list[np.ndarray] = field(default_factory=list)
    _dones: list[np.ndarray] = field(default_factory=list)
    _not_done_masks: list[np.ndarray] = field(default_factory=list)

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
        self.n_envs = initial_hidden_state.shape[0]

    def add(
        self,
        obs: dict[str, np.ndarray],
        not_done_mask: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """Record one collected timestep, across all n_envs.

        Args:
            obs: This step's VecCrowdSimEnv-batched observation -- each
                array already carries a leading [n_envs] axis.
            not_done_mask: [n_envs] float32 -- the mask actually fed to
                the policy for this step (1.0 - previous step's done, per
                env). Stored, not recomputed, so recompute passes during
                PPO update reproduce identical hidden-state reset points.
            action: [n_envs, action_dim] raw (pre-env-clip) sampled action.
            log_prob: [n_envs] policy's log-prob of action at collection
                time -- PPO's ratio denominator.
            value: [n_envs] policy's value estimate at collection time.
            reward: [n_envs] this step's scalar reward, per env.
            done: [n_envs] bool -- whether that env's episode ended
                (terminated or truncated) on this step.
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
        """Return every stored step's observation, stacked as [T, n_envs, ...]."""
        return {
            k: torch.as_tensor(np.stack(v), dtype=torch.float32, device=self.device)
            for k, v in self._obs.items()
        }

    def actions(self) -> Tensor:
        return torch.as_tensor(
            np.stack(self._actions), dtype=torch.float32, device=self.device
        )

    def old_log_probs(self) -> Tensor:
        return torch.as_tensor(
            np.stack(self._log_probs), dtype=torch.float32, device=self.device
        )

    def old_values(self) -> Tensor:
        return torch.as_tensor(
            np.stack(self._values), dtype=torch.float32, device=self.device
        )

    def not_done_masks(self) -> Tensor:
        return torch.as_tensor(
            np.stack(self._not_done_masks), dtype=torch.float32, device=self.device
        )

    # -- GAE ------------------------------------------------------------------

    def compute_returns_and_advantages(
        self, last_value: np.ndarray, gamma: float, gae_lambda: float
    ) -> None:
        """Compute per-step, per-env advantage/return targets via GAE(lambda).

        Must be called once, right after the rollout is fully collected.
        The recursion is plain numpy arithmetic over [n_envs]-shaped rows,
        so each env's GAE accumulator resets independently at that env's
        own episode boundaries (via that column's `dones`) without any
        cross-env leakage -- envs are never mixed within the recursion,
        only stacked into the same array for vectorized computation.

        Args:
            last_value: [n_envs] bootstrap value estimate for the state
                one tick past the buffer's last stored step (see
                ppo_trainer.CrowdNavPPTrainer.collect_rollout).
            gamma: Discount factor.
            gae_lambda: GAE's bias/variance trade-off parameter.
        """
        T = len(self)
        rewards = np.stack(self._rewards)  # [T, n_envs]
        values = np.concatenate(
            [np.stack(self._values), last_value[None, :]], axis=0
        )  # [T + 1, n_envs]
        dones = np.stack(self._dones).astype(np.float32)  # [T, n_envs]

        advantages = np.zeros((T, self.n_envs), dtype=np.float32)
        gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(T)):
            next_non_terminal = 1.0 - dones[t]
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
