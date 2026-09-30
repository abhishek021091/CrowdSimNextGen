"""OriginalPPOTrainer: recurrent PPO for ``navcore.policies.original``.

Differences from ``training/crowd_nav_pp/crowd_nav_pp_trainer.py`` (which is
left untouched):

* Obstacle handling is driven solely by ``policy.config.uses_ray_features``
  (no comparison against a specific ``ObstacleMode`` -- that comparison was
  the source of the two-enum bug, and skipped rays in ENCODER mode during
  the PPO recompute pass).
* Actions are sampled and scored in fp32 from the distribution's mean/std,
  so log-probs and ratios stay fp32 even under bf16 autocast.
* Env-axis minibatching (``num_minibatch``), LR schedules, optional bf16
  AMP, target-KL early stop, TensorBoard/JSONL logging, atomic checkpoints.
* Checkpoints carry ``policy_config`` so evaluation can rebuild the policy.

Sequence recomputation: each PPO epoch re-runs the policy over the whole
stored T-step window from the stored initial hidden state, applying the
stored ``not_done`` masks (identical reset points), with full BPTT.
Time-limit truncation is treated as termination (as in the existing buffer).

AMP note: bf16 only. fp16 cannot work here because the official
``EdgeAttention_M`` does ``masked_fill(mask == 0, -1e9)`` and -1e9 overflows fp16.
"""

from __future__ import annotations

import contextlib
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Normal

from navcore.analysis.metrics_logger import TrainingMetricsLogger
from navcore.training.crowd_nav_pp.rollout_buffer import RecurrentRolloutBuffer
from navcore.training.original.checkpoint import (
    PPO_FORMAT,
    CheckpointInfo,
    config_to_dict,
    describe_optimizer_state,
    load_policy_checkpoint,
)


@dataclass
class PPOConfig:
    n_steps: int = 128
    n_epochs: int = 4
    num_minibatch: int = 1  # splits the env axis; must be <= n_envs
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float | None = 0.2  # None disables value clipping
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float | None = 0.5
    learning_rate: float = 1e-4
    lr_schedule: str = "linear"  # "constant" | "linear" | "cosine"
    lr_min_factor: float = 0.0
    adam_eps: float = 1e-5
    normalize_advantage: bool = True
    target_kl: float | None = None
    amp: bool = False  # bf16 autocast, CUDA only
    device: str = "cpu"


def _gauss(mean: Tensor, std: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
    n = Normal(mean, std)
    return n.log_prob(actions).sum(-1), n.entropy().sum(-1)


def _explained_variance(y_pred: Tensor, y_true: Tensor) -> float:
    var_y = torch.var(y_true)
    if var_y.item() == 0.0:
        return float("nan")
    return float(1.0 - torch.var(y_true - y_pred) / var_y)

def _f(v: float | None, spec: str = ".3f") -> str:
    """Format a number, rendering None/NaN/inf as 'n/a'."""
    if v is None or not math.isfinite(v):
        return "n/a"
    return format(v, spec)


def _hms(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "n/a"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


class OriginalPPOTrainer:
    def __init__(
        self,
        env,
        policy,
        config=None,
        *,
        writer=None,
        metrics_path=None,
        render: bool = False,
        seed: int | None = None,
    ) -> None:

        mode = getattr(getattr(env, "config", None), "action_mode", None)
        if getattr(mode, "value", mode) != "velocity":
            raise ValueError("OriginalPPOTrainer requires ActionMode.VELOCITY.")
        self.env = env
        self.n_envs = env.n_envs
        self.policy = policy
        self.config = config or PPOConfig()
        if not 1 <= self.config.num_minibatch <= self.n_envs:
            raise ValueError(f"num_minibatch must be in [1, n_envs={self.n_envs}].")
        self.device = torch.device(self.config.device)
        self.amp = bool(self.config.amp)
        if self.amp and self.device.type != "cuda":
            warnings.warn("--amp needs CUDA; running in fp32.")
            self.amp = False
        self.policy.to(self.device)
        params = [p for p in policy.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            params, lr=self.config.learning_rate, eps=self.config.adam_eps
        )
        self.buffer = RecurrentRolloutBuffer(device=self.device)
        self.writer = writer
        self.metrics_logger = (
            TrainingMetricsLogger(Path(metrics_path)) if metrics_path else None
        )

        self._obs: dict[str, np.ndarray] | None = None
        self._hidden = policy.initial_hidden_state(nenv=self.n_envs, device=self.device)
        self._prev_done = np.ones(
            self.n_envs, dtype=bool
        )  # forces reset mask on first tick
        self.total_steps = 0
        self.total_updates = 0
        self.best_eval_score: tuple[float, float] | None = None
        self.render = render
        self.seed = seed

        if self._obs is None:
            self._obs = self.env.reset(seed=self.seed)

        if self.render:
            self.env.render()

    # -- helpers ---------------------------------------------------------------

    def _autocast(self):
        if self.amp:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _tensors(self, obs: dict[str, np.ndarray]) -> dict[str, Tensor]:
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in obs.items()
        }

    def _forward(self, obs: dict[str, Tensor], hidden: Tensor, not_done: Tensor):
        rays = obs["ray_features"] if self.policy.config.uses_ray_features else None
        with self._autocast():
            dist, value, new_hidden = self.policy.forward(
                obs["robot"],
                obs["neighbors"],
                obs["neighbor_mask"],
                obs["neighbor_history"],
                obs["neighbor_history_mask"],
                hidden,
                not_done,
                ray_features=rays,
            )
        return dist.mean.float(), dist.stddev.float(), value.float(), new_hidden.float()

    def lr_at(self, total_timesteps: int) -> float:
        c = self.config
        p = min(max(self.total_steps / max(total_timesteps, 1), 0.0), 1.0)
        if c.lr_schedule == "linear":
            f = 1.0 - p
        elif c.lr_schedule == "cosine":
            f = 0.5 * (1.0 + math.cos(math.pi * p))
        elif c.lr_schedule == "constant":
            f = 1.0
        else:
            raise ValueError(f"Unknown lr_schedule {c.lr_schedule!r}.")
        return c.learning_rate * (c.lr_min_factor + (1.0 - c.lr_min_factor) * f)

    # -- rollout -------------------------------------------------------------------

    def collect_rollout(self) -> dict[str, float]:
        self.policy.eval()
        self.buffer.start(self._hidden)
        if self._obs is None:
            self._obs = self.env.reset()

        ep_rewards: list[float] = []
        ep_lengths: list[int] = []
        outcomes = {"success": 0, "collision": 0, "out_of_bounds": 0, "timeout": 0}
        ep_r = np.zeros(self.n_envs, dtype=np.float32)
        ep_l = np.zeros(self.n_envs, dtype=np.int64)

        for _ in range(self.config.n_steps):
            not_done = np.where(self._prev_done, 0.0, 1.0).astype(np.float32)
            obs_t = self._tensors(self._obs)
            nd_t = torch.as_tensor(not_done, device=self.device)
            with torch.no_grad():
                mean, std, value, new_hidden = self._forward(obs_t, self._hidden, nd_t)
                normal = Normal(mean, std)
                action = normal.sample()
                logp = normal.log_prob(action).sum(-1)
            action_np = action.cpu().numpy().astype(np.float32)
            next_obs, reward, done, infos = self.env.step(action_np)

            self.buffer.add(
                obs=self._obs,
                not_done_mask=not_done,
                action=action_np,
                log_prob=logp.cpu().numpy().astype(np.float32),
                value=value.squeeze(-1).cpu().numpy().astype(np.float32),
                reward=np.asarray(reward, dtype=np.float32),
                done=np.asarray(done, dtype=bool),
            )
            ep_r += reward
            ep_l += 1
            self.total_steps += self.n_envs
            self._hidden = new_hidden
            self._prev_done = np.asarray(done, dtype=bool)
            for i in np.nonzero(done)[0]:
                ep_rewards.append(float(ep_r[i]))
                ep_lengths.append(int(ep_l[i]))
                outcomes[self._classify(infos[i])] += 1
                ep_r[i] = 0.0
                ep_l[i] = 0
            self._obs = next_obs

        bootstrap_mask = np.where(self._prev_done, 0.0, 1.0).astype(np.float32)
        with torch.no_grad():
            _, _, last_value, _ = self._forward(
                self._tensors(self._obs),
                self._hidden,
                torch.as_tensor(bootstrap_mask, device=self.device),
            )
        self.buffer.compute_returns_and_advantages(
            last_value=last_value.squeeze(-1).cpu().numpy().astype(np.float32),
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        n = len(ep_rewards)
        nan = float("nan")
        stats = {
            "episodes_completed": float(n),
            "mean_episode_reward": float(np.mean(ep_rewards)) if n else nan,
            "std_episode_reward": float(np.std(ep_rewards)) if n else nan,
            "min_episode_reward": float(np.min(ep_rewards)) if n else nan,
            "max_episode_reward": float(np.max(ep_rewards)) if n else nan,
            "mean_episode_length": float(np.mean(ep_lengths)) if n else nan,
        }
        for k, c in outcomes.items():
            stats[f"{k}_rate"] = c / n if n else nan
        return stats

    @staticmethod
    def _classify(info: dict) -> str:
        if info.get("collision"):
            return "collision"
        if info.get("out_of_bounds"):
            return "out_of_bounds"
        if info.get("terminated"):
            return "success"
        return "timeout"

    # -- update ----------------------------------------------------------------------

    def _recompute(self, obs, not_done, actions, hidden0):
        hidden = hidden0
        lps, vals, ents = [], [], []
        for t in range(actions.shape[0]):
            mean, std, value, hidden = self._forward(
                {k: v[t] for k, v in obs.items()}, hidden, not_done[t]
            )
            lp, ent = _gauss(mean, std, actions[t])
            lps.append(lp)
            vals.append(value.squeeze(-1))
            ents.append(ent)
        return torch.stack(lps), torch.stack(vals), torch.stack(ents)

    def update(self) -> dict[str, float]:
        c = self.config
        self.policy.train()
        adv = self.buffer.advantages
        returns = self.buffer.returns
        old_lp = self.buffer.old_log_probs()
        old_v = self.buffer.old_values()
        obs_all = self.buffer.observations()
        nd_all = self.buffer.not_done_masks()
        act_all = self.buffer.actions()
        h0_all = self.buffer.initial_hidden_state
        assert h0_all is not None

        ev = _explained_variance(old_v, returns)
        if c.normalize_advantage:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        tot = dict.fromkeys(
            (
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "grad_norm",
            ),
            0.0,
        )
        n_iters = 0
        stop = False
        for _ in range(c.n_epochs):
            perm = torch.randperm(self.n_envs, device=self.device)
            for idx in torch.tensor_split(perm, c.num_minibatch):
                obs = {k: v[:, idx] for k, v in obs_all.items()}
                new_lp, new_v, ent = self._recompute(
                    obs, nd_all[:, idx], act_all[:, idx], h0_all[idx].detach()
                )
                a, r, o_lp, o_v = (
                    adv[:, idx],
                    returns[:, idx],
                    old_lp[:, idx],
                    old_v[:, idx],
                )

                log_ratio = new_lp - o_lp
                ratio = log_ratio.exp()
                policy_loss = -torch.min(
                    ratio * a, ratio.clamp(1 - c.clip_range, 1 + c.clip_range) * a
                ).mean()
                if c.clip_range_vf is not None:
                    v_clip = o_v + (new_v - o_v).clamp(
                        -c.clip_range_vf, c.clip_range_vf
                    )
                    value_loss = torch.max((new_v - r) ** 2, (v_clip - r) ** 2).mean()
                else:
                    value_loss = ((new_v - r) ** 2).mean()
                entropy = ent.mean()
                loss = policy_loss + c.vf_coef * value_loss - c.ent_coef * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss (policy={policy_loss.item()}, value={value_loss.item()}, "
                        f"entropy={entropy.item()}). Lower --learning-rate or inspect observations."
                    )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy.parameters() if p.requires_grad],
                    c.max_grad_norm if c.max_grad_norm is not None else float("inf"),
                )
                self.optimizer.step()

                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean().item()  # k3 estimator, >= 0
                    clipf = ((ratio - 1).abs() > c.clip_range).float().mean().item()
                tot["policy_loss"] += policy_loss.item()
                tot["value_loss"] += value_loss.item()
                tot["entropy"] += entropy.item()
                tot["approx_kl"] += kl
                tot["clip_fraction"] += clipf
                tot["grad_norm"] += float(gn)
                n_iters += 1
                if c.target_kl is not None and kl > 1.5 * c.target_kl:
                    stop = True
                    break
            if stop:
                break

        stats = {k: v / max(n_iters, 1) for k, v in tot.items()}
        stats["explained_variance"] = ev
        stats["action_std_mean"] = float(
            self.policy.action_head.log_std.detach().exp().mean()
        )
        stats["optimizer_iterations"] = float(n_iters)
        self.total_updates += 1
        return stats

    # -- loop --------------------------------------------------------------------------

    def _log(self, prefix: str, stats: dict[str, float]) -> None:
        if self.writer is None:
            return
        for k, v in stats.items():
            if isinstance(v, (int, float)) and math.isfinite(v):
                self.writer.add_scalar(f"{prefix}/{k}", v, self.total_steps)

    def train(
        self,
        total_timesteps: int,
        *,
        log_every: int = 1,
        checkpoint_every: int | None = None,
        checkpoint_dir: str | Path | None = None,
        eval_fn: Callable[[], dict[str, float]] | None = None,
        eval_every: int = 0,
    ) -> None:
        t0 = time.time()
        s0 = self.total_steps
        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if ckpt_dir:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        while self.total_steps < total_timesteps:
            lr = self.lr_at(total_timesteps)
            for g in self.optimizer.param_groups:
                g["lr"] = lr
            rs = self.collect_rollout()
            us = self.update()
            us["lr"] = lr
            self._log("rollout", rs)
            self._log("train", us)
            if self.metrics_logger:
                self.metrics_logger.log(self.total_steps, {**rs, **us})
            if self.total_updates % log_every == 0:
                print(
                    self._format_update(
                        rs,
                        us,
                        total_timesteps,
                        elapsed=time.time() - t0,
                        steps_this_run=self.total_steps - s0,
                    ),
                    flush=True,
                )
            if (
                ckpt_dir
                and checkpoint_every
                and self.total_updates % checkpoint_every == 0
            ):
                self.save_checkpoint(
                    ckpt_dir / f"original_ppo_step{self.total_steps}.pt"
                )
                self.save_checkpoint(ckpt_dir / "latest.pt")
            if eval_fn and eval_every and self.total_updates % eval_every == 0:
                ev = eval_fn()
                self._log("eval", ev)
                print(
                    "  eval: "
                    + " ".join(
                        f"{k}={v:.3f}" for k, v in ev.items() if isinstance(v, float)
                    ),
                    flush=True,
                )
                score = (ev.get("success_rate", 0.0), ev.get("mean_reward", 0.0))
                if ckpt_dir and (
                    self.best_eval_score is None or score > self.best_eval_score
                ):
                    self.best_eval_score = score
                    self.save_checkpoint(ckpt_dir / "best.pt")
        if ckpt_dir:
            self.save_checkpoint(ckpt_dir / "latest.pt")
            self.save_checkpoint(ckpt_dir / f"original_ppo_step{self.total_steps}.pt")
        if self.writer is not None:
            self.writer.flush()

    # -- checkpoints --------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "format": PPO_FORMAT,
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "total_updates": self.total_updates,
                "policy_config": config_to_dict(self.policy.config),
                "ppo_config": {k: v for k, v in vars(self.config).items()},
            },
            tmp,
        )
        os.replace(tmp, path)

    def load_training_state(self, path: str | Path) -> CheckpointInfo:
        """Resume: weights (+ optimizer/counters if it is a PPO checkpoint).

        Any other format is accepted as weights-only with a warning. Envs are
        reset and hidden states zeroed (they are not checkpointed).
        """
        info = load_policy_checkpoint(self.policy, path)  # prints full report
        optimizer_restored = False
        if info.format == "ppo":
            ck = torch.load(path, map_location="cpu", weights_only=True)
            if "optimizer_state_dict" in ck:
                try:
                    self.optimizer.load_state_dict(ck["optimizer_state_dict"])
                    optimizer_restored = True
                except ValueError as e:
                    warnings.warn(
                        f"Optimizer state not restored ({e}); fresh Adam state."
                    )
            self.total_steps = int(ck.get("total_steps", 0))
            self.total_updates = int(ck.get("total_updates", 0))
        else:
            warnings.warn(
                f"{path} is a {info.format!r} checkpoint: weights loaded, counters/optimizer fresh."
            )
        print(describe_optimizer_state(info, optimizer_restored))
        self._obs = None
        self._prev_done = np.ones(self.n_envs, dtype=bool)
        self._hidden = self.policy.initial_hidden_state(
            nenv=self.n_envs, device=self.device
        )
        return info

    def _format_update(
        self,
        rs: dict[str, float],
        us: dict[str, float],
        total_timesteps: int,
        elapsed: float,
        steps_this_run: int,
    ) -> str:
        sps = steps_this_run / max(elapsed, 1e-9)
        remaining = max(total_timesteps - self.total_steps, 0)
        eta = remaining / sps if sps > 0 else float("nan")
        pct = 100.0 * self.total_steps / max(total_timesteps, 1)
        return "\n".join(
            [
                f"update={self.total_updates} "
                f"steps={self.total_steps:,}/{total_timesteps:,} ({pct:.1f}%) "
                f"elapsed={_hms(elapsed)} sps={sps:.0f} eta={_hms(eta)} "
                f"lr={us['lr']:.2e}",
                f"  episodes={int(rs['episodes_completed'])} "
                f"success={_f(rs['success_rate'], '.1%')} "
                f"collision={_f(rs['collision_rate'], '.1%')} "
                f"oob={_f(rs['out_of_bounds_rate'], '.1%')} "
                f"timeout={_f(rs['timeout_rate'], '.1%')}",
                f"  reward: mean={_f(rs['mean_episode_reward'], '.2f')} "
                f"std={_f(rs['std_episode_reward'], '.2f')} "
                f"min={_f(rs['min_episode_reward'], '.2f')} "
                f"max={_f(rs['max_episode_reward'], '.2f')} "
                f"| ep_len mean={_f(rs['mean_episode_length'], '.1f')}",
                f"  loss: policy={_f(us['policy_loss'], '.4f')} "
                f"value={_f(us['value_loss'], '.3f')} "
                f"entropy={_f(us['entropy'], '.3f')}",
                f"  ppo: approx_kl={_f(us['approx_kl'], '.4f')} "
                f"clip_frac={_f(us['clip_fraction'], '.3f')} "
                f"grad_norm={_f(us['grad_norm'], '.2f')} "
                f"expl_var={_f(us['explained_variance'], '.3f')} "
                f"action_std={_f(us['action_std_mean'], '.3f')} "
                f"opt_iters={int(us['optimizer_iterations'])}",
            ]
        )
