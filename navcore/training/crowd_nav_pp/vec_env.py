# navcore/training/crowd_nav_pp/vec_env.py
"""VecCrowdSimEnv: synchronous vectorized wrapper around CrowdSimEnv.

Owns n_envs independent CrowdSimEnv instances (each with its own Task, since
GoalReachingTask carries per-episode state that must not leak across envs -- see
GoalReachingTask.reset()). Stepped sequentially in a Python loop (no
multiprocessing) -- navcore's simulation step is cheap relative to the policy
forward pass, so this keeps the implementation simple and avoids pickling
issues with rvo2's C extension across process boundaries.

Auto-resets on done, matching the standard vec-env convention: a done env's
returned observation is already the first observation of the next episode.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from navcore.gym_wrapper.crowd_sim_env import CrowdSimEnv


class VecCrowdSimEnv:
    """Steps n_envs CrowdSimEnv instances together, stacking obs/reward/done.

    Attributes:
        envs: The owned CrowdSimEnv instances, one per parallel slot.
        n_envs: len(envs).
        observation_space: envs[0]'s space (assumed identical across envs --
            all built from the same CrowdSimEnvConfig).
        action_space: Same assumption.
        config: envs[0]'s CrowdSimEnvConfig, for callers that need it (e.g.
            CrowdNavPPTrainer's action-mode validation).
    """

    def __init__(self, env_fns: list[Callable[[], CrowdSimEnv]]) -> None:
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.config = self.envs[0].config

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        obs_list = []
        for i, env in enumerate(self.envs):
            env_seed = None if seed is None else seed + i
            obs, _ = env.reset(seed=env_seed)
            obs_list.append(obs)
        return self._stack(obs_list)

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict]]:
        """Step every env once.

        Args:
            actions: [n_envs, action_dim].

        Returns:
            (obs, rewards, dones, infos). ``obs`` is already post-reset for
            any env that finished this tick. ``dones`` is terminated OR
            truncated, matching CrowdSimEnv.step's own combination.
        """
        obs_list = []
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        infos: list[dict] = []

        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            done = terminated or truncated
            if done:
                info = dict(info)
                info["terminal_observation"] = obs
                obs, _ = env.reset()
            obs_list.append(obs)
            rewards[i] = reward
            dones[i] = done
            infos.append(info)

        return self._stack(obs_list), rewards, dones, infos

    @staticmethod
    def _stack(obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        keys = obs_list[0].keys()
        return {k: np.stack([o[k] for o in obs_list]) for k in keys}
