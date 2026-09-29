"""Shared helpers for the original-policy training/evaluation scripts."""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from navcore.policies.obstacle_mode import ObstacleMode
from navcore.policies.original.adapter import CrowdNavPPPolicy, CrowdNavPPPolicyConfig
from navcore.training.original.checkpoint import CheckpointInfo, config_from_dict


def set_global_seed(seed: int, deterministic_torch: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def pick_device(name: str | None) -> torch.device:
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available.")
    return dev


# -- policy ------------------------------------------------------------------------


def resolve_policy_config(
    *,
    obstacle_mode: str | None,
    max_neighbors: int | None,
    obstacle_max_range: float | None,
    checkpoint_info: CheckpointInfo | None = None,
    obstacle_hit_radius: float | None = None,
) -> CrowdNavPPPolicyConfig:
    """CLI value wins; else the checkpoint's weights/config; else defaults."""
    meta = checkpoint_info.policy_config if checkpoint_info else None
    base = (
        config_from_dict(CrowdNavPPPolicyConfig, meta)
        if meta
        else CrowdNavPPPolicyConfig()
    )

    if obstacle_mode not in (None, "auto"):
        mode = ObstacleMode(obstacle_mode)
    elif (
        checkpoint_info is not None
        and checkpoint_info.obstacle_mode is not ObstacleMode.NONE
    ):
        mode = checkpoint_info.obstacle_mode
    elif meta and isinstance(meta.get("obstacle_mode"), str):
        mode = ObstacleMode[meta["obstacle_mode"]]
    else:
        mode = ObstacleMode.NONE

    overrides: dict[str, Any] = {"obstacle_mode": mode, "use_obstacle_encoder": None}
    if max_neighbors is not None:
        overrides["max_neighbors"] = max_neighbors
    if obstacle_max_range is not None:
        overrides["obstacle_max_range"] = obstacle_max_range
    if obstacle_hit_radius is not None:
        overrides["obstacle_hit_radius"] = obstacle_hit_radius
    return dataclasses.replace(base, **overrides)


def build_policy(cfg: CrowdNavPPPolicyConfig, device: torch.device) -> CrowdNavPPPolicy:
    return CrowdNavPPPolicy(cfg).to(device)


# -- environment ---------------------------------------------------------------------


@dataclass(frozen=True)
class EnvSettings:
    max_neighbors: int = 10
    history_steps: int = 8
    max_episode_steps: int = 1500
    static_obstacles: bool = False
    obstacle_num_rays: int = 60
    obstacle_max_range: float = 5.0


def resolve_static_obstacles(arg: str, cfg: CrowdNavPPPolicyConfig) -> bool:
    """'auto' -> static tables only when the policy can actually see them."""
    return {"on": True, "off": False}.get(arg, cfg.uses_ray_features)


def make_env_class():
    """Lazy so importing this module doesn't require gymnasium/rvo2 at import."""
    from navcore.gym_wrapper.crowd_sim_env import CrowdSimEnv

    class SeededCrowdSimEnv(CrowdSimEnv):
        """CrowdSimEnv whose ``Step`` RNG is seeded.

        ``Step`` builds ``np.random.default_rng()`` unseeded, and uses it to
        randomly freeze pedestrians (1%/tick), which makes replays diverge.
        We reseed it after every reset from the env's own (gym-seeded)
        ``np_random`` stream -- no change to existing files needed.
        """

        def reset(self, *, seed=None, options=None):
            obs, info = super().reset(seed=seed, options=options)
            if self._step_driver is not None:
                self._step_driver.rand = np.random.default_rng(
                    int(self.np_random.integers(0, 2**31 - 1))
                )
            return obs, info

    return SeededCrowdSimEnv


def make_env(settings: EnvSettings, render_mode: str | None = None):
    from navcore.gym_wrapper.crowd_sim_env import ActionMode, CrowdSimEnvConfig
    from navcore.gym_wrapper.goal_reaching_task import GoalReachingTask

    cfg = CrowdSimEnvConfig(
        action_mode=ActionMode.VELOCITY,
        max_neighbors=settings.max_neighbors,
        history_steps=settings.history_steps,
        max_episode_steps=settings.max_episode_steps,
        include_static_obstacles=settings.static_obstacles,
        obstacle_num_rays=settings.obstacle_num_rays,
        obstacle_max_range=settings.obstacle_max_range,
    )
    return make_env_class()(GoalReachingTask(), cfg, render_mode=render_mode)


def make_vec_env(settings: EnvSettings, n_envs: int):
    from navcore.training.crowd_nav_pp.vec_env import VecCrowdSimEnv

    return VecCrowdSimEnv([(lambda: make_env(settings)) for _ in range(n_envs)])
