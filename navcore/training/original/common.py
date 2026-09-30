"""Shared helpers for the original-policy training/evaluation scripts."""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from navcore.entities.components.sensors.ray_spec import (
    RaySpec,
    check_ray_counts,
    check_ray_specs,
    default_ray_spec,
)
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


# -- ray configuration -------------------------------------------------------------


def resolve_ray_spec(num_rays: int | None, max_range: float | None) -> RaySpec:
    """The ONLY place a CLI override touches the ray configuration."""
    return default_ray_spec().with_overrides(num_rays=num_rays, max_range=max_range)


def verify_ray_wiring(env, policy, *, verbose: bool = True) -> RaySpec:
    """Fail fast (with component names + values) if any part disagrees on the ray fan."""
    base_env = env.envs[0] if hasattr(env, "envs") else env
    enc, cfg = base_env._obs_encoder, policy.config
    specs = {
        "CrowdSimEnvConfig.ray_spec": base_env.config.ray_spec,
        "ObstacleDetector.config": enc.obstacle_detector.config.spec,
        "ObservationEncoder.ray_spec": enc.ray_spec,
        "CrowdNavPPPolicy.config.ray_spec": cfg.ray_spec,
    }
    tok = getattr(policy, "obstacle_tokenizer", None)
    if tok is not None and tok.config.num_rays is not None:
        specs["ObstacleTokenizer.config"] = RaySpec(
            tok.config.num_rays, tok.config.max_range
        )
    check_ray_specs(specs, context="verify_ray_wiring")

    ref = base_env.config.ray_spec.num_rays
    counts = {
        "RaySpec (shared)": ref,
        "ObservationEncoder.space['ray_features']": enc.space["ray_features"].shape[0],
        "env.observation_space['ray_features']": env.observation_space[
            "ray_features"
        ].shape[0],
    }
    if cfg.uses_ray_features and hasattr(policy, "max_humans"):
        counts["policy.num_obstacle_slots"] = cfg.num_obstacle_slots
        counts["policy.base.human_num - max_neighbors"] = (
            policy.base.human_num - policy.max_humans
        )
    check_ray_counts(counts, context="verify_ray_wiring")

    if verbose:
        print(
            "[ray-config] source of truth: env.toml [obstacle_sensor] (+ single CLI override)"
        )
        for name, s in specs.items():
            print(
                f"[ray-config]   {name:<34} num_rays={s.num_rays:<4} max_range={s.max_range}"
            )
        for name, n in counts.items():
            print(f"[ray-config]   {name:<42} {n}")
        print(f"[ray-config] OK: all components agree on {ref} rays")
    return base_env.config.ray_spec


# -- policy ------------------------------------------------------------------------


def resolve_policy_config(
    *,
    obstacle_mode: str | None,
    max_neighbors: int | None,
    ray_spec: RaySpec,
    checkpoint_info: CheckpointInfo | None = None,
    obstacle_hit_radius: float | None = None,
) -> CrowdNavPPPolicyConfig:
    """CLI value wins; else the checkpoint's weights/config; else defaults.

    ``ray_spec`` is always the caller's (env-owned) spec; it is never restored
    from a checkpoint.
    """
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

    overrides: dict[str, Any] = {
        "obstacle_mode": mode,
        "use_obstacle_encoder": None,
        "ray_spec": ray_spec,
    }
    if max_neighbors is not None:
        overrides["max_neighbors"] = max_neighbors
    if obstacle_hit_radius is not None:
        overrides["obstacle_hit_radius"] = obstacle_hit_radius
    return dataclasses.replace(base, **overrides)


def build_policy(cfg, device, gst_predictor=None):
    return CrowdNavPPPolicy(cfg, gst_predictor=gst_predictor).to(device)


# -- environment ---------------------------------------------------------------------


@dataclass(frozen=True)
class EnvSettings:
    max_neighbors: int = 10
    history_steps: int = 8
    max_episode_steps: int = 1500
    static_obstacles: bool = False
    ray_spec: RaySpec = field(default_factory=default_ray_spec)


def resolve_static_obstacles(arg: str, cfg: CrowdNavPPPolicyConfig) -> bool:
    """'auto' -> static tables only when the policy can actually see them."""
    return {"on": True, "off": False}.get(arg, cfg.uses_ray_features)


def make_env_class():
    """Lazy so importing this module doesn't require gymnasium/rvo2 at import."""
    from navcore.gym_wrapper.crowd_sim_env import CrowdSimEnv

    class SeededCrowdSimEnv(CrowdSimEnv):
        """CrowdSimEnv whose ``Step`` RNG is seeded (see original docstring)."""

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
        ray_spec=settings.ray_spec,
    )
    return make_env_class()(GoalReachingTask(), cfg, render_mode=render_mode)


def make_vec_env(settings, n_envs, render=False):
    from navcore.training.crowd_nav_pp.vec_env import VecCrowdSimEnv

    return VecCrowdSimEnv(
        [
            (
                lambda i=i: make_env(
                    settings, render_mode="human" if render and i == 0 else None
                )
            )
            for i in range(n_envs)
        ]
    )
