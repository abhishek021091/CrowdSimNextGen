"""Format-agnostic checkpoint loading for ``navcore.policies.original``.

The caller never has to know what kind of file they hold. Recognised:

``official``  raw ``actor_critic.state_dict()`` from the official repo
              (``base.*`` / ``dist.*``), e.g. ``41200.pt``. Also accepted
              wrapped (``{"state_dict": ...}``, ``[model, ob_rms]``), with
              ``module.`` / ``_orig_mod.`` prefixes, or as a pickled
              ``nn.Module`` (see ``_alias_official_modules``).
``adapter``   state dict of ``CrowdNavPPPolicy`` that includes
              ``obstacle_encoder.*`` / ``obstacle_fusion.*`` keys.
``ppo``       container written by ``OriginalPPOTrainer.save_checkpoint``
              (weights + optimizer + counters + policy config).

Safety: ``torch.load(weights_only=True)`` is tried first. Only if that fails
is an unrestricted pickle load attempted (with a warning) -- only do that
for files you trust.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import sys
import types
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from navcore.policies.obstacle_mode import ObstacleMode

DEFAULT_PRETRAINED_NAME = "41200.pt"
PPO_FORMAT = "navcore_original_ppo_v1"
OBSTACLE_PREFIXES = ("obstacle_encoder.", "obstacle_fusion.")

_STATE_DICT_KEYS = (
    "policy_state_dict",
    "state_dict",
    "model_state_dict",
    "actor_critic",
    "model",
)
_STRIP_PREFIXES = ("module.", "_orig_mod.")
_BASE_MODULE_NAMES = (
    "humanNodeRNN.",
    "attn.",
    "spatial_attn.",
    "robot_linear.",
    "actor.",
    "critic.",
    "critic_linear.",
    "spatial_linear.",
    "human_node_final_linear.",
)


@dataclass
class CheckpointInfo:
    """What was found in (and loaded from) a checkpoint file."""

    path: str
    format: str  # "official" | "adapter" | "ppo"
    wrapped: bool
    weights_layout: str  # "official" (no obstacle keys) | "adapter"
    obstacle_mode: ObstacleMode  # inferred from weights; NONE if no obstacle keys
    has_optimizer: bool = False
    total_steps: int | None = None
    total_updates: int | None = None
    policy_config: dict[str, Any] | None = None
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.path}: format={self.format} layout={self.weights_layout} "
            f"obstacle_mode={self.obstacle_mode.name} wrapped={self.wrapped} "
            f"steps={self.total_steps} missing={len(self.missing_keys)} "
            f"unexpected={len(self.unexpected_keys)}"
        )


# -- default pretrained checkpoint --------------------------------------------


def default_pretrained_candidates() -> list[Path]:
    here = Path(__file__).resolve()
    navcore_dir = here.parents[2]  # .../navcore
    repo_root = here.parents[3]
    env = os.environ.get("NAVCORE_PRETRAINED_CHECKPOINT")
    cands = [Path(env)] if env else []
    cands += [
        Path.cwd() / DEFAULT_PRETRAINED_NAME,
        repo_root / DEFAULT_PRETRAINED_NAME,
        navcore_dir / "policies" / "original" / "checkpoints" / DEFAULT_PRETRAINED_NAME,
        navcore_dir / "policies" / "original" / DEFAULT_PRETRAINED_NAME,
        here.parent / "pretrained" / DEFAULT_PRETRAINED_NAME,
    ]
    return cands


def resolve_pretrained(path: str | os.PathLike | None = None) -> Path:
    """Explicit path if given, else search the default locations for 41200.pt.

    Raises FileNotFoundError (never silently falls back to scratch).
    """
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    cands = default_pretrained_candidates()
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"Default pretrained checkpoint {DEFAULT_PRETRAINED_NAME!r} not found. Searched:\n  "
        + "\n  ".join(map(str, cands))
        + "\nPass --pretrained PATH, set NAVCORE_PRETRAINED_CHECKPOINT, or use --no-pretrained."
    )


# -- loading ---------------------------------------------------------------------


def _alias_official_modules() -> None:
    """Let torch.load unpickle a whole official ``nn.Module`` (classes live at
    ``rl.networks.*`` upstream) by aliasing those names to the ported modules.
    Untested against a real official pickle."""
    if "rl" not in sys.modules:
        sys.modules["rl"] = types.ModuleType("rl")
    if "rl.networks" not in sys.modules:
        pkg = types.ModuleType("rl.networks")
        sys.modules["rl.networks"] = pkg
        sys.modules["rl"].networks = pkg  # type: ignore[attr-defined]
    for name in (
        "model",
        "srnn_model",
        "selfAttn_srnn_temp_node",
        "distributions",
        "network_utils",
    ):
        mod = importlib.import_module(f"navcore.policies.original.{name}")
        sys.modules[f"rl.networks.{name}"] = mod
        setattr(sys.modules["rl.networks"], name, mod)


def _load(path: Path, weights_only: bool):
    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except TypeError:  # very old torch without the kwarg
        return torch.load(path, map_location="cpu")


def _torch_load(path: Path):
    try:
        return _load(path, True)
    except Exception as first:  # noqa: BLE001
        warnings.warn(
            f"weights_only load of {path} failed ({type(first).__name__}: "
            f"{str(first)[:120]}); retrying with full pickle. Only do this for trusted files."
        )
    try:
        return _load(path, False)
    except ModuleNotFoundError as e:
        if not str(e).startswith("No module named 'rl"):
            raise
        _alias_official_modules()
        return _load(path, False)


def _looks_like_state_dict(d: Any) -> bool:
    return (
        isinstance(d, Mapping)
        and len(d) > 0
        and all(isinstance(k, str) for k in d)
        and all(torch.is_tensor(v) for v in d.values())
    )


def _extract(obj: Any, depth: int = 0) -> tuple[dict, dict] | None:
    """Return (state_dict, meta) where meta['container'] is the outermost dict."""
    if depth > 4:
        return None
    if isinstance(obj, nn.Module):
        return dict(obj.state_dict()), {"module": True}
    if isinstance(obj, Mapping):
        if _looks_like_state_dict(obj):
            return dict(obj), {}
        for key in _STATE_DICT_KEYS:
            if key in obj:
                found = _extract(obj[key], depth + 1)
                if found is not None:
                    return found[0], {**found[1], "container": obj, "wrapped": True}
        return None
    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = _extract(item, depth + 1)
            if found is not None:
                return found[0], {**found[1], "wrapped": True}
    return None


def _normalize_keys(sd: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = dict(sd)
    changed = True
    while changed:
        changed = False
        for pfx in _STRIP_PREFIXES:
            if out and all(k.startswith(pfx) for k in out):
                out = {k[len(pfx) :]: v for k, v in out.items()}
                changed = True
    if not any(k.startswith(("base.", "dist.")) for k in out) and any(
        k.startswith(_BASE_MODULE_NAMES) for k in out
    ):
        out = {f"base.{k}": v for k, v in out.items()}  # base-only save
    return out


def infer_obstacle_mode(sd: Mapping[str, Any]) -> ObstacleMode:
    """POINT_TOKENS -> ``obstacle_encoder.mlp.*``; ENCODER -> ``obstacle_encoder.net.*``."""
    if any(k.startswith("obstacle_encoder.mlp.") for k in sd):
        return ObstacleMode.POINT_TOKENS
    if any(k.startswith("obstacle_encoder.net.") for k in sd):
        return ObstacleMode.ENCODER
    return ObstacleMode.NONE


def inspect_checkpoint(
    path: str | os.PathLike,
) -> tuple[dict[str, torch.Tensor], CheckpointInfo]:
    """Load a file and classify it, without touching any policy."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    obj = _torch_load(path)
    found = _extract(obj)
    if found is None:
        keys = list(obj.keys())[:8] if isinstance(obj, Mapping) else None
        raise ValueError(
            f"Unrecognised checkpoint {path}: top-level {type(obj).__name__}, keys={keys}. "
            "Expected a state dict, a wrapper containing one, or an nn.Module."
        )
    sd, meta = found
    sd = _normalize_keys(sd)
    container = meta.get("container") or {}
    is_ppo = isinstance(container, Mapping) and (
        container.get("format") == PPO_FORMAT
        or "optimizer_state_dict" in container
        or "total_steps" in container
    )
    layout = (
        "adapter" if any(k.startswith(OBSTACLE_PREFIXES) for k in sd) else "official"
    )
    info = CheckpointInfo(
        path=str(path),
        format="ppo" if is_ppo else layout,
        wrapped=bool(meta.get("wrapped") or meta.get("module")),
        weights_layout=layout,
        obstacle_mode=infer_obstacle_mode(sd),
    )
    if is_ppo:
        info.has_optimizer = "optimizer_state_dict" in container
        info.total_steps = container.get("total_steps")
        info.total_updates = container.get("total_updates")
        info.policy_config = container.get("policy_config")
    return sd, info


def load_policy_checkpoint(
    policy: nn.Module, path: str | os.PathLike, strict: bool = True
) -> CheckpointInfo:
    """Load weights from any supported format into ``policy``.

    Missing ``obstacle_*`` keys are tolerated (all-or-nothing, enforced by the
    adapter's ``load_state_dict``) so an official checkpoint can initialise an
    obstacle-enabled policy; those modules then keep their fresh init.
    Loading an obstacle checkpoint into a policy of a *different* obstacle
    mode raises.
    """
    sd, info = inspect_checkpoint(path)
    policy_mode = policy.config.obstacle_mode
    if (
        info.obstacle_mode is not ObstacleMode.NONE
        and info.obstacle_mode is not policy_mode
    ):
        raise ValueError(
            f"Checkpoint {path} holds a {info.obstacle_mode.name} obstacle branch but the "
            f"policy is {policy_mode.name}. Use --obstacle-mode {info.obstacle_mode.value} "
            f"(or 'auto' when evaluating)."
        )
    # The adapter registers gst_predictor as a submodule; official checkpoints
    # never contain its keys, so keep the policy's current values for them.
    for k, v in policy.state_dict().items():
        if k.startswith("gst_predictor.") and k not in sd:
            sd[k] = v
    result = policy.load_state_dict(sd, strict=strict)
    info.missing_keys = list(getattr(result, "missing_keys", []))
    info.unexpected_keys = list(getattr(result, "unexpected_keys", []))
    return info


# -- config (de)serialisation for PPO checkpoints --------------------------------


def config_to_dict(cfg: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        out[f.name] = v.name if isinstance(v, Enum) else v
    return out


def config_from_dict(cls: type, d: Mapping[str, Any]):
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name == "use_obstacle_encoder" or f.name not in d:
            continue  # derived field
        v = d[f.name]
        if f.name == "obstacle_mode" and isinstance(v, str):
            v = ObstacleMode[v]
        kwargs[f.name] = v
    return cls(**kwargs)
