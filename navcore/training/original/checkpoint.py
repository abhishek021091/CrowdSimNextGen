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

_RULE = "=" * 58
_GST_PREFIX = "gst_predictor."

#: (state_dict prefix, display name). Used only for reporting.
_MODULE_NAMES: tuple[tuple[str, str], ...] = (
    ("base.spatial_attn", "human_human_attention"),
    ("base.attn", "robot_human_attention"),
    ("base.spatial_linear", "spatial_embedding"),
    ("base.robot_linear", "robot_encoder"),
    ("base.humanNodeRNN", "recurrent_encoder"),
    ("base.actor", "actor"),
    ("base.critic", "critic"),
    ("base.critic_linear", "critic_head"),
    ("base.human_node_final_linear", "aux_head (frozen)"),
    ("dist", "action_head"),
    ("obstacle_encoder", "obstacle_encoder"),
    ("obstacle_fusion", "obstacle_fusion"),
    ("gst_predictor", "gst_predictor"),
)

_PPO_CONFIG_KEYS = (
    "obstacle_mode",
    "max_neighbors",
    "use_gst_prediction",
    "obstacle_max_range",
    "rnn_hidden_size",
    "node_output_size",
    "action_dim",
)


def module_of(key: str) -> str:
    """Map a state_dict key to its major component name (reporting only)."""
    for prefix, name in _MODULE_NAMES:
        if key == prefix or key.startswith(prefix + "."):
            return name
    return key.split(".", 1)[0]


def _rows(rows, width: int = 17) -> list[str]:
    return ["" if r is None else f"{r[0]:<{width}}: {r[1]}" for r in rows]


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
    load_report: LoadReport | None = None

    def summary(self) -> str:
        loaded = self.load_report is not None
        rows = [
            ("Path", self.path),
            ("Format", self.format),
            ("Weights layout", self.weights_layout),
            ("Obstacle mode", self.obstacle_mode.name),
            ("Wrapped", self.wrapped),
            None,
            ("Optimizer", "Yes" if self.has_optimizer else "No"),
            ("Training steps", self.total_steps),
            ("Updates", self.total_updates),
            None,
            ("Missing keys", len(self.missing_keys) if loaded else "n/a (not loaded)"),
            (
                "Unexpected keys",
                len(self.unexpected_keys) if loaded else "n/a (not loaded)",
            ),
        ]
        return "\n".join([_RULE, "Checkpoint", _RULE, *_rows(rows), _RULE])


class CheckpointShapeError(RuntimeError):
    """Tensor present in checkpoint and model, but with different shapes."""


@dataclass(slots=True, frozen=True)
class ShapeMismatch:
    key: str
    checkpoint: tuple[int, ...]
    model: tuple[int, ...]


@dataclass(slots=True)
class LoadReport:
    """Pure bookkeeping about one checkpoint -> policy load (no torch state).

    Attributes:
        missing_keys: Model keys absent from the checkpoint (excluding GST
            keys, which ``load_policy_checkpoint`` keeps at current values).
        kept_keys: GST keys absent from the checkpoint and kept as-is.
        unexpected_keys: Checkpoint keys the model does not have.
        loaded_numel / random_numel / total_numel: Counted by element
            count. ``random_numel`` = everything not taken from the file.
        module_status: component name -> (tensors from checkpoint, tensors in model).
    """

    checkpoint_tensors: int
    model_tensors: int
    matched_tensors: int
    missing_keys: list[str]
    kept_keys: list[str]
    unexpected_keys: list[str]
    loaded_numel: int
    random_numel: int
    total_numel: int
    module_status: dict[str, tuple[int, int]]

    @property
    def loaded_ratio(self) -> float:
        return self.loaded_numel / self.total_numel if self.total_numel else 0.0

    @property
    def loaded_modules(self) -> list[str]:
        return [m for m, (l, t) in self.module_status.items() if l == t]

    @property
    def fresh_modules(self) -> list[str]:
        return [m for m, (l, _) in self.module_status.items() if l == 0]

    @property
    def partial_modules(self) -> list[str]:
        return [m for m, (l, t) in self.module_status.items() if 0 < l < t]


def find_shape_mismatches(
    model_sd: Mapping[str, torch.Tensor], ckpt_sd: Mapping[str, torch.Tensor]
) -> list[ShapeMismatch]:
    return [
        ShapeMismatch(k, tuple(ckpt_sd[k].shape), tuple(v.shape))
        for k, v in model_sd.items()
        if k in ckpt_sd and tuple(ckpt_sd[k].shape) != tuple(v.shape)
    ]


def format_shape_mismatches(path: str, mismatches: list[ShapeMismatch]) -> str:
    blocks = [f"Shape mismatch while loading {path}", ""]
    for m in mismatches:
        blocks += [
            m.key,
            "",
            "Checkpoint:",
            str(m.checkpoint),
            "",
            "Model:",
            str(m.model),
            "",
        ]
    return "\n".join(blocks).rstrip()


def build_load_report(
    model_sd: Mapping[str, torch.Tensor], ckpt_sd: Mapping[str, torch.Tensor]
) -> LoadReport:
    """Diff key sets and count parameters. Call only after the shape check."""
    kept = [k for k in model_sd if k.startswith(_GST_PREFIX) and k not in ckpt_sd]
    kept_set = set(kept)
    matched = [k for k in model_sd if k in ckpt_sd]
    status: dict[str, tuple[int, int]] = {}
    for k in model_sd:
        name = module_of(k)
        got, total = status.get(name, (0, 0))
        status[name] = (got + (k in ckpt_sd), total + 1)
    total_numel = sum(v.numel() for v in model_sd.values())
    loaded_numel = sum(model_sd[k].numel() for k in matched)
    return LoadReport(
        checkpoint_tensors=len(ckpt_sd),
        model_tensors=len(model_sd),
        matched_tensors=len(matched),
        missing_keys=[k for k in model_sd if k not in ckpt_sd and k not in kept_set],
        kept_keys=kept,
        unexpected_keys=[k for k in ckpt_sd if k not in model_sd],
        loaded_numel=loaded_numel,
        random_numel=total_numel - loaded_numel,
        total_numel=total_numel,
        module_status=status,
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
    policy: nn.Module,
    path: str | os.PathLike,
    strict: bool = True,
    verbose: bool = True,
) -> CheckpointInfo:
    """Load weights from any supported format into ``policy``.

    Pipeline: inspect -> obstacle-mode check (unchanged ValueError) ->
    shape check (detailed CheckpointShapeError) -> key diff -> load ->
    ``validate_checkpoint``. Missing ``obstacle_*`` keys are still tolerated
    all-or-nothing by the adapter's ``load_state_dict``.
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

    model_sd = policy.state_dict()
    mismatches = find_shape_mismatches(model_sd, sd)
    if mismatches:
        raise CheckpointShapeError(format_shape_mismatches(str(path), mismatches))

    report = build_load_report(model_sd, sd)
    info.missing_keys = list(report.missing_keys)
    info.unexpected_keys = list(report.unexpected_keys)
    info.load_report = report

    # The adapter registers gst_predictor as a submodule; official checkpoints
    # never contain its keys, so keep the policy's current values for them.
    for k in report.kept_keys:
        sd[k] = model_sd[k]

    try:
        result = policy.load_state_dict(sd, strict=strict)
    except RuntimeError:
        if verbose:  # show exact key names before the exception propagates
            print(
                "\n".join(
                    [
                        info.summary(),
                        "",
                        *_key_section("Missing keys", report.missing_keys),
                        *_key_section("Unexpected keys", report.unexpected_keys),
                    ]
                )
            )
        raise
    info.missing_keys = list(getattr(result, "missing_keys", []))
    info.unexpected_keys = list(getattr(result, "unexpected_keys", []))
    validate_checkpoint(policy, info, report, verbose=verbose)
    return info


# -- reporting ---------------------------------------------------------------


def _key_section(title: str, keys: list[str]) -> list[str]:
    return [title, "", *([f"  {k}" for k in keys] or ["  (none)"]), ""]


def _explain_missing(info: CheckpointInfo, report: LoadReport) -> list[str]:
    lines: list[str] = []
    obstacle = sorted(
        {module_of(k) for k in report.missing_keys if k.startswith(OBSTACLE_PREFIXES)}
    )
    if obstacle:
        lines += [
            "Checkpoint does not contain obstacle weights.",
            "",
            "The following modules are randomly initialized:",
            "",
            *[f"- {m}" for m in obstacle],
            "",
            "Training will continue using pretrained CrowdNav weights plus "
            "newly initialized obstacle modules.",
            "",
        ]
    if report.kept_keys:
        lines += [
            "Checkpoint does not contain GST predictor weights.",
            "",
            "gst_predictor keeps its current values (pretrained only if you "
            "loaded one via GSTPredictorTrainer.load_predictor; otherwise random).",
            "",
        ]
    other = sorted(
        {
            module_of(k)
            for k in report.missing_keys
            if not k.startswith(OBSTACLE_PREFIXES)
        }
    )
    if other:
        lines += [
            "Other modules missing from the checkpoint (randomly initialized):",
            "",
            *[f"- {m}" for m in other],
            "",
        ]
    if report.unexpected_keys:
        lines += [
            "The checkpoint holds tensors this policy has no slot for; they were ignored.",
            "",
        ]
    return lines


def _obstacle_compat(info: CheckpointInfo, policy_mode: ObstacleMode) -> list[str]:
    ck = info.obstacle_mode
    if ck is ObstacleMode.NONE and policy_mode is not ObstacleMode.NONE:
        return [
            "Checkpoint contains no obstacle branch.",
            "",
            "Obstacle modules will use fresh initialization.",
        ]
    if ck is not ObstacleMode.NONE and ck is policy_mode:
        return [f"Obstacle branch ({ck.name}) restored from checkpoint."]
    return ["Obstacle branch: not used by checkpoint or policy."]


def _ppo_block(info: CheckpointInfo) -> list[str]:
    cfg = info.policy_config
    rows: list = [
        ("Training steps", info.total_steps),
        ("Training updates", info.total_updates),
    ]
    lines = ["PPO checkpoint", "", *_rows(rows), "", "Policy configuration"]
    if not cfg:
        return [*lines, "  not stored in this checkpoint", ""]
    for k in _PPO_CONFIG_KEYS:
        if k in cfg:
            lines.append(f"  {k:<19}: {cfg[k]}")
    # Not part of policy_config or of the checkpoint format, so it cannot be reported.
    lines += ["  history_steps      : not stored in checkpoint (env setting)", ""]
    return lines


def _collect_warnings(
    policy: nn.Module, info: CheckpointInfo, report: LoadReport
) -> list[str]:
    warns: list[str] = []
    if report.loaded_numel == 0:
        warns.append("No parameters were loaded from this checkpoint.")
    for m in sorted(
        {
            module_of(k)
            for k in report.missing_keys
            if not k.startswith(OBSTACLE_PREFIXES)
        }
    ):
        warns.append(
            f"Module '{m}' is missing from the checkpoint and randomly initialized."
        )
    if report.unexpected_keys:
        warns.append(
            f"{len(report.unexpected_keys)} unexpected checkpoint tensors were ignored."
        )
    if info.format == "ppo" and not info.has_optimizer:
        warns.append("PPO checkpoint has no optimizer state.")
    saved, cfg = info.policy_config or {}, getattr(policy, "config", None)
    if saved and cfg is not None:
        for name in ("max_neighbors", "use_gst_prediction"):
            if (
                name in saved
                and hasattr(cfg, name)
                and saved[name] != getattr(cfg, name)
            ):
                warns.append(
                    f"Saved {name}={saved[name]!r} differs from the policy's {getattr(cfg, name)!r}."
                )
        mode = getattr(cfg, "obstacle_mode", None)
        if (
            "obstacle_mode" in saved
            and mode is not None
            and saved["obstacle_mode"] != mode.name
        ):
            warns.append(
                f"Saved obstacle_mode={saved['obstacle_mode']} differs from the policy's {mode.name}."
            )
    return warns


def validate_checkpoint(
    policy: nn.Module,
    info: CheckpointInfo,
    report: LoadReport,
    verbose: bool = True,
) -> list[str]:
    """Print a full post-load report and return the list of warnings."""
    warns = _collect_warnings(policy, info, report)
    out: list[str] = [info.summary(), ""]
    if info.format == "ppo":
        out += _ppo_block(info)
    out += [
        "Tensors",
        "",
        *_rows(
            [
                ("Checkpoint", report.checkpoint_tensors),
                ("Model", report.model_tensors),
                ("Matched", report.matched_tensors),
                ("Missing", len(report.missing_keys)),
                ("Unexpected", len(report.unexpected_keys)),
            ]
        ),
        "",
        "Parameters (by element count)",
        "",
        *_rows(
            [
                ("Loaded", f"{report.loaded_numel:,}"),
                ("Random init", f"{report.random_numel:,}"),
                ("Total", f"{report.total_numel:,}"),
                None,
                ("Loaded ratio", f"{report.loaded_ratio:.2%}"),
            ]
        ),
        "",
        *_key_section("Missing keys", report.missing_keys),
        *_key_section("Unexpected keys", report.unexpected_keys),
        *_explain_missing(info, report),
        "Loaded modules",
        "",
        *[f"  ✓ {m}" for m in report.loaded_modules],
        "",
        "Fresh initialization",
        "",
        *([f"  • {m}" for m in report.fresh_modules] or ["  (none)"]),
    ]
    if report.partial_modules:
        out += [
            "",
            "Partially loaded",
            "",
            *[
                f"  ~ {m} ({report.module_status[m][0]}/{report.module_status[m][1]} tensors)"
                for m in report.partial_modules
            ],
        ]
    out += [
        "",
        *_obstacle_compat(info, policy.config.obstacle_mode),
        "",
        f"Optimizer state in file: {'yes' if info.has_optimizer else 'no'}",
    ]
    if warns:
        out += ["", "Warnings", "", *[f"  ! {w}" for w in warns]]
    out.append(_RULE)
    if verbose:
        print("\n".join(out))
    return warns


def describe_optimizer_state(info: CheckpointInfo, restored: bool | None) -> str:
    """Explain what happened to optimizer state.

    ``restored``: True/False after a resume attempt; None when only weights
    were loaded (``--checkpoint`` / evaluation).
    """
    if restored:
        return (
            "Optimizer state restored.\n"
            "Training will resume exactly where it stopped.\n"
            "(Environments and recurrent hidden states are re-initialised; "
            "they are not checkpointed.)"
        )
    if info.has_optimizer and restored is None:
        return (
            "Optimizer state is present but was not restored (weights-only load).\n"
            "Use --resume to continue training exactly where it stopped."
        )
    if info.has_optimizer:
        return "Optimizer state could not be restored.\nAdam optimizer will start from scratch."
    return (
        "Optimizer state not found.\n\n"
        "Adam optimizer will start from scratch.\n"
        "Learning-rate schedule restarts."
    )


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
