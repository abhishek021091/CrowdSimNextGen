"""Smoke tests for the ray-hit -> pseudo-human token path (ObstacleMode.POINT_TOKENS).

Same harness convention as test_crowdnav_pp_smoke.py (no pytest dependency).
Uses synthetic tensors rather than a live CrowdSimEnv so a failure points at
the tokenizer/policy wiring, not at environment spawn luck.

Run directly:

    python -m navcore.test_obstacle_tokens_smoke
"""

from __future__ import annotations

import sys
import traceback

import torch

from navcore.entities.components.sensors.obstacle_detector import RAY_FEATURE_DIM
from navcore.policies.crowdnav_pp.obstacle_encoder import (
    ObstacleEncoder,
    ObstacleEncoderConfig,
)
from navcore.policies.crowdnav_pp.obstacle_tokenizer import (
    ObstacleTokenizer,
    ObstacleTokenizerConfig,
)
from navcore.policies.crowdnav_pp.crowd_nav_pp_policy import (
    CrowdNavPPPolicy,
    CrowdNavPPPolicyConfig,
    ObstacleMode,
)

NUM_RAYS = 20
MAX_NEIGHBORS = 10
HISTORY_STEPS = 8
MAX_RANGE = 5.0

_CHECKS: list[tuple[str, callable]] = []


def check(name: str):
    def decorator(fn):
        _CHECKS.append((name, fn))
        return fn

    return decorator


def _make_batch(
    nenv: int = 2, n_humans: int = 2, n_hits: int = 5, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Synthetic single-tick batch with ``n_humans`` real neighbors and
    ``n_hits`` rays that hit something."""
    g = torch.Generator().manual_seed(seed)
    batch = {
        "robot": torch.randn(nenv, 8, generator=g),
        "neighbors": torch.zeros(nenv, MAX_NEIGHBORS, 5),
        "neighbor_mask": torch.zeros(nenv, MAX_NEIGHBORS),
        "neighbor_history": torch.zeros(nenv, HISTORY_STEPS, MAX_NEIGHBORS, 5),
        "neighbor_history_mask": torch.zeros(nenv, HISTORY_STEPS, MAX_NEIGHBORS),
        "ray_features": torch.zeros(nenv, NUM_RAYS, RAY_FEATURE_DIM),
    }
    if n_humans:
        batch["neighbors"][:, :n_humans] = torch.randn(nenv, n_humans, 5, generator=g)
        batch["neighbor_mask"][:, :n_humans] = 1
        batch["neighbor_history"][:, :, :n_humans] = torch.randn(
            nenv, HISTORY_STEPS, n_humans, 5, generator=g
        )
        batch["neighbor_history_mask"][:, :, :n_humans] = 1
    if n_hits:
        batch["ray_features"][:, :n_hits, 0] = 1.0
        batch["ray_features"][:, :n_hits, 1:] = (
            torch.rand(nenv, n_hits, 2, generator=g) * 2 - 1
        )
    return batch


def _run(policy: CrowdNavPPPolicy, batch, use_rays: bool = True, deterministic=True):
    nenv = batch["robot"].shape[0]
    return policy.forward(
        batch["robot"],
        batch["neighbors"],
        batch["neighbor_mask"],
        batch["neighbor_history"],
        batch["neighbor_history_mask"],
        policy.initial_hidden_state(nenv=nenv),
        torch.ones(nenv),
        ray_features=batch["ray_features"] if use_rays else None,
    )


def _point_policy(**overrides) -> CrowdNavPPPolicy:
    torch.manual_seed(0)
    config = CrowdNavPPPolicyConfig(
        obstacle_mode=ObstacleMode.POINT_TOKENS,
        obstacle_max_range=MAX_RANGE,
        **overrides,
    )
    policy = CrowdNavPPPolicy(config)
    policy.eval()
    return policy


@check("1. Tokenizer contract: meters, zero velocity, radius 0.3, padding zeroed")
def _test_tokenizer_contract():
    tokenizer = ObstacleTokenizer(
        ObstacleTokenizerConfig(max_range=MAX_RANGE, hit_radius=0.3)
    )
    rays = torch.zeros(1, NUM_RAYS, RAY_FEATURE_DIM)
    rays[0, 3] = torch.tensor([1.0, 0.4, -0.2])  # hit at (2.0, -1.0) m

    tokens, mask = tokenizer.tokenize(rays)

    assert tokens.shape == (1, NUM_RAYS, 5), tokens.shape
    assert mask.dtype == torch.bool and int(mask.sum()) == 1 and bool(mask[0, 3])
    expected = torch.tensor([2.0, -1.0, 0.0, 0.0, 0.3])
    assert torch.allclose(tokens[0, 3], expected, atol=1e-6), tokens[0, 3]
    assert torch.count_nonzero(tokens[0, [i for i in range(NUM_RAYS) if i != 3]]) == 0


@check("2. POINT_TOKENS forward()/act(): shapes correct, outputs finite")
def _test_forward_shapes():
    policy = _point_policy()
    batch = _make_batch(nenv=3)
    distribution, value, new_hidden = _run(policy, batch)

    assert value.shape == (3, 1)
    assert new_hidden.shape == (3, policy.config.rnn_hidden_size)
    assert distribution.sample().shape == (3, policy.config.action_dim)
    assert torch.isfinite(distribution.mean).all() and torch.isfinite(value).all()


@check("3. Obstacle hits actually change the output (not just the dummy human)")
def _test_obstacles_influence_output():
    policy = _point_policy()
    with_hits = _make_batch(n_humans=2, n_hits=6)
    without_hits = {k: v.clone() for k, v in with_hits.items()}
    without_hits["ray_features"].zero_()

    with torch.no_grad():
        mean_a = _run(policy, with_hits)[0].mean
        mean_b = _run(policy, without_hits)[0].mean
    assert not torch.allclose(mean_a, mean_b, atol=1e-6), (
        "Output identical with and without ray hits -- obstacle tokens are "
        "not reaching the attention layers."
    )


@check("4. Ray order is irrelevant (permutation invariance over hit tokens)")
def _test_ray_permutation_invariance():
    policy = _point_policy()
    batch = _make_batch(n_humans=2, n_hits=6)
    permuted = {k: v.clone() for k, v in batch.items()}
    perm = torch.randperm(NUM_RAYS, generator=torch.Generator().manual_seed(1))
    permuted["ray_features"] = batch["ray_features"][:, perm]

    with torch.no_grad():
        mean_a = _run(policy, batch)[0].mean
        mean_b = _run(policy, permuted)[0].mean
    assert torch.allclose(mean_a, mean_b, atol=1e-5), (
        "Output depends on ray index -- obstacle tokens should be an unordered set."
    )


@check("5. Zero humans and zero hits: dummy-human substitution, no NaN")
def _test_empty_scene_no_nan():
    policy = _point_policy()
    batch = _make_batch(n_humans=0, n_hits=0)
    with torch.no_grad():
        distribution, value, _ = _run(policy, batch)
    assert torch.isfinite(distribution.mean).all() and torch.isfinite(value).all()


@check("6. Obstacles only (no humans) still runs and hits are attended")
def _test_obstacles_only_scene():
    policy = _point_policy()
    scene = _make_batch(n_humans=0, n_hits=6)
    empty = {k: v.clone() for k, v in scene.items()}
    empty["ray_features"].zero_()
    with torch.no_grad():
        mean_a = _run(policy, scene)[0].mean
        mean_b = _run(policy, empty)[0].mean
    assert torch.isfinite(mean_a).all()
    assert not torch.allclose(mean_a, mean_b, atol=1e-6)


@check("7. Gradients reach the shared embedding MLP through obstacle tokens")
def _test_gradient_through_obstacle_tokens():
    torch.manual_seed(0)
    policy = CrowdNavPPPolicy(
        CrowdNavPPPolicyConfig(
            obstacle_mode=ObstacleMode.POINT_TOKENS, obstacle_max_range=MAX_RANGE
        )
    )
    policy.train()
    batch = _make_batch(n_humans=2, n_hits=6)
    hidden = torch.randn(2, policy.config.rnn_hidden_size)  # nonzero: see smoke test 4

    distribution, value, _ = policy.forward(
        batch["robot"],
        batch["neighbors"],
        batch["neighbor_mask"],
        batch["neighbor_history"],
        batch["neighbor_history_mask"],
        hidden,
        torch.ones(2),
        ray_features=batch["ray_features"],
    )
    (distribution.rsample().pow(2).sum() + value.sum()).backward()

    grad = policy.human_human_attention.embed[0].weight.grad
    assert grad is not None and torch.any(grad != 0)
    # The is_obstacle flag is the LAST input column; it only receives
    # gradient if obstacle tokens actually pass through this layer.
    assert torch.any(grad[:, -1] != 0), "is_obstacle flag column got no gradient."


@check("8. Misconfiguration fails loudly")
def _test_misconfiguration():
    # Injected encoder in POINT_TOKENS mode would register unused params.
    try:
        CrowdNavPPPolicy(
            CrowdNavPPPolicyConfig(obstacle_mode=ObstacleMode.POINT_TOKENS),
            obstacle_encoder=ObstacleEncoder(ObstacleEncoderConfig()),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Injected obstacle_encoder was silently accepted.")

    policy = _point_policy()
    batch = _make_batch()
    try:
        _run(policy, batch, use_rays=False)
    except ValueError:
        pass
    else:
        raise AssertionError("POINT_TOKENS forward() accepted missing ray_features.")

    none_policy = CrowdNavPPPolicy(CrowdNavPPPolicyConfig())
    try:
        _run(none_policy, batch, use_rays=True)
    except ValueError:
        pass
    else:
        raise AssertionError("NONE mode forward() accepted stray ray_features.")


@check("9. ENCODER mode (ablation baseline) still runs and trains its encoder")
def _test_encoder_mode_regression():
    torch.manual_seed(0)
    encoder = ObstacleEncoder(ObstacleEncoderConfig(embedding_dim=64))
    policy = CrowdNavPPPolicy(
        CrowdNavPPPolicyConfig(obstacle_mode=ObstacleMode.ENCODER),
        obstacle_encoder=encoder,
    )
    policy.train()
    batch = _make_batch(n_humans=2, n_hits=6)
    hidden = torch.randn(2, policy.config.rnn_hidden_size)

    distribution, value, _ = policy.forward(
        batch["robot"],
        batch["neighbors"],
        batch["neighbor_mask"],
        batch["neighbor_history"],
        batch["neighbor_history_mask"],
        hidden,
        torch.ones(2),
        ray_features=batch["ray_features"],
    )
    assert value.shape == (2, 1)
    (distribution.rsample().pow(2).sum() + value.sum()).backward()
    grad = encoder.net[0].weight.grad
    assert grad is not None and torch.any(grad != 0)


def main() -> int:
    failed = 0
    for name, fn in _CHECKS:
        print(f"--- {name} " + "-" * max(1, 78 - len(name) - 5))
        try:
            fn()
        except Exception:
            failed += 1
            print("FAIL")
            print(traceback.format_exc())
        else:
            print("PASS")
        print()
    print(f"SUMMARY: {len(_CHECKS) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
