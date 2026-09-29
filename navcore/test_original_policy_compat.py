"""Checkpoint-compat, forward-equivalence and obstacle-decoupling checks for
navcore.policies.original.

Run:  python -m navcore.test_original_policy_compat

Check 7 additionally diffs against the REAL official repo if it is importable
(put Shuijing725/CrowdNav_Prediction_AttnGraph on PYTHONPATH so that
``rl.networks.model`` resolves); otherwise it is skipped.
"""

from __future__ import annotations

import sys
import traceback

import torch

from navcore.policies.original.adapter import (
    CrowdNavPPPolicy,
    CrowdNavPPPolicyConfig,
    ObstacleMode,
    build_action_space,
    build_base_args,
    build_obs_space,
)

NUM_RAYS, MAX_N, HIST = 20, 10, 8
_CHECKS: list = []


def check(name):
    def deco(fn):
        _CHECKS.append((name, fn))
        return fn

    return deco


def _batch(nenv=2, n_humans=3, n_hits=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    b = {
        "robot": torch.randn(nenv, 8, generator=g),
        "neighbors": torch.zeros(nenv, MAX_N, 5),
        "neighbor_mask": torch.zeros(nenv, MAX_N, dtype=torch.int8),
        "neighbor_history": torch.zeros(nenv, HIST, MAX_N, 5),
        "neighbor_history_mask": torch.zeros(nenv, HIST, MAX_N, dtype=torch.int8),
        "ray_features": torch.zeros(nenv, NUM_RAYS, 3),
    }
    if n_humans:
        b["neighbors"][:, :n_humans] = torch.randn(nenv, n_humans, 5, generator=g)
        b["neighbor_mask"][:, :n_humans] = 1
    if n_hits:
        b["ray_features"][:, :n_hits, 0] = 1.0
        b["ray_features"][:, :n_hits, 1:] = (
            torch.rand(nenv, n_hits, 2, generator=g) * 2 - 1
        )
    return b


def _policy(mode):
    torch.manual_seed(0)
    p = CrowdNavPPPolicy(CrowdNavPPPolicyConfig(obstacle_mode=mode))
    p.eval()
    return p


def _call(p, b, use_rays=True):
    n = b["robot"].shape[0]
    return p.forward(
        b["robot"],
        b["neighbors"],
        b["neighbor_mask"],
        b["neighbor_history"],
        b["neighbor_history_mask"],
        torch.full((n, 128), 0.1),
        torch.ones(n),
        ray_features=b["ray_features"] if use_rays else None,
    )


@check("1. NONE-mode state_dict contains only official base.*/dist.* keys")
def _t1():
    keys = list(_policy(ObstacleMode.NONE).state_dict())
    assert keys and all(k.startswith(("base.", "dist.")) for k in keys), keys


@check("2. Strict load: official keys OK; partial/unexpected/shape errors raise")
def _t2():
    sd = _policy(ObstacleMode.NONE).state_dict()
    tgt = _policy(ObstacleMode.POINT_TOKENS)
    res = tgt.load_state_dict(sd)
    assert not res.unexpected_keys and res.missing_keys
    assert all(
        k.startswith(("obstacle_encoder.", "obstacle_fusion."))
        for k in res.missing_keys
    )
    assert torch.equal(
        tgt.state_dict()["base.critic_linear.weight"], sd["base.critic_linear.weight"]
    )

    for bad in (
        {k: v for k, v in sd.items() if k != "base.critic_linear.bias"},
        {**sd, "base.bogus": torch.zeros(1)},
    ):
        try:
            _policy(ObstacleMode.NONE).load_state_dict(bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError("bad checkpoint accepted")

    partial = {
        **sd,
        "obstacle_fusion.bias": torch.zeros(256),
    }  # obstacle branch half-present
    try:
        tgt.load_state_dict(partial)
    except RuntimeError:
        pass
    else:
        raise AssertionError("partial obstacle branch accepted")


@check("3. Adapter forward == direct official base.forward (NONE mode)")
def _t3():
    p, b = _policy(ObstacleMode.NONE), _batch(nenv=3)
    with torch.no_grad():
        dist, value, hid = _call(p, b, use_rays=False)
        inputs = p._build_inputs(
            b["robot"],
            b["neighbors"],
            b["neighbor_mask"],
            b["neighbor_history"],
            b["neighbor_history_mask"],
        )
        rnn = {
            "human_node_rnn": torch.full((3, 1, 128), 0.1),
            "human_human_edge_rnn": torch.zeros(3, 1, 256),
        }
        p.base.nenv = 3
        v, af, out = p.base(inputs, rnn, torch.ones(3, 1), infer=True)
    assert torch.allclose(value, v) and torch.allclose(dist.mean, p.dist(af).mean)
    assert torch.allclose(hid, out["human_node_rnn"].reshape(3, -1))


@check("4. Obstacle rays never change human-pipeline tensors (POINT_TOKENS, ENCODER)")
def _t4():
    for mode in (ObstacleMode.POINT_TOKENS, ObstacleMode.ENCODER):
        p, b = _policy(mode), _batch()
        seen: dict = {}
        p.base.spatial_attn.register_forward_pre_hook(
            lambda m, a: seen.setdefault("in", []).append([x.clone() for x in a])
        )
        p.base.spatial_attn.register_forward_hook(
            lambda m, a, o: seen.setdefault("hh", []).append(o.clone())
        )
        p.base.attn.register_forward_hook(
            lambda m, a, o: seen.setdefault("rh", []).append(o[0].clone())
        )
        empty = {**b, "ray_features": torch.zeros_like(b["ray_features"])}
        with torch.no_grad():
            m1 = _call(p, b)[0].mean
            m2 = _call(p, empty)[0].mean
        for i in range(2):
            assert all(torch.equal(x, y) for x, y in zip(seen["in"][0], seen["in"][1]))
        assert torch.equal(seen["hh"][0], seen["hh"][1]) and torch.equal(
            seen["rh"][0], seen["rh"][1]
        )
        assert not torch.allclose(m1, m2), f"{mode}: rays had no effect on output"


@check("5. Edge cases finite; per-env result independent of batch size")
def _t5():
    for mode in (ObstacleMode.NONE, ObstacleMode.POINT_TOKENS, ObstacleMode.ENCODER):
        p = _policy(mode)
        use = mode is not ObstacleMode.NONE
        with torch.no_grad():
            for nh, hits in ((0, 0), (0, 4), (MAX_N, NUM_RAYS)):
                d, v, h = _call(p, _batch(2, nh, hits), use_rays=use)
                assert torch.isfinite(d.mean).all() and torch.isfinite(v).all()
            b = _batch(3, 4, 6)
            full = _call(p, b, use_rays=use)[0].mean
            one = _call(p, {k: x[:1] for k, x in b.items()}, use_rays=use)[0].mean
        assert torch.allclose(full[:1], one, atol=1e-5), mode


@check("6. not_done_mask=0 resets hidden state")
def _t6():
    p, b = _policy(ObstacleMode.NONE), _batch(1)
    with torch.no_grad():
        a = p.act(
            b["robot"],
            b["neighbors"],
            b["neighbor_mask"],
            b["neighbor_history"],
            b["neighbor_history_mask"],
            torch.randn(1, 128),
            torch.zeros(1),
            deterministic=True,
        )[3]
        z = p.act(
            b["robot"],
            b["neighbors"],
            b["neighbor_mask"],
            b["neighbor_history"],
            b["neighbor_history_mask"],
            torch.zeros(1, 128),
            torch.ones(1),
            deterministic=True,
        )[3]
    assert torch.allclose(a, z, atol=1e-5)


@check(
    "7. Key/shape identity with the real official Policy (skipped if not importable)"
)
def _t7():
    try:
        from rl.networks.model import Policy as Official
    except ImportError:
        print("SKIP: official repo not importable")
        return
    cfg = CrowdNavPPPolicyConfig()
    off = Official(
        build_obs_space(cfg),
        build_action_space(cfg),
        base="selfAttn_merge_srnn",
        base_kwargs=build_base_args(cfg),
    )
    _policy(ObstacleMode.NONE).load_state_dict(off.state_dict())  # strict
    assert {k: v.shape for k, v in off.state_dict().items()} == {
        k: v.shape for k, v in _policy(ObstacleMode.NONE).state_dict().items()
    }


def main() -> int:
    failed = 0
    for name, fn in _CHECKS:
        print(f"--- {name}")
        try:
            fn()
        except Exception:
            failed += 1
            print("FAIL\n" + traceback.format_exc())
        else:
            print("PASS")
    print(f"SUMMARY: {len(_CHECKS) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
