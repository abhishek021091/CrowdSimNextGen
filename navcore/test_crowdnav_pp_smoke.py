"""Shape/gradient smoke test for the assembled CrowdNavPPPolicy.

Flagged as the immediate next step in policy.py's own module docstring:
every submodule (RobotStateEncoder, HumanHumanAttention, RobotHumanAttention,
RecurrentNodeUpdate, ActorCriticHeads, DiagGaussianHead) was shape-tested
individually, but the full forward()/act() path was never run end-to-end
against real CrowdSimEnv observation shapes. This is that missing check.

Deliberately not pytest-based, matching test_regression.py's harness
convention (no pytest dependency, runs anywhere the project runs).

Run directly:

    python -m navcore.test_crowdnav_pp_smoke
"""

from __future__ import annotations

import sys
import traceback

import torch

from navcore.gym_wrapper.crowd_sim_env import ActionMode, CrowdSimEnv, CrowdSimEnvConfig
from navcore.gym_wrapper.goal_reaching_task import GoalReachingTask
from navcore.policies.crowdnav_pp.policy import CrowdNavPPPolicy, CrowdNavPPPolicyConfig

_CHECKS: list[tuple[str, callable]] = []


def check(name: str):
    def decorator(fn):
        _CHECKS.append((name, fn))
        return fn

    return decorator


def _build_env_and_obs():
    config = CrowdSimEnvConfig(
        action_mode=ActionMode.VELOCITY, include_static_obstacles=False
    )
    env = CrowdSimEnv(GoalReachingTask(), config)
    obs, _ = env.reset(seed=0)
    return env, obs


def _to_batch(obs, nenv=1):
    return {
        k: torch.as_tensor(v, dtype=torch.float32)
        .unsqueeze(0)
        .repeat(nenv, *([1] * v.ndim))
        for k, v in obs.items()
    }


@check("1. forward() runs and returns correctly-shaped outputs")
def _test_forward_shapes():
    env, obs = _build_env_and_obs()
    policy_config = CrowdNavPPPolicyConfig(
        robot_feature_dim=obs["robot"].shape[-1],
        neighbor_feature_dim=obs["neighbors"].shape[-1],
    )
    policy = CrowdNavPPPolicy(policy_config)

    nenv = 3
    batch = _to_batch(obs, nenv=nenv)
    hidden = policy.initial_hidden_state(nenv=nenv)
    not_done = torch.ones(nenv)

    distribution, value, new_hidden = policy.forward(
        batch["robot"],
        batch["neighbors"],
        batch["neighbor_mask"],
        batch["neighbor_history"],
        batch["neighbor_history_mask"],
        hidden,
        not_done,
    )

    assert value.shape == (nenv, 1), f"value shape {value.shape}, expected {(nenv, 1)}"
    assert new_hidden.shape == (nenv, policy_config.rnn_hidden_size), (
        f"hidden shape {new_hidden.shape}"
    )
    sample = distribution.sample()
    assert sample.shape == (nenv, policy_config.action_dim), (
        f"action sample shape {sample.shape}"
    )
    env.close() if hasattr(env, "close") else None


@check("2. act() returns finite action/log_prob/value, and hidden state changes")
def _test_act_and_hidden_state_updates():
    env, obs = _build_env_and_obs()
    policy_config = CrowdNavPPPolicyConfig(
        robot_feature_dim=obs["robot"].shape[-1],
        neighbor_feature_dim=obs["neighbors"].shape[-1],
    )
    policy = CrowdNavPPPolicy(policy_config)
    policy.eval()

    batch = _to_batch(obs, nenv=1)
    hidden = policy.initial_hidden_state(nenv=1)
    not_done = torch.ones(1)

    with torch.no_grad():
        action, log_prob, value, new_hidden = policy.act(
            batch["robot"],
            batch["neighbors"],
            batch["neighbor_mask"],
            batch["neighbor_history"],
            batch["neighbor_history_mask"],
            hidden,
            not_done,
            deterministic=False,
        )

    assert torch.isfinite(action).all(), "action contains NaN/inf"
    assert torch.isfinite(log_prob).all(), "log_prob contains NaN/inf"
    assert torch.isfinite(value).all(), "value contains NaN/inf"
    assert not torch.allclose(new_hidden, hidden), (
        "hidden state did not change after one tick -- RecurrentNodeUpdate "
        "may not be wired correctly."
    )


@check("3. not_done_mask=0 actually resets the hidden state")
def _test_hidden_state_reset_on_done():
    env, obs = _build_env_and_obs()
    policy_config = CrowdNavPPPolicyConfig(
        robot_feature_dim=obs["robot"].shape[-1],
        neighbor_feature_dim=obs["neighbors"].shape[-1],
    )
    policy = CrowdNavPPPolicy(policy_config)
    policy.eval()

    batch = _to_batch(obs, nenv=1)
    nonzero_hidden = torch.randn(1, policy_config.rnn_hidden_size)

    with torch.no_grad():
        _, _, _, new_hidden = policy.act(
            batch["robot"],
            batch["neighbors"],
            batch["neighbor_mask"],
            batch["neighbor_history"],
            batch["neighbor_history_mask"],
            nonzero_hidden,
            torch.zeros(1),  # not_done_mask=0 -> reset
            deterministic=True,
        )

    with torch.no_grad():
        _, _, _, expected_hidden = policy.act(
            batch["robot"],
            batch["neighbors"],
            batch["neighbor_mask"],
            batch["neighbor_history"],
            batch["neighbor_history_mask"],
            torch.zeros(1, policy_config.rnn_hidden_size),
            torch.ones(1),
            deterministic=True,
        )

    assert torch.allclose(new_hidden, expected_hidden, atol=1e-5), (
        "not_done_mask=0 did not reset the incoming hidden state to zero "
        "before the GRU update."
    )


@check("4. Gradients flow into every submodule from a single backward pass")
def _test_gradients_reach_every_submodule():
    env, obs = _build_env_and_obs()
    policy_config = CrowdNavPPPolicyConfig(
        robot_feature_dim=obs["robot"].shape[-1],
        neighbor_feature_dim=obs["neighbors"].shape[-1],
    )
    policy = CrowdNavPPPolicy(policy_config)
    policy.train()

    nenv = 2
    batch = _to_batch(obs, nenv=nenv)

    # Force at least 2 real, non-degenerate visible neighbors per env,
    # rather than trusting whatever the random env reset happened to
    # produce. A single visible neighbor makes softmax attention
    # saturate to exactly 1.0 regardless of the query/key scores, which
    # zeroes the gradient w.r.t. every score-producing layer (the
    # query/key projections in both attention modules) no matter how
    # the network is wired -- not a bug, just a degenerate case this
    # test must not depend on pedestrian spawn luck to avoid. The same
    # applies to the temporal encoder's LSTM: if no real neighbor is
    # visible across the history window, its cell never sees a nonzero
    # input and its weights never get gradient either.
    max_neighbors = batch["neighbors"].shape[1]
    history_steps = batch["neighbor_history"].shape[1]
    n_forced = min(2, max_neighbors)

    batch["neighbors"][:, :n_forced, :] = torch.randn(
        nenv, n_forced, batch["neighbors"].shape[-1]
    )
    batch["neighbor_mask"][:, :n_forced] = 1
    batch["neighbor_mask"][:, n_forced:] = 0

    batch["neighbor_history"][:, :, :n_forced, :] = torch.randn(
        nenv, history_steps, n_forced, batch["neighbor_history"].shape[-1]
    )
    batch["neighbor_history_mask"][:, :, :n_forced] = 1
    batch["neighbor_history_mask"][:, :, n_forced:] = 0

    # A zeroed initial hidden state makes RecurrentNodeUpdate.gru_cell's
    # gradient w.r.t. weight_hh identically zero for a single tick --
    # GRUCell's gate equations multiply weight_hh by the *incoming*
    # hidden state, so zero in means zero gradient out, independent of
    # everything else. This is inherent to a one-step call from a fresh
    # state, not something forcing neighbors fixes. Seed a nonzero
    # hidden state instead, with not_done_mask=1 so RecurrentNodeUpdate
    # doesn't reset it back to zero before the update.
    hidden = torch.randn(nenv, policy_config.rnn_hidden_size)
    not_done = torch.ones(nenv)

    distribution, value, _ = policy.forward(
        batch["robot"],
        batch["neighbors"],
        batch["neighbor_mask"],
        batch["neighbor_history"],
        batch["neighbor_history_mask"],
        hidden,
        not_done,
    )

    # rsample(), not sample(): torch.distributions.Normal.sample() is
    # explicitly non-differentiable (drawn under no_grad internally), so
    # a loss built from it carries zero gradient back through the
    # action head regardless of correctness. rsample()'s reparameterized
    # draw (mean + std * noise, noise detached) is what actually needs
    # to be checked here.
    loss = distribution.rsample().pow(2).sum() + value.sum()
    loss.backward()

    dead = [
        name
        for name, p in policy.named_parameters()
        if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))
    ]
    assert not dead, f"No gradient reached: {dead}"


@check("5. Zero-visible-neighbors episode doesn't NaN (dummy-human substitution)")
def _test_zero_neighbors_no_nan():
    policy_config = CrowdNavPPPolicyConfig(robot_feature_dim=8, neighbor_feature_dim=5)
    policy = CrowdNavPPPolicy(policy_config)
    policy.eval()

    nenv = 1
    robot = torch.randn(nenv, 8)
    neighbors = torch.zeros(nenv, 10, 5)
    mask = torch.zeros(nenv, 10, dtype=torch.int8)  # nobody visible
    history = torch.zeros(nenv, 8, 10, 5)
    history_mask = torch.zeros(nenv, 8, 10, dtype=torch.int8)
    hidden = policy.initial_hidden_state(nenv=nenv)
    not_done = torch.ones(nenv)

    with torch.no_grad():
        distribution, value, _ = policy.forward(
            robot, neighbors, mask, history, history_mask, hidden, not_done
        )

    assert torch.isfinite(distribution.mean).all(), (
        "NaN in action mean with 0 neighbors"
    )
    assert torch.isfinite(value).all(), "NaN in value with 0 neighbors"

    @check(
        "6. Obstacle-encoder path: forward() runs, shapes correct, gradients reach ObstacleEncoder"
    )
    def _test_obstacle_encoder_integration():
        from navcore.policies.crowdnav_pp.obstacle_encoder import (
            ObstacleEncoder,
            ObstacleEncoderConfig,
            RAY_FEATURE_DIM,
        )

        env, obs = _build_env_and_obs()
        obstacle_encoder = ObstacleEncoder(ObstacleEncoderConfig(embedding_dim=64))
        policy_config = CrowdNavPPPolicyConfig(
            robot_feature_dim=obs["robot"].shape[-1],
            neighbor_feature_dim=obs["neighbors"].shape[-1],
            use_obstacle_encoder=True,
        )
        policy = CrowdNavPPPolicy(policy_config, obstacle_encoder=obstacle_encoder)
        policy.train()

        nenv, num_rays = 2, 60
        batch = _to_batch(obs, nenv=nenv)
        ray_features = torch.randn(nenv, num_rays, RAY_FEATURE_DIM)
        hidden = policy.initial_hidden_state(nenv=nenv)
        not_done = torch.ones(nenv)

        distribution, value, new_hidden = policy.forward(
            batch["robot"],
            batch["neighbors"],
            batch["neighbor_mask"],
            batch["neighbor_history"],
            batch["neighbor_history_mask"],
            hidden,
            not_done,
            ray_features=ray_features,
        )
        assert value.shape == (nenv, 1)

        loss = distribution.rsample().pow(2).sum() + value.sum()
        loss.backward()
        assert obstacle_encoder.net[0].weight.grad is not None
        assert torch.any(obstacle_encoder.net[0].weight.grad != 0)

    @check(
        "6. Obstacle-encoder path: forward() runs, shapes correct, gradients reach ObstacleEncoder"
    )
    def _test_obstacle_encoder_integration():
        from navcore.policies.crowdnav_pp.obstacle_encoder import (
            ObstacleEncoder,
            ObstacleEncoderConfig,
        )

        env, obs = _build_env_and_obs()
        obstacle_encoder = ObstacleEncoder(ObstacleEncoderConfig(embedding_dim=64))
        policy_config = CrowdNavPPPolicyConfig(
            robot_feature_dim=obs["robot"].shape[-1],
            neighbor_feature_dim=obs["neighbors"].shape[-1],
            use_obstacle_encoder=True,
        )
        policy = CrowdNavPPPolicy(policy_config, obstacle_encoder=obstacle_encoder)
        policy.train()

        nenv = 2
        batch = _to_batch(obs, nenv=nenv)  # now includes real "ray_features"
        hidden = policy.initial_hidden_state(nenv=nenv)
        not_done = torch.ones(nenv)

        distribution, value, new_hidden = policy.forward(
            batch["robot"],
            batch["neighbors"],
            batch["neighbor_mask"],
            batch["neighbor_history"],
            batch["neighbor_history_mask"],
            hidden,
            not_done,
            ray_features=batch["ray_features"],
        )
        assert value.shape == (nenv, 1), f"value shape {value.shape}"

        loss = distribution.rsample().pow(2).sum() + value.sum()
        loss.backward()

        assert obstacle_encoder.net[0].weight.grad is not None
        assert torch.any(obstacle_encoder.net[0].weight.grad != 0), (
            "No gradient reached ObstacleEncoder's first conv layer"
        )


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
