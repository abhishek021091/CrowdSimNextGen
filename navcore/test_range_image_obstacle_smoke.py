"""Shape/gradient smoke test for the new range-image obstacle branch.

Deliberately not pytest-based, matching test_crowdnav_pp_smoke.py's and
test_regression.py's harness convention (no pytest dependency, runs
anywhere the project runs).

Run directly:

    python -m navcore.test_range_image_obstacle_smoke
"""

from __future__ import annotations

import inspect
import sys
import traceback

import torch

from navcore.entities.components.sensors.obstacle_detector import (
    ObstacleDetector,
    ObstacleDetectorConfig,
)
from navcore.entities.components.sensors.range_image import (
    RangeImageBuilder,
    RangeImageBuilderConfig,
)
from navcore.policies.crowdnav_pp.fusion_gate import ContextFusionGate, FusionGateConfig
from navcore.policies.crowdnav_pp.policy import CrowdNavPPPolicy, CrowdNavPPPolicyConfig
from navcore.policies.crowdnav_pp.range_image_encoder import (
    RangeImageEncoder,
    RangeImageEncoderConfig,
)
from navcore.policies.crowdnav_pp.robot_human_attention import RobotHumanAttention
from navcore.policies.crowdnav_pp.robot_obstacle_attention import (
    RobotObstacleAttention,
    RobotObstacleAttentionConfig,
)

_CHECKS: list[tuple[str, callable]] = []


def check(name: str):
    def decorator(fn):
        _CHECKS.append((name, fn))
        return fn

    return decorator


_RANGE_IMAGE_CONFIG = RangeImageBuilderConfig(
    num_rays=180, num_range_bins=128, max_range=5.0
)
_ENCODER_CONFIG = RangeImageEncoderConfig(
    in_height=128,
    in_width=180,
    stem_channels=32,
    stage_channels=(32, 64, 128),
    blocks_per_stage=(2, 2, 2),
    downsample_after_stage=(True, True, False),
    token_embedding_dim=128,
    num_obstacle_tokens=15,
)


def _policy_config(use_range_image_obstacles: bool) -> CrowdNavPPPolicyConfig:
    return CrowdNavPPPolicyConfig(
        robot_feature_dim=8,
        neighbor_feature_dim=5,
        use_range_image_obstacles=use_range_image_obstacles,
        range_image_encoder=_ENCODER_CONFIG,
        robot_obstacle_attention=RobotObstacleAttentionConfig(
            robot_embedding_dim=256, obstacle_embedding_dim=128, num_attention_heads=4
        ),
        fusion_gate=FusionGateConfig(embedding_dim=256, hidden_size=128),
    )


def _dummy_batch(nenv: int, max_neighbors: int = 10, history_steps: int = 8):
    robot = torch.randn(nenv, 8)
    neighbors = torch.randn(nenv, max_neighbors, 5)
    neighbor_mask = torch.ones(nenv, max_neighbors, dtype=torch.int8)
    history = torch.randn(nenv, history_steps, max_neighbors, 5)
    history_mask = torch.ones(nenv, history_steps, max_neighbors, dtype=torch.int8)
    range_image = torch.randint(0, 2, (nenv, 1, 128, 180)).float()
    return robot, neighbors, neighbor_mask, history, history_mask, range_image


@check("1. RangeImageBuilder produces a well-formed binary image")
def _test_range_image_builder():
    detector = ObstacleDetector(ObstacleDetectorConfig(num_rays=180, max_range=5.0))
    builder = RangeImageBuilder(_RANGE_IMAGE_CONFIG)

    # No obstacles at all -> every ray effectively "hits" the sensing
    # boundary at max_range (ObstacleDetector's own no-hit default).
    scan = detector.sense(0.0, 0.0, obstacles=[])
    image = builder.build(scan)
    assert image.shape == (1, 128, 180), f"unexpected shape {image.shape}"
    assert image.dtype == __import__("numpy").float32
    assert set(image.reshape(-1).tolist()) <= {0.0, 1.0}, "image must be binary"
    # Only the far edge (row 0) should be blocked; everything else free.
    assert (image[0, 0, :] == 0.0).all(), "far edge (max range) must be blocked"
    assert (image[0, 1:, :] == 1.0).all(), (
        "everything closer than max range must be free"
    )


@check("2. RangeImageEncoder forward pass produces correctly-shaped tokens")
def _test_range_image_encoder_forward():
    encoder = RangeImageEncoder(_ENCODER_CONFIG)
    nenv = 3
    range_image = torch.randint(0, 2, (nenv, 1, 128, 180)).float()

    tokens = encoder(range_image)
    assert tokens.shape == (nenv, 15, 128), f"unexpected token shape {tokens.shape}"
    assert torch.isfinite(tokens).all()


@check("3. RobotObstacleAttention forward pass and shape validation")
def _test_robot_obstacle_attention():
    attn = RobotObstacleAttention(
        RobotObstacleAttentionConfig(
            robot_embedding_dim=256, obstacle_embedding_dim=128, num_attention_heads=4
        )
    )
    nenv = 4
    robot_embedding = torch.randn(nenv, 256)
    obstacle_tokens = torch.randn(nenv, 15, 128)

    context = attn(robot_embedding, obstacle_tokens)
    assert context.shape == (nenv, 128)

    try:
        attn(torch.randn(nenv, 99), obstacle_tokens)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on robot_embedding width mismatch")


@check("4. ContextFusionGate keeps gate values within [0, 1]")
def _test_fusion_gate_range():
    gate_module = ContextFusionGate(FusionGateConfig(embedding_dim=256, hidden_size=64))
    nenv = 8
    human = torch.randn(nenv, 256) * 10.0  # large magnitudes to stress the sigmoid
    obstacle = torch.randn(nenv, 256) * 10.0

    fused = gate_module(human, obstacle)
    assert fused.shape == (nenv, 256)
    assert gate_module.last_gate is not None
    assert torch.all(gate_module.last_gate >= 0.0) and torch.all(
        gate_module.last_gate <= 1.0
    )
    assert gate_module.last_gate.shape == (nenv, 256), (
        "gate must be feature-wise, not scalar"
    )


@check("5. RobotHumanAttention no longer accepts obstacle_embedding/obstacle_mask")
def _test_robot_human_attention_signature():
    params = inspect.signature(RobotHumanAttention.forward).parameters
    assert "obstacle_embedding" not in params
    assert "obstacle_mask" not in params
    assert list(params.keys()) == [
        "self",
        "robot_embedding",
        "human_embeddings",
        "visible_mask",
    ]


@check("6. CrowdNavPPPolicy forward pass with obstacle branch enabled")
def _test_policy_forward_with_obstacles():
    config = _policy_config(use_range_image_obstacles=True)
    policy = CrowdNavPPPolicy(config)
    nenv = 2
    robot, neighbors, neighbor_mask, history, history_mask, range_image = _dummy_batch(
        nenv
    )
    hidden = policy.initial_hidden_state(nenv=nenv)
    not_done = torch.ones(nenv)

    distribution, value, new_hidden = policy.forward(
        robot,
        neighbors,
        neighbor_mask,
        history,
        history_mask,
        hidden,
        not_done,
        range_image=range_image,
    )
    assert value.shape == (nenv, 1)
    assert new_hidden.shape == (nenv, config.rnn_hidden_size)
    assert distribution.sample().shape == (nenv, config.action_dim)


@check(
    "7. CrowdNavPPPolicy rejects range_image when the branch is disabled, and vice versa"
)
def _test_policy_obstacle_flag_consistency():
    config_off = _policy_config(use_range_image_obstacles=False)
    policy_off = CrowdNavPPPolicy(config_off)
    nenv = 2
    robot, neighbors, neighbor_mask, history, history_mask, range_image = _dummy_batch(
        nenv
    )
    hidden = policy_off.initial_hidden_state(nenv=nenv)
    not_done = torch.ones(nenv)

    # Branch disabled + no range_image -> fine.
    policy_off.forward(
        robot, neighbors, neighbor_mask, history, history_mask, hidden, not_done
    )

    # Branch disabled + range_image given -> must raise.
    try:
        policy_off.forward(
            robot,
            neighbors,
            neighbor_mask,
            history,
            history_mask,
            hidden,
            not_done,
            range_image=range_image,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError: range_image given, branch disabled")

    # Branch enabled + no range_image -> must raise.
    config_on = _policy_config(use_range_image_obstacles=True)
    policy_on = CrowdNavPPPolicy(config_on)
    try:
        policy_on.forward(
            robot, neighbors, neighbor_mask, history, history_mask, hidden, not_done
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError: branch enabled, range_image missing")


@check("8. Gradients reach every obstacle-branch submodule from one backward pass")
def _test_gradients_reach_obstacle_branch():
    config = _policy_config(use_range_image_obstacles=True)
    policy = CrowdNavPPPolicy(config)
    policy.train()
    nenv = 2
    robot, neighbors, neighbor_mask, history, history_mask, range_image = _dummy_batch(
        nenv
    )
    hidden = torch.randn(nenv, config.rnn_hidden_size)  # nonzero, see smoke-test #4's
    # sibling check in test_crowdnav_pp_smoke.py for why a zero initial
    # hidden state would zero out the GRU's weight_hh gradient regardless
    # of correctness elsewhere.
    not_done = torch.ones(nenv)

    distribution, value, _ = policy.forward(
        robot,
        neighbors,
        neighbor_mask,
        history,
        history_mask,
        hidden,
        not_done,
        range_image=range_image,
    )
    loss = distribution.rsample().pow(2).sum() + value.sum()
    loss.backward()

    obstacle_modules = {
        "range_image_encoder": policy.range_image_encoder,
        "robot_obstacle_attention": policy.robot_obstacle_attention,
        "obstacle_context_proj": policy.obstacle_context_proj,
        "fusion_gate": policy.fusion_gate,
        "robot_encoder": policy.robot_encoder,
        "human_human_attention": policy.human_human_attention,
        "robot_human_attention": policy.robot_human_attention,
        "recurrent_update": policy.recurrent_update,
    }
    dead: list[str] = []
    for name, module in obstacle_modules.items():
        assert module is not None, f"{name} was not constructed"
        for param_name, p in module.named_parameters():
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0)):
                dead.append(f"{name}.{param_name}")

    assert not dead, f"No gradient reached: {dead}"


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
