"""Comprehensive verification test for CrowdNav++ original policy and adapter."""

import sys
import traceback
import types
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from navcore.policies.original.policy import Policy as OriginalPolicy
from navcore.policies.original.adapter import (
    CrowdNavPPPolicy,
    CrowdNavPPPolicyConfig,
    ObstacleMode,
    ObstacleTokenEncoder,
)
from navcore.test_crowdnav_pp_smoke import main as smoke_main
from navcore.test_obstacle_tokens_smoke import main as obstacle_smoke_main


def test_1_original_checkpoint_loading():
    print("Test 1: Official CrowdNav++ Checkpoint Loading...")
    algo_args = types.SimpleNamespace(
        human_node_rnn_size=128,
        human_human_edge_rnn_size=256,
        human_node_output_size=256,
        human_node_input_size=3,
        human_human_edge_input_size=2,
        human_node_embedding_size=64,
        human_human_edge_embedding_size=64,
        attention_size=64,
        seq_length=30,
        num_processes=1,
        num_mini_batch=1,
        use_self_attn=True,
        use_hr_attn=True,
        env_name="CrowdSimPredRealGST-v0",
        sort_humans=True,
        no_cuda=True,
    )
    obs_space_dict = {
        "robot_node": spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7)),
        "temporal_edges": spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2)),
        "spatial_edges": spaces.Box(low=-np.inf, high=np.inf, shape=(10, 12)),
        "visible_masks": spaces.Box(low=-np.inf, high=np.inf, shape=(10,)),
        "detected_human_num": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
    }
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    policy = OriginalPolicy(
        obs_space_dict, action_space, base="selfAttn_merge_srnn", base_kwargs=algo_args
    )

    ckpt_path = "/media/abhishek/New Volume/workspace/abhishek0209/AutoSweepCrowdNav/trained_models/my_model/checkpoints/00000.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"  Checkpoint has {len(ckpt)} tensors.")

    # Strict loading into original policy
    incompatible = policy.load_state_dict(ckpt, strict=True)
    assert len(incompatible.missing_keys) == 0, (
        f"Missing keys: {incompatible.missing_keys}"
    )
    assert len(incompatible.unexpected_keys) == 0, (
        f"Unexpected keys: {incompatible.unexpected_keys}"
    )
    print(
        "  PASSED: 100% strict match on original Policy (0 missing, 0 unexpected keys)."
    )

    # Strict base/dist loading into adapter CrowdNavPPPolicy
    adapter_policy = CrowdNavPPPolicy(
        CrowdNavPPPolicyConfig(obstacle_mode=ObstacleMode.POINT_TOKENS)
    )
    incompatible_adapter = adapter_policy.load_state_dict(ckpt)
    # Obstacle params are newly introduced and expected to be missing in original checkpoint
    for k in incompatible_adapter.missing_keys:
        assert k.startswith("obstacle_"), f"Unexpected missing key: {k}"
    assert len(incompatible_adapter.unexpected_keys) == 0, (
        f"Unexpected keys: {incompatible_adapter.unexpected_keys}"
    )
    print("  PASSED: Checkpoint loads cleanly into CrowdNavPPPolicy adapter.")


def test_2_architectural_invariance_and_obstacle_decoupling():
    print("Test 2: Architecture verification (Obstacles separate, humans untouched)...")
    policy = CrowdNavPPPolicy(
        CrowdNavPPPolicyConfig(obstacle_mode=ObstacleMode.POINT_TOKENS)
    )
    policy.eval()

    # Batch with 2 humans, 4 rays
    B = 2
    N = 10
    R = 20
    robot = torch.randn(B, 8)
    neighbors = torch.randn(B, N, 5)
    neighbor_mask = torch.zeros(B, N)
    neighbor_mask[:, :3] = 1.0
    history = torch.randn(B, 8, N, 5)
    history_mask = torch.zeros(B, 8, N)
    history_mask[:, :, :3] = 1.0
    hidden = policy.initial_hidden_state(B)
    not_done = torch.ones(B)

    # Rays: 4 valid hits
    rays = torch.zeros(B, R, 3)
    rays[:, :4, 0] = 1.0
    rays[:, :4, 1:] = torch.randn(B, 4, 2)

    with torch.no_grad():
        dist_with_obs, val_with_obs, h_with_obs = policy.forward(
            robot,
            neighbors,
            neighbor_mask,
            history,
            history_mask,
            hidden,
            not_done,
            ray_features=rays,
        )

        # Zero-out rays
        rays_zero = torch.zeros_like(rays)
        dist_no_obs, val_no_obs, h_no_obs = policy.forward(
            robot,
            neighbors,
            neighbor_mask,
            history,
            history_mask,
            hidden,
            not_done,
            ray_features=rays_zero,
        )

    # Obstacles must influence output
    assert not torch.allclose(dist_with_obs.mean, dist_no_obs.mean, atol=1e-5), (
        "Obstacles did not affect the output!"
    )
    print("  PASSED: Obstacle hits successfully influence policy output.")

    # Permuting rays must yield identical output
    perm = torch.randperm(R)
    rays_perm = rays[:, perm]
    with torch.no_grad():
        dist_perm, _, _ = policy.forward(
            robot,
            neighbors,
            neighbor_mask,
            history,
            history_mask,
            hidden,
            not_done,
            ray_features=rays_perm,
        )
    assert torch.allclose(dist_with_obs.mean, dist_perm.mean, atol=1e-5), (
        "Permuting rays changed output -- obstacle aggregation is not permutation invariant!"
    )
    print("  PASSED: Obstacle token aggregation is strictly permutation-invariant.")


def main():
    print("=" * 70)
    print("Running CrowdNav++ Original & Adapter Test Suite")
    print("=" * 70)
    try:
        test_1_original_checkpoint_loading()
        test_2_architectural_invariance_and_obstacle_decoupling()
    except Exception:
        print("FAIL in custom tests:")
        traceback.print_exc()
        return 1

    print("\nRunning smoke tests:")
    s1 = smoke_main()
    s2 = obstacle_smoke_main()
    if s1 != 0 or s2 != 0:
        print("Smoke tests failed!")
        return 1

    print("\nALL TESTS PASSED SUCCESSFULLY!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
