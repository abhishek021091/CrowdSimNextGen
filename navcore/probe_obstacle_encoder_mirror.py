"""Diagnostic probe: does ObstacleEncoder's final average-pool destroy
left/right directional information about nearby obstacles?

Drop this file anywhere importable from your repo root (e.g.
navcore/diagnostics/probe_obstacle_encoder_mirror.py) and run:

    python -m navcore.diagnostics.probe_obstacle_encoder_mirror

or just `python probe_obstacle_encoder_mirror.py` from the repo root if
navcore is on your PYTHONPATH.

Method
------
For several wall configurations, we build the *exact* mirror-image scan
analytically (reflecting each ray's angle and hit point across the
y-axis -- this requires re-indexing rays, not just negating x at a fixed
index, since a reflection maps ray i to a *different* ray j; see
mirror_scan() below), run both the original and mirrored scan through
ObstacleEncoder (random-init, untrained -- we're testing the
architecture's structural capacity, not learned weights), and check:

1. Cosine similarity between paired (original, mirror) embeddings.
   Close to 1.0 means the two configurations look nearly identical to
   the encoder after pooling.
2. Whether a linear probe (logistic regression) can tell "wall was on
   the left" from "wall was on the right" from the embedding alone.
   Chance (~0.5 accuracy) means no separable directional signal
   survives pooling.

A sanity check first verifies the analytic mirror construction against
an actual re-cast of a physically mirrored obstacle, so the rest of the
result isn't resting on unverified index arithmetic.
"""

from __future__ import annotations

import numpy as np
import torch
from shapely.geometry import LineString

from navcore.entities.components.sensors.obstacle_detector import (
    ObstacleDetector,
    ObstacleDetectorConfig,
    ObstacleScan,
    scan_to_features,
)
from navcore.policies.crowdnav_pp.obstacle_encoder import (
    ObstacleEncoder,
    ObstacleEncoderConfig,
)

SEED = 0


def mirror_scan(scan: ObstacleScan, num_rays: int) -> ObstacleScan:
    """Return the exact scan of the scene reflected across the y-axis.

    Rays are cast at angle theta_i = 2*pi*i/num_rays (full-circle fan,
    see ObstacleDetector._build_ray_offsets). Reflecting the *scene*
    across the y-axis (x -> -x) turns a hit originally seen by the ray
    at angle theta into a hit seen by the ray at angle (pi - theta) --
    a different ray index, not the same index with x negated. For
    evenly spaced angles this index mapping is exactly
    j = (num_rays // 2 - i) mod num_rays (requires num_rays even).
    """
    if num_rays % 2 != 0:
        raise ValueError("mirror_scan requires an even num_rays.")

    mirrored_hit_mask = np.zeros_like(scan.hit_mask)
    mirrored_hit_type = np.zeros_like(scan.hit_type)
    mirrored_relative_positions = np.zeros_like(scan.relative_positions)

    half = num_rays // 2
    for i in range(num_rays):
        j = (half - i) % num_rays
        mirrored_hit_mask[j] = scan.hit_mask[i]
        mirrored_hit_type[j] = scan.hit_type[i]
        dx, dy = scan.relative_positions[i]
        mirrored_relative_positions[j] = (-dx, dy)

    return ObstacleScan(
        hit_mask=mirrored_hit_mask,
        hit_type=mirrored_hit_type,
        relative_positions=mirrored_relative_positions,
    )


def build_wall_scans(
    detector: ObstacleDetector, offsets_and_extents: list[tuple[float, float, float]]
) -> list[tuple[ObstacleScan, ObstacleScan]]:
    """For each (x_offset, y_min, y_max), cast a scan against a vertical
    wall segment at x=x_offset and return (original, analytic_mirror).
    """
    pairs = []
    for x_offset, y_min, y_max in offsets_and_extents:
        wall = LineString([(x_offset, y_min), (x_offset, y_max)])
        scan = detector.sense(0.0, 0.0, [wall])
        mirrored = mirror_scan(scan, detector.config.num_rays)
        pairs.append((scan, mirrored))
    return pairs


def sanity_check_mirror_math(detector: ObstacleDetector) -> None:
    """Cross-check the analytic mirror against an actual re-cast against
    a physically mirrored wall, for one configuration.
    """
    left_wall = LineString([(-2.0, -1.5), (-2.0, 1.5)])
    right_wall = LineString([(2.0, -1.5), (2.0, 1.5)])

    left_scan = detector.sense(0.0, 0.0, [left_wall])
    right_scan = detector.sense(0.0, 0.0, [right_wall])
    analytic_mirror = mirror_scan(left_scan, detector.config.num_rays)

    mask_matches = bool(np.array_equal(analytic_mirror.hit_mask, right_scan.hit_mask))
    pos_max_err = float(
        np.max(
            np.abs(analytic_mirror.relative_positions - right_scan.relative_positions)
        )
    )
    print(f"[sanity check] hit_mask matches real mirrored cast: {mask_matches}")
    print(f"[sanity check] max abs error in relative_positions: {pos_max_err:.6f}")
    if not mask_matches or pos_max_err > 1e-6:
        print(
            "[sanity check] WARNING: analytic mirror does not exactly match a "
            "real re-cast -- treat downstream results with caution."
        )


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    detector_config = ObstacleDetectorConfig(num_rays=60, max_range=5.0)
    detector = ObstacleDetector(detector_config)

    sanity_check_mirror_math(detector)

    # Many distinct wall configurations, all on the left (negative x);
    # mirror_scan gives us the matching right-side scan for each. Swept
    # over a grid rather than a handful of points: the linear probe
    # below needs n comfortably larger than the 256-dim embedding to
    # say anything trustworthy about separability, not just about
    # overfitting capacity.
    configs = [
        (x_offset, y_center - half_extent, y_center + half_extent)
        for x_offset in (-0.8, -1.0, -1.3, -1.6, -2.0, -2.4, -2.8, -3.2, -3.6, -4.0)
        for y_center, half_extent in ((0.0, 1.0), (1.2, 0.6), (-1.2, 0.6))
    ]
    pairs = build_wall_scans(detector, configs)

    # embedding_dim=256 matches policy.py's default when the obstacle
    # encoder is auto-built (interaction_embedding_dim=256).
    encoder = ObstacleEncoder(ObstacleEncoderConfig(embedding_dim=256))
    encoder.eval()

    left_embeddings = []
    right_embeddings = []
    cosine_sims = []

    with torch.no_grad():
        for left_scan, right_scan in pairs:
            left_features = scan_to_features(left_scan, detector_config.max_range)
            right_features = scan_to_features(right_scan, detector_config.max_range)

            left_tensor = torch.as_tensor(left_features, dtype=torch.float32).unsqueeze(
                0
            )
            right_tensor = torch.as_tensor(
                right_features, dtype=torch.float32
            ).unsqueeze(0)

            left_emb = encoder(left_tensor).squeeze(0)
            right_emb = encoder(right_tensor).squeeze(0)

            left_embeddings.append(left_emb.numpy())
            right_embeddings.append(right_emb.numpy())

            cos_sim = torch.nn.functional.cosine_similarity(
                left_emb.unsqueeze(0), right_emb.unsqueeze(0)
            ).item()
            cosine_sims.append(cos_sim)

    print(
        f"\n[pairwise cosine similarity, original vs. mirrored embedding, n={len(pairs)}]"
    )
    print(
        f"  mean cos_sim = {np.mean(cosine_sims):.4f}  (range {min(cosine_sims):.4f}-{max(cosine_sims):.4f})"
    )
    print(
        "  Caveat: ObstacleEncoder.net ends in Linear->ReLU, so every "
        "embedding lives in the non-negative orthant. Two random "
        "non-negative high-dim vectors already have inflated cosine "
        "similarity by construction, independent of shared information "
        "content -- treat this number as suggestive, not conclusive. "
        "The linear probe below is the more trustworthy signal."
    )

    # Linear probe: can logistic regression tell "wall on the left"
    # (label 0) from "wall on the right" (label 1) from the embedding
    # alone? With d=256 and n in the dozens, an under-regularized probe
    # can memorize any label split -- that would look like "information
    # survives pooling" even when it doesn't. Two guards against that:
    # strong L2 regularization (small C), and a shuffled-label
    # permutation baseline computed with the exact same procedure, so
    # the real probe's accuracy is judged against this test's own
    # empirical chance level rather than a naive 0.5.
    X = np.stack(left_embeddings + right_embeddings)
    y = np.array([0] * len(left_embeddings) + [1] * len(right_embeddings))

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        clf = LogisticRegression(max_iter=5000, C=0.1)
        real_scores = cross_val_score(clf, X, y, cv=cv)
        real_acc = real_scores.mean()

        rng = np.random.default_rng(SEED)
        n_permutations = 20
        shuffled_accs = []
        for _ in range(n_permutations):
            y_shuffled = rng.permutation(y)
            scores = cross_val_score(clf, X, y_shuffled, cv=cv)
            shuffled_accs.append(scores.mean())
        shuffled_accs = np.array(shuffled_accs)

        print(f"\n[linear probe] n={len(y)}, dim={X.shape[1]}, C=0.1, 5-fold CV")
        print(f"  real-label accuracy:     {real_acc:.3f}")
        print(
            f"  shuffled-label baseline: {shuffled_accs.mean():.3f} "
            f"+/- {shuffled_accs.std():.3f}  (this test's own empirical chance level)"
        )
        margin = real_acc - shuffled_accs.mean()
        print(f"  margin over baseline:    {margin:+.3f}")
    except ImportError:
        print("\n[linear probe] scikit-learn not installed -- skipping probe.")
        print("  pip install scikit-learn --break-system-packages to enable it.")

    print(
        "\nInterpretation: high mean cos_sim (expected, see caveat above) plus "
        "a real-label probe accuracy that sits within ~1 std of the shuffled "
        "baseline means the average-pool is destroying left/right directional "
        "information -- confirms the hypothesis, worth the per-segment-token "
        "fix. A real-label accuracy clearly above the shuffled baseline means "
        "some directional signal survives pooling even in this untrained "
        "encoder, which weakens the case for redesigning ObstacleEncoder "
        "before checking whether the training run's failure is actually "
        "coming from somewhere else (e.g. RobotHumanAttention not learning "
        "to use what's there)."
    )


if __name__ == "__main__":
    main()
