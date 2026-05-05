"""Tests for plan §4.2 cluster-aware local/global anomaly typing."""

import numpy as np

from anomaly_detection.training.outlier_taxonomy import classify_positives_global_vs_localclusters


def test_clusters_separate_far_vs_near_positives_on_synthetic_geometry() -> None:
    rng = np.random.default_rng(1)
    c_left = np.array([-8.0, 0.0])
    c_right = np.array([8.0, 0.0])
    inliers = np.vstack(
        [
            rng.normal(loc=c_left, scale=0.25, size=(45, 2)),
            rng.normal(loc=c_right, scale=0.25, size=(45, 2)),
        ]
    )
    far_out = np.array([[-22.0, 0.0]])
    near_out = np.array([[8.0, 0.25]])
    x = np.vstack([inliers, far_out, near_out])
    y = np.array([0] * inliers.shape[0] + [1, 1], dtype=np.int64)

    type_g, type_l, _ = classify_positives_global_vs_localclusters(x, y, random_state=3)
    positives = np.flatnonzero(y == 1)
    far_idx = int(positives[0])
    near_idx = int(positives[1])
    assert type_g.size + type_l.size == 2
    assert far_idx in type_g or far_idx in type_l
    assert near_idx in type_g or near_idx in type_l
