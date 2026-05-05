"""Tests for ODDS isolation-decile recall proxy."""

import numpy as np

from anomaly_detection.training.isolation_bins import odds_isolation_decile_metrics


def test_isolation_deciles_emit_ten_recall_metrics() -> None:
    """Synthetic blob inliers and offset outliers produce decile bucket metrics."""
    rng = np.random.default_rng(0)
    x_in = rng.normal(scale=0.2, size=(60, 2))
    x_out = rng.normal(loc=3.5, scale=0.1, size=(12, 2))
    x = np.vstack([x_in, x_out])
    y = np.array([0] * 60 + [1] * 12, dtype=np.int64)
    scores = np.linalg.norm(x - x.mean(axis=0), axis=1)
    blob = odds_isolation_decile_metrics(x, y, scores, contamination=float(np.mean(y)))
    assert any(k.startswith("idb_recall_d") for k in blob)
    assert len([k for k in blob if k.startswith("idb_recall_d")]) == 10


def test_isolation_deciles_empty_when_no_outliers() -> None:
    x = np.ones((10, 3), dtype=np.float64)
    y = np.zeros(10, dtype=np.int64)
    scores = np.arange(10.0, dtype=np.float64)
    assert odds_isolation_decile_metrics(x, y, scores, contamination=0.05) == {}
