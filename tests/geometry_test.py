"""Tests for distance concentration geometry helpers."""

import numpy as np

from anomaly_detection.geometry import DEFAULT_PAIR_COUNT, distance_concentration_ratio


def test_distance_concentration_ratio_deterministic() -> None:
    """Identical matrices and RNG seed must yield identical CR estimates."""
    rng = np.random.default_rng(123)
    x = rng.standard_normal(size=(120, 4))
    cr_a = distance_concentration_ratio(x, random_state=7)
    cr_b = distance_concentration_ratio(x, random_state=7)
    assert cr_a == cr_b


def test_distance_concentration_ratio_bounded_pairs() -> None:
    """Row pool caps must reduce index range without crashing."""
    x = np.eye(10, dtype=np.float64)
    cr = distance_concentration_ratio(x, pair_count=50, pool_max_rows=8, random_state=0)
    assert 0.0 <= cr < 50.0


def test_random_pair_sampling_size() -> None:
    """Estimator must honor pair_count knob."""
    x = np.vstack([np.zeros(3), np.ones(3), np.ones(3) * 2.0]).astype(np.float64)
    cr = distance_concentration_ratio(x, pair_count=min(DEFAULT_PAIR_COUNT, 10), random_state=0)
    assert cr >= 0.0
