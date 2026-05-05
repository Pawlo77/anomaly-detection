"""Tests for sensitivity analysis utilities."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_detection.sensitivity import (
    BootstrapStabilityReport,
    CandidateStability,
    HyperparameterSpec,
    SensitivityError,
    SobolSpace,
    bootstrap_pr_auc_stability,
    evaluate_sobol_objective,
    select_stable_best,
)


def test_bootstrap_stability_returns_deterministic_report_shape() -> None:
    """Bootstrap engine returns bounded PR-AUC summary with expected lengths."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 3))
    y = np.zeros(80, dtype=np.int64)
    y[:16] = 1
    rng.shuffle(y)

    def linear_score_fn(sample_x: np.ndarray) -> np.ndarray:
        return sample_x[:, 0] + 0.25 * sample_x[:, 1]

    report = bootstrap_pr_auc_stability(
        x=x,
        y_true=y,
        score_fn=linear_score_fn,
        n_resamples=8,
        sample_fraction=0.9,
        random_state=42,
    )
    assert isinstance(report, BootstrapStabilityReport)
    assert report.n_resamples == 8
    assert len(report.pr_auc_values) == 8
    assert 0.0 <= report.mean_pr_auc <= 1.0
    assert 0.0 <= report.std_pr_auc <= 1.0


def test_select_stable_best_prefers_highest_mean_under_constraint() -> None:
    """Selection filters unstable candidates then ranks by mean PR-AUC."""
    candidates = [
        CandidateStability(params={"n_neighbors": 10}, mean_pr_auc=0.81, std_pr_auc=0.028),
        CandidateStability(params={"n_neighbors": 20}, mean_pr_auc=0.84, std_pr_auc=0.035),
        CandidateStability(params={"n_neighbors": 30}, mean_pr_auc=0.83, std_pr_auc=0.020),
    ]
    best = select_stable_best(candidates)
    assert best.params["n_neighbors"] == 30
    assert best.mean_pr_auc == pytest.approx(0.83)


def test_select_stable_best_raises_when_no_candidate_is_stable() -> None:
    """Selection fails fast when all candidates violate stability bound."""
    candidates = [
        CandidateStability(params={"gamma": 0.1}, mean_pr_auc=0.90, std_pr_auc=0.10),
        CandidateStability(params={"gamma": 1.0}, mean_pr_auc=0.91, std_pr_auc=0.07),
    ]
    with pytest.raises(SensitivityError, match="No candidate satisfies stability constraint"):
        _ = select_stable_best(candidates)


def test_evaluate_sobol_objective_decodes_typed_hyperparameters() -> None:
    """Sobol objective evaluation decodes int/categorical/float values."""
    space = SobolSpace(
        parameters=(
            HyperparameterSpec(name="n_estimators", kind="int", lower=50, upper=500),
            HyperparameterSpec(name="kernel", kind="categorical", choices=("rbf", "linear")),
            HyperparameterSpec(name="nu", kind="float", lower=0.01, upper=0.5),
        )
    )
    sampled_values = np.array(
        [
            [100.2, 0.0, 0.10],
            [350.9, 1.0, 0.35],
        ],
        dtype=np.float64,
    )

    seen: list[dict[str, float | int | str]] = []

    def objective(params: dict[str, float | int | str]) -> float:
        seen.append(params)
        return float(params["nu"])  # deterministic scalar objective

    values = evaluate_sobol_objective(space, sampled_values, objective)
    assert values.shape == (2,)
    assert seen[0]["n_estimators"] == 100
    assert seen[0]["kernel"] == "rbf"
    assert seen[1]["kernel"] == "linear"
