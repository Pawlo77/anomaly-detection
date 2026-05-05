"""Scheduled phase-four sensitivity workloads (Saltelli Sobol + bootstrap guards)."""

import itertools as it
import numbers
from time import perf_counter
from typing import Any

import numpy as np

from ..datasets.loader import DatasetLoader
from ..geometry import distance_concentration_ratio
from ..metrics import MetricThresholdConfig, evaluate_metrics, metric_pr_auc
from ..models import IForestModel, OCSVMModel
from ..models.params import IForestParams, OCSVMParams
from ..sensitivity import (
    CandidateStability,
    HyperparameterSpec,
    SensitivityError,
    SobolSpace,
    bootstrap_pr_auc_stability,
    evaluate_sobol_objective,
    select_stable_best,
    sobol_analyze,
    sobol_sample,
)
from .isolation_bins import odds_isolation_decile_metrics
from .lhs_ocsvm import lhs_dict_rows
from .runner import TaskResult, _load_phase4_bundle
from .tasks import ExperimentTask

IFOREST_SOBOL_SPACE = SobolSpace(
    parameters=(
        HyperparameterSpec(name="n_estimators", kind="int", lower=50.0, upper=500.0),
        HyperparameterSpec(name="max_samples", kind="int", lower=64.0, upper=512.0),
        HyperparameterSpec(name="contamination", kind="float", lower=0.01, upper=0.35),
        HyperparameterSpec(name="max_features", kind="float", lower=0.5, upper=1.0),
    )
)
"""Sobol space for IForest hyperparameters in phase 4 sensitivity study."""

OCSVM_SOBOL_SPACE = SobolSpace(
    parameters=(
        HyperparameterSpec(name="nu", kind="float", lower=0.01, upper=0.5),
        HyperparameterSpec(name="log10_gamma", kind="float", lower=-4.0, upper=1.0),
    )
)
"""Sobol space for OCSVM hyperparameters in phase 4 sensitivity study."""


def _iforest_fit_scores(
    x: np.ndarray,
    random_state: int,
    forest_kw: dict[str, Any],
) -> np.ndarray:
    """Fit ``IsolationForest`` with orchestration-aligned defaults once.

    Args:
        x: Training features for both fitting and scoring in these studies.
        random_state: Deterministic RNG seed for tree bagging reproducibility.
        forest_kw: Subset matching ``IForestParams`` excluding scheduler-controlled flags.

    Returns:
        Floating scores where larger implies stronger anomaly suspicion.
    """
    model = IForestModel(
        params=IForestParams(
            bootstrap=False,
            random_state=random_state,
            n_jobs=-1,
            **forest_kw,
        )
    )
    model.fit(x)
    return np.asarray(model.score_samples(x), dtype=np.float64)


def _ocsvm_rbf_fit_scores(
    x: np.ndarray,
    nu: float,
    gamma: float | str,
) -> np.ndarray:
    """Fit RBF OCSVM once and return training-set anomaly scores."""
    model = OCSVMModel(params=OCSVMParams(kernel="rbf", nu=float(nu), gamma=gamma))
    model.fit(x)
    return np.asarray(model.score_samples(x), dtype=np.float64)


def _dense_iforest_bootstrap_grid(limit: int = 48, seed: int = 90210) -> tuple[dict[str, Any], ...]:
    """Enumerate a Cartesian IForest probe grid then deterministically thin it."""
    combos = []
    for n_estimators, max_samples, contamination, max_features in it.product(
        (50, 75, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500),
        (64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512),
        (0.03, 0.05, 0.07, 0.095, 0.12, 0.15, 0.18, 0.21, 0.24, 0.29),
        (0.50, 0.625, 0.75, 0.875, 1.00),
    ):
        combos.append(
            {
                "n_estimators": int(n_estimators),
                "max_samples": int(max_samples),
                "contamination": float(contamination),
                "max_features": float(max_features),
            }
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(combos))
    trimmed = [combos[int(idx)] for idx in order[: min(limit, len(combos))]]
    return tuple(trimmed)


IFOREST_BOOTSTRAP_CANDIDATES: tuple[dict[str, Any], ...] = _dense_iforest_bootstrap_grid()
"""Candidate IForest hyperparameter settings for bootstrap stability study in phase 4."""


def run_sobol_iforest_task(task: ExperimentTask, loader: DatasetLoader) -> TaskResult:
    """Execute Saltelli sampling for IForest hyperparameters maximizing PR-AUC.

    Loads data using the sibling benchmark study configuration so Sobol workloads
    share preprocessing with Track A manifests, evaluates every draw, derives
    first-order Sobol indices, and retrains champion settings for tagging metrics.

    Args:
        task: Manifest entry carrying dataset id and Sobol knobs.
        loader: Hydrates canonical tensors with catalog-aware preprocessing.

    Returns:
        ``TaskResult`` bundling classifier metrics plus Sobol-derived extras.

    Raises:
        SensitivityError: Propagated when SALib is unavailable in the interpreter.
    """
    started = perf_counter()
    synthetic = ExperimentTask(
        phase="phase4",
        dataset_id=task.dataset_id,
        algorithm="IForest",
        seed=task.seed,
        task_index=task.task_index,
        variant=task.variant,
        params={"study": "benchmark"},
    )
    bundle = _load_phase4_bundle(synthetic, loader, loader.catalog)
    x = np.asarray(bundle.X.to_numpy(), dtype=np.float64)
    y = np.asarray(bundle.y.to_numpy(), dtype=np.int64)
    spec = loader.catalog.get(task.dataset_id)
    threshold = MetricThresholdConfig(contamination=spec.contamination)

    base_n = int(task.params.get("saltelli_n", 28))
    calc_second = bool(task.params.get("sobol_calc_second_order", False))
    sample_matrix = sobol_sample(IFOREST_SOBOL_SPACE, base_n, calc_second_order=calc_second)

    def objective(raw: dict[str, float | int | str]) -> float:
        forest_kw = {
            "n_estimators": int(raw["n_estimators"]),
            "max_samples": int(raw["max_samples"]),
            "contamination": float(raw["contamination"]),
            "max_features": float(raw["max_features"]),
        }
        scores = _iforest_fit_scores(x, random_state=42, forest_kw=forest_kw)
        return float(metric_pr_auc(y, scores))

    values = evaluate_sobol_objective(IFOREST_SOBOL_SPACE, sample_matrix, objective)
    analysis = sobol_analyze(IFOREST_SOBOL_SPACE, values, calc_second_order=calc_second)
    best_row = int(np.argmax(values))
    best_params = IFOREST_SOBOL_SPACE.decode_row(sample_matrix[best_row])
    scores = _iforest_fit_scores(
        x,
        random_state=42,
        forest_kw={
            "n_estimators": int(best_params["n_estimators"]),
            "max_samples": int(best_params["max_samples"]),
            "contamination": float(best_params["contamination"]),
            "max_features": float(best_params["max_features"]),
        },
    )
    report = evaluate_metrics(y_true=y, scores=scores, threshold_config=threshold)

    extra: dict[str, float] = {
        "sobol_mean_objective_pr_auc": float(np.mean(values)),
        "sobol_max_objective_pr_auc": float(np.max(values)),
        "sobol_best_n_estimators": float(best_params["n_estimators"]),
        "saltelli_base_n": float(base_n),
        "distance_concentration_ratio": distance_concentration_ratio(x),
        "n_features": float(x.shape[1]),
        "n_samples_fit": float(x.shape[0]),
    }
    names = IFOREST_SOBOL_SPACE.salib_problem["names"]
    first_order = np.asarray(analysis["S1"], dtype=np.float64).ravel()
    for index, name in enumerate(names):
        extra[f"sobol_S1_{name}"] = float(first_order[index])

    iso = odds_isolation_decile_metrics(x, y, scores, contamination=spec.contamination)
    extra.update(iso)

    elapsed = perf_counter() - started
    return TaskResult(
        run_key=task.run_key,
        dataset_id=task.dataset_id,
        algorithm=task.algorithm,
        seed=task.seed,
        duration_seconds=float(elapsed),
        metrics=report,
        extra_metrics=extra,
        artifacts={
            "sobol_summary": {
                "saltelli_rows": int(sample_matrix.shape[0]),
                "parameter_names": list(names),
                "best_row": best_row,
            }
        },
    )


def run_sobol_ocsvm_task(task: ExperimentTask, loader: DatasetLoader) -> TaskResult:
    """Saltelli Sobol on RBF OCSVM ``nu`` and scalar ``gamma`` (10**log10_gamma)."""
    started = perf_counter()
    synthetic = ExperimentTask(
        phase="phase4",
        dataset_id=task.dataset_id,
        algorithm="OCSVM",
        seed=task.seed,
        task_index=task.task_index,
        variant=task.variant,
        params={"study": "benchmark"},
    )
    bundle = _load_phase4_bundle(synthetic, loader, loader.catalog)
    x = np.asarray(bundle.X.to_numpy(), dtype=np.float64)
    y = np.asarray(bundle.y.to_numpy(), dtype=np.int64)
    spec = loader.catalog.get(task.dataset_id)
    threshold = MetricThresholdConfig(contamination=spec.contamination)

    base_n = int(task.params.get("saltelli_n", 28))
    calc_second = bool(task.params.get("sobol_calc_second_order", False))
    sample_matrix = sobol_sample(OCSVM_SOBOL_SPACE, base_n, calc_second_order=calc_second)

    def objective(raw: dict[str, float | int | str]) -> float:
        nu = float(raw["nu"])
        gamma = float(10 ** float(raw["log10_gamma"]))
        scores = _ocsvm_rbf_fit_scores(x, nu=nu, gamma=gamma)
        return float(metric_pr_auc(y, scores))

    values = evaluate_sobol_objective(OCSVM_SOBOL_SPACE, sample_matrix, objective)
    analysis = sobol_analyze(OCSVM_SOBOL_SPACE, values, calc_second_order=calc_second)
    best_row = int(np.argmax(values))
    best_params = OCSVM_SOBOL_SPACE.decode_row(sample_matrix[best_row])
    champion_nu = float(best_params["nu"])
    champion_gamma = float(10 ** float(best_params["log10_gamma"]))
    scores = _ocsvm_rbf_fit_scores(x, nu=champion_nu, gamma=champion_gamma)
    report = evaluate_metrics(y_true=y, scores=scores, threshold_config=threshold)

    extra: dict[str, float] = {
        "sobol_mean_objective_pr_auc": float(np.mean(values)),
        "sobol_max_objective_pr_auc": float(np.max(values)),
        "sobol_best_nu": float(champion_nu),
        "saltelli_base_n": float(base_n),
        "distance_concentration_ratio": distance_concentration_ratio(x),
        "n_features": float(x.shape[1]),
        "n_samples_fit": float(x.shape[0]),
    }
    names = OCSVM_SOBOL_SPACE.salib_problem["names"]
    first_order = np.asarray(analysis["S1"], dtype=np.float64).ravel()
    for index, name in enumerate(names):
        extra[f"sobol_S1_{name}"] = float(first_order[index])

    iso = odds_isolation_decile_metrics(x, y, scores, contamination=spec.contamination)
    extra.update(iso)

    elapsed = perf_counter() - started
    return TaskResult(
        run_key=task.run_key,
        dataset_id=task.dataset_id,
        algorithm=task.algorithm,
        seed=task.seed,
        duration_seconds=float(elapsed),
        metrics=report,
        extra_metrics=extra,
        artifacts={
            "sobol_summary": {
                "saltelli_rows": int(sample_matrix.shape[0]),
                "parameter_names": list(names),
                "best_row": best_row,
            }
        },
    )


def run_lhs_ocsvm_task(task: ExperimentTask, loader: DatasetLoader) -> TaskResult:
    """Latin Hypercube draws over discrete ``nu`` and ``gamma`` grids (plan §2.1)."""
    started = perf_counter()
    synthetic = ExperimentTask(
        phase="phase4",
        dataset_id=task.dataset_id,
        algorithm="OCSVM",
        seed=task.seed,
        task_index=task.task_index,
        variant=task.variant,
        params={"study": "benchmark"},
    )
    bundle = _load_phase4_bundle(synthetic, loader, loader.catalog)
    x = np.asarray(bundle.X.to_numpy(), dtype=np.float64)
    y = np.asarray(bundle.y.to_numpy(), dtype=np.int64)
    spec = loader.catalog.get(task.dataset_id)
    threshold = MetricThresholdConfig(contamination=spec.contamination)

    draws = int(task.params.get("lhs_n_samples", 80))
    lhs_seed = int(task.params.get("lhs_seed", task.seed))
    configs = lhs_dict_rows(n_samples=draws, seed=lhs_seed)

    aucs = np.empty(shape=(len(configs),), dtype=np.float64)
    for idx, combo in enumerate(configs):
        scores_try = _ocsvm_rbf_fit_scores(x, nu=float(combo["nu"]), gamma=combo["gamma"])
        aucs[idx] = float(metric_pr_auc(y, scores_try))

    best_idx = int(np.argmax(aucs))
    champion_cfg = dict(configs[best_idx])
    champion_scores = _ocsvm_rbf_fit_scores(
        x, nu=float(champion_cfg["nu"]), gamma=champion_cfg["gamma"]
    )

    gamma_val = champion_cfg["gamma"]
    gamma_encoding = float(gamma_val) if isinstance(gamma_val, numbers.Real) else float("nan")

    report = evaluate_metrics(y_true=y, scores=champion_scores, threshold_config=threshold)

    extra: dict[str, float] = {
        "lhs_mean_objective_pr_auc": float(np.mean(aucs)),
        "lhs_std_objective_pr_auc": float(np.std(aucs, ddof=0)),
        "lhs_max_objective_pr_auc": float(np.max(aucs)),
        "lhs_best_nu": float(champion_cfg["nu"]),
        "lhs_best_gamma_encoded": gamma_encoding,
        "lhs_draw_count": float(len(configs)),
        "distance_concentration_ratio": distance_concentration_ratio(x),
        "n_features": float(x.shape[1]),
        "n_samples_fit": float(x.shape[0]),
    }

    iso = odds_isolation_decile_metrics(x, y, champion_scores, contamination=spec.contamination)
    extra.update(iso)

    elapsed = perf_counter() - started
    return TaskResult(
        run_key=task.run_key,
        dataset_id=task.dataset_id,
        algorithm=task.algorithm,
        seed=task.seed,
        duration_seconds=float(elapsed),
        metrics=report,
        extra_metrics=extra,
        artifacts={
            "lhs_summary": {
                "draws": len(configs),
                "best_index": best_idx,
                "best_row": champion_cfg,
                "lhs_seed": lhs_seed,
                "gamma_literal": repr(gamma_val),
                "auc_vector": tuple(float(value) for value in aucs.tolist()),
            },
        },
    )


def run_bootstrap_iforest_stable(task: ExperimentTask, loader: DatasetLoader) -> TaskResult:
    """Pick an IForest configuration balancing mean PR-AUC vs bootstrap dispersion.

    Args:
        task: Manifest entry referencing ``bootstrap_dataset_id`` probes.
        loader: Retrieves preprocessed ODDS tensors for thyroid by default.

    Returns:
        ``TaskResult`` for the stabilized champion including isolation-bin extras.

    Notes:
        When strict stability constraints disqualify everyone, selection falls back
        to the highest mean PR-AUC configuration instead of throwing.
    """
    started = perf_counter()
    ds_id = str(task.params.get("bootstrap_dataset_id", "thyroid"))
    bundle = loader.load(ds_id, view="preprocessed")
    x = np.asarray(bundle.X.to_numpy(), dtype=np.float64)
    y = np.asarray(bundle.y.to_numpy(), dtype=np.int64)
    spec = loader.catalog.get(ds_id)

    summaries: list[CandidateStability] = []
    for candidate in IFOREST_BOOTSTRAP_CANDIDATES:

        def score_fn(xx: np.ndarray, p: dict[str, Any] = candidate) -> np.ndarray:
            return _iforest_fit_scores(xx, random_state=task.seed, forest_kw=p)

        boot = bootstrap_pr_auc_stability(
            x=x,
            y_true=y,
            score_fn=score_fn,
            n_resamples=10,
            sample_fraction=0.9,
            random_state=task.seed,
        )
        summaries.append(
            CandidateStability(
                params=dict(candidate),
                mean_pr_auc=boot.mean_pr_auc,
                std_pr_auc=boot.std_pr_auc,
            )
        )

    try:
        chosen = select_stable_best(summaries)
    except SensitivityError:
        chosen = max(summaries, key=lambda c: (c.mean_pr_auc, -c.std_pr_auc))
    scores = _iforest_fit_scores(x, random_state=task.seed, forest_kw=dict(chosen.params))
    report = evaluate_metrics(
        y_true=y,
        scores=scores,
        threshold_config=MetricThresholdConfig(contamination=spec.contamination),
    )

    extra: dict[str, float] = {
        "bootstrap_candidate_count": float(len(IFOREST_BOOTSTRAP_CANDIDATES)),
        "bootstrap_chosen_mean_pr_auc": float(chosen.mean_pr_auc),
        "bootstrap_chosen_std_pr_auc": float(chosen.std_pr_auc),
        "bootstrap_chosen_n_estimators": float(chosen.params["n_estimators"]),
        "distance_concentration_ratio": distance_concentration_ratio(x),
        "n_features": float(x.shape[1]),
        "n_samples_fit": float(x.shape[0]),
    }
    extra.update(odds_isolation_decile_metrics(x, y, scores, contamination=spec.contamination))

    elapsed = perf_counter() - started
    return TaskResult(
        run_key=task.run_key,
        dataset_id=ds_id,
        algorithm=task.algorithm,
        seed=task.seed,
        duration_seconds=float(elapsed),
        metrics=report,
        extra_metrics=extra,
        artifacts={
            "bootstrap_candidates": {
                "winner": dict(chosen.params),
                "candidates": [c.model_dump() for c in summaries],
            }
        },
    )
