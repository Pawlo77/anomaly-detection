"""Task-level model fit and evaluation runner."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..warnings_filters import apply_known_sklearn_experiment_warnings
import numpy.typing as npt
import pandas as pd
from sklearn.decomposition import PCA

from ..datasets.catalog import DatasetCatalog
from ..datasets.loader import DatasetLoader
from ..datasets.types import DatasetBundle
from ..geometry import distance_concentration_ratio
from ..metrics import (
    MetricsReport,
    MetricThresholdConfig,
    evaluate_metrics,
    smtp_extreme_skew_metrics,
)
from ..models import DBSCANModel, ECODModel, HBOSModel, IForestModel, LOFModel, OCSVMModel
from ..models.params import (
    DBSCANParams,
    ECODParams,
    HBOSParams,
    IForestParams,
    LOFParams,
    OCSVMParams,
)
from .blind_features import prepare_blind_feature_matrix
from .tasks import ExperimentTask

type FloatArray = npt.NDArray[np.float64]

ORCH_METADATA_KEYS = frozenset(
    {
        "study",
        "pca_variance",
        "apply_pca",
        "data_view",
        "saltelli_n",
        "sobol_calc_second_order",
        "bootstrap_dataset_id",
        "lhs_n_samples",
        "lhs_seed",
    }
)
"""Orchestrator metadata keys reserved from model hyperparameters and MLflow logging."""

TEST_DATA_FILENAME = "test_data.csv"
"""Input CSV file used by phase-5 task runner."""


@dataclass(frozen=True, slots=True)
class TaskResult:
    """Serializable result of a single fit/evaluate task.

    Attributes:
        run_key: Deterministic run key tied to manifest entry.
        dataset_id: Dataset identifier used for this task.
        algorithm: Algorithm name used by model wrapper.
        seed: Random seed used by this task.
        duration_seconds: Wall-clock duration of fit/evaluation stage in seconds.
        metrics: Core metrics report from unified evaluation protocol.
        extra_metrics: Additional phase-specific scalar metrics.
        artifacts: Additional phase-specific structured artifact payloads.
    """

    run_key: str
    dataset_id: str
    algorithm: str
    seed: int
    duration_seconds: float
    metrics: MetricsReport
    extra_metrics: dict[str, float]
    artifacts: dict[str, dict[str, Any]]


def _build_model(algorithm: str, params: dict[str, Any]) -> Any:
    """Hydrate estimator wrappers keyed by categorical algorithm names.

    Args:
        algorithm: Case-sensitive key such as ``IForest`` or ``DBSCAN``.
        params: Hyperparameters passed into typed pydantic parameter models.

    Returns:
        Protocol-compliant anomaly wrapper exposing ``fit``/``score_samples``.

    Raises:
        ValueError: When ``algorithm`` is unknown.
    """
    if algorithm == "OCSVM":
        return OCSVMModel(params=OCSVMParams(**params))
    if algorithm == "IForest":
        merged = {**params, "n_jobs": -1}
        return IForestModel(params=IForestParams(**merged))
    if algorithm == "LOF":
        merged = {**params, "n_jobs": -1}
        return LOFModel(params=LOFParams(**merged))
    if algorithm == "DBSCAN":
        merged = {**params, "n_jobs": -1}
        return DBSCANModel(params=DBSCANParams(**merged))
    if algorithm == "ECOD":
        return ECODModel(params=ECODParams(**params))
    if algorithm == "HBOS":
        return HBOSModel(params=HBOSParams(**params))
    raise ValueError(f"Unknown algorithm: {algorithm}")


def run_task(task: ExperimentTask, loader: DatasetLoader | None = None) -> TaskResult:
    """Run load, optional PCA view wiring, fit, score, and unified metrics.

    IForest/OCSVM Sobol schedules and IForest bootstrap suites short-circuit into
    ``sensitivity_execution`` instead of the generic fit path.

    Args:
        task: Experiment task definition including manifest metadata.
        loader: Optional dataset loader override (defaults to new ``DatasetLoader``).

    Returns:
        Finished task bundle with ``MetricsReport``, extras, and artifact payloads.
    """
    apply_known_sklearn_experiment_warnings()
    active_loader = loader or DatasetLoader()
    catalog = active_loader.catalog
    study = str(task.params.get("study", "default"))

    if task.phase == "phase4" and study == "sobol_iforest":
        from .sensitivity_execution import run_sobol_iforest_task

        return run_sobol_iforest_task(task, active_loader)

    if task.phase == "phase4" and study == "sobol_ocsvm":
        from .sensitivity_execution import run_sobol_ocsvm_task

        return run_sobol_ocsvm_task(task, active_loader)

    if task.phase == "phase4" and study == "bootstrap_stable_iforest":
        from .sensitivity_execution import run_bootstrap_iforest_stable

        return run_bootstrap_iforest_stable(task, active_loader)

    if task.phase == "phase4" and study == "lhs_ocsvm":
        from .sensitivity_execution import run_lhs_ocsvm_task

        return run_lhs_ocsvm_task(task, active_loader)

    if task.dataset_id == "test_data":
        x, y = _load_phase5_table()
        threshold_config = MetricThresholdConfig()
    elif task.phase == "phase4":
        bundle = _load_phase4_bundle(task, active_loader, catalog)
        x = np.asarray(bundle.X.to_numpy(), dtype=np.float64)
        y = np.asarray(bundle.y.to_numpy(), dtype=np.int64)
        spec = catalog.get(task.dataset_id)
        threshold_config = MetricThresholdConfig(contamination=spec.contamination)
    else:
        bundle = active_loader.load(dataset_id=task.dataset_id, view="preprocessed")
        x = np.asarray(bundle.X.to_numpy(), dtype=np.float64)
        y = np.asarray(bundle.y.to_numpy(), dtype=np.int64)
        spec = catalog.get(task.dataset_id)
        threshold_config = MetricThresholdConfig(contamination=spec.contamination)

    if (
        task.phase == "phase4"
        and study == "dimensionality"
        and bool(task.params.get("apply_pca", True))
    ):
        x = _apply_pca_variance(x, variance=float(task.params["pca_variance"]), seed=task.seed)

    model_params = {k: v for k, v in task.params.items() if k not in ORCH_METADATA_KEYS}
    model = _build_model(task.algorithm, model_params)

    started = perf_counter()
    model.fit(x)
    scores: FloatArray = np.asarray(model.score_samples(x), dtype=np.float64)
    report = evaluate_metrics(y_true=y, scores=scores, threshold_config=threshold_config)
    elapsed = perf_counter() - started

    extra_metrics: dict[str, float] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    if task.phase == "phase4":
        extra_metrics["distance_concentration_ratio"] = distance_concentration_ratio(x)
        extra_metrics["n_features"] = float(x.shape[1])
        extra_metrics["n_samples_fit"] = float(x.shape[0])
        spec_fit = catalog.get(task.dataset_id)
        extra_metrics["subsampled"] = float(
            spec_fit.stratified_subsample_for_n2 and task.algorithm in ("OCSVM", "LOF", "DBSCAN")
        )
        if study == "dimensionality":
            applied = float(bool(task.params.get("apply_pca", True)))
            extra_metrics["pca_applied"] = applied
            if applied:
                extra_metrics["pca_variance"] = float(task.params["pca_variance"])
        if study == "arrhythmia_pca95":
            extra_metrics["pca95_view"] = 1.0
            extra_metrics["pca_variance_target"] = float(
                active_loader.settings.arrhythmia_pca_variance
            )
        if study == "taxonomy":
            taxonomy = _taxonomy_recall(x=x, y=y, scores=scores)
            extra_metrics.update(
                {
                    "recall_type_g": taxonomy["recall_type_g"],
                    "recall_type_l": taxonomy["recall_type_l"],
                }
            )
            artifacts["taxonomy_summary"] = taxonomy
        if study.startswith("oracle_"):
            extra_metrics["oracle_retrospective"] = 1.0
        odds_proxy = "/" not in task.dataset_id and np.any(y == 1)
        if odds_proxy and (study == "benchmark" or study.startswith("oracle_")):
            from .isolation_bins import odds_isolation_decile_metrics

            spec_iso = catalog.get(task.dataset_id)
            extra_metrics.update(
                odds_isolation_decile_metrics(x, y, scores, contamination=spec_iso.contamination)
            )
        if task.dataset_id == "smtp" and study == "benchmark":
            extra_metrics.update(
                smtp_extreme_skew_metrics(
                    contamination=spec_fit.contamination,
                    model_accuracy=report.accuracy,
                )
            )

    return TaskResult(
        run_key=task.run_key,
        dataset_id=task.dataset_id,
        algorithm=task.algorithm,
        seed=task.seed,
        duration_seconds=float(elapsed),
        metrics=report,
        extra_metrics=extra_metrics,
        artifacts=artifacts,
    )


def _load_phase5_table() -> tuple[FloatArray, npt.NDArray[np.int64]]:
    """Load ``test_data.csv`` pairing engineered features with optional labels.

    Returns:
        Feature matrix alongside label vector zero-filled when unspecified.

    Raises:
        FileNotFoundError: When test artifact missing from cwd expectations.
    """
    table_path = Path(TEST_DATA_FILENAME)
    if not table_path.exists():
        raise FileNotFoundError(f"{TEST_DATA_FILENAME} not found for phase5 tasks.")
    table = pd.read_csv(table_path)
    x = prepare_blind_feature_matrix(table)

    if "class" in table.columns:
        y = np.asarray(table["class"].astype(int).to_numpy(), dtype=np.int64)
    elif "label" in table.columns:
        y = np.asarray(table["label"].astype(int).to_numpy(), dtype=np.int64)
    else:
        y = np.zeros(shape=(table.shape[0],), dtype=np.int64)
    return x, y


def _load_phase4_bundle(
    task: ExperimentTask, loader: DatasetLoader, catalog: DatasetCatalog
) -> DatasetBundle:
    """Load preprocessed artifact, applying plan §1.4 subsampling when required."""
    spec = catalog.get(task.dataset_id)
    raw_view = task.params.get("data_view")
    view = raw_view if raw_view in {"preprocessed", "pca95", "raw"} else "preprocessed"
    pca_rs = task.seed if view == "pca95" else None
    if task.algorithm in ("OCSVM", "LOF", "DBSCAN") and spec.stratified_subsample_for_n2:
        return loader.load_subsample(
            task.dataset_id,
            task.algorithm,
            task.seed,
            view=view,
            pca_random_state=pca_rs,
        )
    return loader.load(dataset_id=task.dataset_id, view=view, pca_random_state=pca_rs)


def _apply_pca_variance(x: FloatArray, variance: float, seed: int) -> FloatArray:
    """Project features with PCA retaining cumulative variance quota.

    Args:
        x: Dense float matrix sampled from loader bundles.
        variance: Fractional variance target forwarded to sklearn ``PCA``.
        seed: RNG seed guarding randomized decompositions when applicable.

    Returns:
        Transformed matrix occupying reduced dimensionality embeddings.
    """
    pca = PCA(n_components=variance, svd_solver="full", random_state=seed)
    return np.asarray(pca.fit_transform(x), dtype=np.float64)


def _taxonomy_recall(x: FloatArray, y: npt.NDArray[np.int64], scores: FloatArray) -> dict[str, Any]:
    """Split labelled-noise probes into §4.2 ``Type G`` vs ``Type L`` using cluster geometry."""
    from .outlier_taxonomy import classify_positives_global_vs_localclusters

    positives = np.where(y == 1)[0]
    if positives.size == 0:
        return {"recall_type_g": 0.0, "recall_type_l": 0.0, "type_g_count": 0, "type_l_count": 0}

    type_g, type_l, _ = classify_positives_global_vs_localclusters(x, y)

    contamination = positives.size / y.size
    pred = (scores >= np.quantile(scores, 1.0 - contamination)).astype(np.int64)
    recall_g = float(np.mean(pred[type_g] == 1)) if type_g.size else 0.0
    recall_l = float(np.mean(pred[type_l] == 1)) if type_l.size else 0.0
    return {
        "recall_type_g": recall_g,
        "recall_type_l": recall_l,
        "type_g_count": int(type_g.size),
        "type_l_count": int(type_l.size),
    }
