"""Deterministic manifest generation for phased anomaly-detection benchmarks.

Produces ordered ``ExperimentTask`` tuples spanning Track A sweeps, Track B oracle
studies, scheduled sensitivity aggregates, and blind phase-five scoring.
"""

from dataclasses import dataclass
from typing import Any

from ..datasets.catalog import DatasetCatalog, build_default_catalog
from ..training.tasks import ExperimentTask

PHASE4_DATASETS: tuple[str, ...] = ("arrhythmia", "musk", "speech", "mnist")
"""Primary phase-4 datasets used in dimensionality stress test."""
PHASE4_TAXONOMY_DATASETS: tuple[str, ...] = ("wut/x2", "sipu/flame")
"""Phase-4 datasets with explicit noise labels for local/global taxonomy analysis."""
PHASE4_ALGORITHMS: tuple[str, ...] = ("OCSVM", "IForest", "LOF", "DBSCAN", "ECOD", "HBOS")
"""Algorithms executed in phase-4 studies."""
PHASE4_PCA_VARIANCE: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
"""PCA retained-variance ladder for phase-4 dimensionality study."""

PHASE5_ALGORITHMS: tuple[str, ...] = ("OCSVM", "IForest", "LOF", "DBSCAN", "ECOD", "HBOS")
"""Algorithms used by blind phase-5 ensemble workflow."""

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 42)
"""Default random seeds for repeated experiment runs."""

SOBOL_IFOREST_DATASETS: tuple[str, ...] = ("ionosphere", "breastw", "smtp")
"""ODDS probes for scheduled IForest Sobol sensitivity."""
SOBOL_OCSVM_DATASETS: tuple[str, ...] = ("breastw", "ionosphere", "pima")
"""Prior-heavy ODDS probes for RBF OCSVM Saltelli Sobol (plan §2.1, §3.2)."""

LHS_OCSVM_DATASETS: tuple[str, ...] = SOBOL_OCSVM_DATASETS
"""Datasets queued for LHS ``nu``-by-``gamma`` draws (mirrors Sobol probes)."""

ARRHYTHMIA_PCA_DEPENDENT_ALGOS: frozenset[str] = frozenset({"LOF", "DBSCAN"})
"""Algorithms that must use PCA on ``arrhythmia`` per plan §1.3 (defer to ``pca95`` tasks)."""

OCSVM_ORACLE_DATASET_ID = "ionosphere"
"""Dataset ID for OCSVM oracle grid."""
OCSVM_ORACLE_NUS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.35, 0.50)
"""Nu values for OCSVM oracle grid."""
OCSVM_ORACLE_GAMMA: tuple[str | float, ...] = ("scale", "auto", 0.001, 0.01, 0.1, 1.0, 10.0)
"""Gamma values for OCSVM oracle grid."""

IFOREST_ORACLE_DATASET_ID = "breastw"
"""Dataset ID for IForest oracle grid."""
IFOREST_ORACLE_N_EST: tuple[int, ...] = (50, 100, 200, 500)
"""Number of estimators for IForest oracle grid."""
IFOREST_ORACLE_MAX_SAMPLES: tuple[int, ...] = (128, 256)
"""Maximum samples for IForest oracle grid."""
IFOREST_ORACLE_CONT: tuple[float, ...] = (0.05, 0.10)
"""Contamination values for IForest oracle grid."""

LOF_ORACLE_DATASET_ID = "satellite"
"""Dataset ID for LOF oracle grid."""
LOF_ORACLE_K: tuple[int, ...] = (5, 10, 15, 20, 30, 50, 75, 100)
"""Neighborhood sizes for LOF oracle grid."""

DBSCAN_ORACLE_DATASET_ID = "satellite"
"""Dataset ID for DBSCAN oracle grid."""
DBSCAN_EPS_MULTS: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
"""Epsilon multipliers for DBSCAN oracle grid."""
DBSCAN_ORACLE_MIN_SAMPLES: tuple[int, ...] = (3, 5, 10, 15, 20, 30)
"""Minimum samples for DBSCAN oracle grid."""

LOF_ORACLE_EXTRA_METRICS: tuple[dict[str, Any], ...] = (
    {"metric": "manhattan"},
    {"metric": "cosine"},
    {"metric": "minkowski", "p": 3},
)
"""Extra LOF oracle metrics beyond Euclidean primary sweep (plan §2.3)."""

ECOD_ORACLE_DATASET_ID = "pima"
"""Dataset ID for ECOD oracle grid."""
ECOD_ORACLE_CONT: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.35, 0.50)
"""Contamination values for ECOD oracle grid."""
HBOS_ORACLE_DATASET_ID = "satellite"
"""Dataset ID for HBOS oracle grid."""
HBOS_ORACLE_N_BINS: tuple[int, ...] = (5, 10, 20, 30, 50)
"""Number of bins for HBOS oracle grid."""
HBOS_ORACLE_ALPHA: tuple[float, ...] = (0.0, 0.1, 0.2)
"""Alpha values for HBOS oracle grid."""


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    """Ordered list of experiment tasks for one run request.

    Attributes:
        tasks: Deterministic task sequence consumed by reconciliation and scheduler.
    """

    tasks: tuple[ExperimentTask, ...]


def _benchmark_hyperparams(algorithm: str, seed: int) -> dict[str, Any]:
    """Return Track A baseline hyperparameters (experimental plan §3.2).

    Args:
        algorithm: Registered estimator name.
        seed: Random-state seed echoed into stochastic models.

    Returns:
        Payload merged into benchmark ``ExperimentTask.params``.
    """
    if algorithm == "IForest":
        return {
            "n_estimators": 200,
            "max_samples": 256,
            "bootstrap": False,
            "random_state": seed,
        }
    return {}


def _append_oracle_track_b(catalog_obj: DatasetCatalog, tasks: list[ExperimentTask], i: int) -> int:
    """Append Track B oracle grids (plan §3.2) for representative ODDS probes.

    Args:
        catalog_obj: Catalog used to validate dataset identifiers.
        tasks: Mutable manifest queue.
        i: Monotonic task counter before appends.

    Returns:
        Task counter after oracle tasks are enqueued.
    """
    catalog_obj.get(OCSVM_ORACLE_DATASET_ID)
    for nu in OCSVM_ORACLE_NUS:
        for gamma in OCSVM_ORACLE_GAMMA:
            for seed in DEFAULT_SEEDS:
                tasks.append(
                    ExperimentTask(
                        phase="phase4",
                        dataset_id=OCSVM_ORACLE_DATASET_ID,
                        algorithm="OCSVM",
                        seed=seed,
                        task_index=i,
                        variant="oracle-ocsvm-rbf",
                        params={
                            "study": "oracle_ocsvm_rbf",
                            "kernel": "rbf",
                            "nu": nu,
                            "gamma": gamma,
                        },
                        tags={"track": "b", "study": "oracle_ocsvm_rbf"},
                    )
                )
                i += 1

    catalog_obj.get(IFOREST_ORACLE_DATASET_ID)
    for n_est in IFOREST_ORACLE_N_EST:
        for max_s in IFOREST_ORACLE_MAX_SAMPLES:
            for contamin in IFOREST_ORACLE_CONT:
                for seed in DEFAULT_SEEDS:
                    tasks.append(
                        ExperimentTask(
                            phase="phase4",
                            dataset_id=IFOREST_ORACLE_DATASET_ID,
                            algorithm="IForest",
                            seed=seed,
                            task_index=i,
                            variant="oracle-iforest-grid",
                            params={
                                "study": "oracle_iforest",
                                "n_estimators": n_est,
                                "max_samples": max_s,
                                "contamination": contamin,
                                "bootstrap": False,
                                "random_state": seed,
                            },
                            tags={"track": "b", "study": "oracle_iforest"},
                        )
                    )
                    i += 1

    catalog_obj.get(LOF_ORACLE_DATASET_ID)
    for nk in LOF_ORACLE_K:
        for seed in DEFAULT_SEEDS:
            tasks.append(
                ExperimentTask(
                    phase="phase4",
                    dataset_id=LOF_ORACLE_DATASET_ID,
                    algorithm="LOF",
                    seed=seed,
                    task_index=i,
                    variant="oracle-lof-neighbors-euclidean",
                    params={
                        "study": "oracle_lof_primary",
                        "n_neighbors": nk,
                        "metric": "euclidean",
                    },
                    tags={"track": "b", "study": "oracle_lof_primary"},
                )
            )
            i += 1

    for metric_kw in LOF_ORACLE_EXTRA_METRICS:
        for nk in LOF_ORACLE_K:
            for seed in DEFAULT_SEEDS:
                tasks.append(
                    ExperimentTask(
                        phase="phase4",
                        dataset_id=LOF_ORACLE_DATASET_ID,
                        algorithm="LOF",
                        seed=seed,
                        task_index=i,
                        variant="oracle-lof-neighbors-metrics",
                        params={
                            "study": "oracle_lof_primary",
                            "n_neighbors": nk,
                            **metric_kw,
                        },
                        tags={"track": "b", "study": "oracle_lof_primary"},
                    )
                )
                i += 1

    catalog_obj.get(DBSCAN_ORACLE_DATASET_ID)
    for mult in DBSCAN_EPS_MULTS:
        for ms in DBSCAN_ORACLE_MIN_SAMPLES:
            for metric_name in ("euclidean", "manhattan"):
                for seed in DEFAULT_SEEDS:
                    tasks.append(
                        ExperimentTask(
                            phase="phase4",
                            dataset_id=DBSCAN_ORACLE_DATASET_ID,
                            algorithm="DBSCAN",
                            seed=seed,
                            task_index=i,
                            variant="oracle-dbscan-knee-mult",
                            params={
                                "study": "oracle_dbscan_eps",
                                "eps_mode": "knee",
                                "eps_knee_multiplier": mult,
                                "min_samples": ms,
                                "metric": metric_name,
                            },
                            tags={"track": "b", "study": "oracle_dbscan_eps"},
                        )
                    )
                    i += 1

    catalog_obj.get(ECOD_ORACLE_DATASET_ID)
    for contamin in ECOD_ORACLE_CONT:
        for seed in DEFAULT_SEEDS:
            tasks.append(
                ExperimentTask(
                    phase="phase4",
                    dataset_id=ECOD_ORACLE_DATASET_ID,
                    algorithm="ECOD",
                    seed=seed,
                    task_index=i,
                    variant="oracle-ecod-cont",
                    params={"study": "oracle_ecod", "contamination": contamin},
                    tags={"track": "b", "study": "oracle_ecod"},
                )
            )
            i += 1

    catalog_obj.get(HBOS_ORACLE_DATASET_ID)
    for bins in HBOS_ORACLE_N_BINS:
        for alpha in HBOS_ORACLE_ALPHA:
            for seed in DEFAULT_SEEDS:
                tasks.append(
                    ExperimentTask(
                        phase="phase4",
                        dataset_id=HBOS_ORACLE_DATASET_ID,
                        algorithm="HBOS",
                        seed=seed,
                        task_index=i,
                        variant="oracle-hbins-alpha",
                        params={
                            "study": "oracle_hbos_bins",
                            "n_bins": bins,
                            "alpha": alpha,
                        },
                        tags={"track": "b", "study": "oracle_hbos_bins"},
                    )
                )
                i += 1
    return i


def _append_sobol_and_bootstrap(
    catalog_obj: DatasetCatalog, tasks: list[ExperimentTask], i: int
) -> int:
    """Enqueue scheduled IForest Sobol runs and the bootstrap-stable aggregate.

    Args:
        catalog_obj: Catalog used for dataset lookups.
        tasks: Mutable manifest queue.
        i: Starting task counter.

    Returns:
        Incremented counter after queued sensitivity tasks.
    """
    for ds in SOBOL_IFOREST_DATASETS:
        catalog_obj.get(ds)
        tasks.append(
            ExperimentTask(
                phase="phase4",
                dataset_id=ds,
                algorithm="IForest",
                seed=42,
                task_index=i,
                variant=f"sobol-iforest-{ds}",
                params={"study": "sobol_iforest", "saltelli_n": 28},
                tags={"track": "sensitivity", "study": "sobol_iforest"},
            )
        )
        i += 1

    for ds in SOBOL_OCSVM_DATASETS:
        catalog_obj.get(ds)
        tasks.append(
            ExperimentTask(
                phase="phase4",
                dataset_id=ds,
                algorithm="OCSVM",
                seed=42,
                task_index=i,
                variant=f"sobol-ocsvm-{ds}",
                params={"study": "sobol_ocsvm", "saltelli_n": 28},
                tags={"track": "sensitivity", "study": "sobol_ocsvm"},
            )
        )
        i += 1

    for ds in LHS_OCSVM_DATASETS:
        catalog_obj.get(ds)
        tasks.append(
            ExperimentTask(
                phase="phase4",
                dataset_id=ds,
                algorithm="OCSVM",
                seed=42,
                task_index=i,
                variant=f"lhs-ocsvm-{ds}",
                params={"study": "lhs_ocsvm", "lhs_n_samples": 80, "lhs_seed": 42},
                tags={"track": "sensitivity", "study": "lhs_ocsvm"},
            )
        )
        i += 1

    catalog_obj.get("thyroid")
    tasks.append(
        ExperimentTask(
            phase="phase4",
            dataset_id="thyroid",
            algorithm="IForest",
            seed=42,
            task_index=i,
            variant="bootstrap-iforest-stable",
            params={
                "study": "bootstrap_stable_iforest",
                "bootstrap_dataset_id": "thyroid",
            },
            tags={"track": "sensitivity", "study": "bootstrap_stable_iforest"},
        )
    )
    return i + 1


def build_manifest(phase: str, catalog: DatasetCatalog | None = None) -> ExperimentManifest:
    """Build deterministic task list for selected experiment phase.

    Args:
        phase: Requested phase: ``phase4`` (primary + optional subsets), ``oracle``
            (Track B grids + Sobol + bootstrap sensitivity only), ``phase5``, ``all``.
        catalog: Optional dataset catalog override.

    Returns:
        Immutable experiment manifest.
    """
    catalog_obj = catalog or build_default_catalog()
    task_index = 0
    tasks: list[ExperimentTask] = []
    selected = _normalize_phase(phase)

    include_primary = selected in {"phase4", "all"}
    include_oracle_pack = selected in {"oracle", "all"}
    include_phase5 = selected in {"phase5", "all"}

    if include_primary:
        track_dataset_ids = tuple(sorted(cid for cid in catalog_obj.ids()))
        for dataset_id in track_dataset_ids:
            catalog_obj.get(dataset_id)
            for algorithm in PHASE4_ALGORITHMS:
                if dataset_id == "arrhythmia" and algorithm in ARRHYTHMIA_PCA_DEPENDENT_ALGOS:
                    continue
                for seed in DEFAULT_SEEDS:
                    tasks.append(
                        ExperimentTask(
                            phase="phase4",
                            dataset_id=dataset_id,
                            algorithm=algorithm,
                            seed=seed,
                            task_index=task_index,
                            variant="track-a-defaults",
                            params={
                                **_benchmark_hyperparams(algorithm, seed),
                                "study": "benchmark",
                            },
                            tags={"track": "a", "study": "benchmark"},
                        )
                    )
                    task_index += 1

        catalog_obj.get("arrhythmia")
        for algorithm in PHASE4_ALGORITHMS:
            for seed in DEFAULT_SEEDS:
                tasks.append(
                    ExperimentTask(
                        phase="phase4",
                        dataset_id="arrhythmia",
                        algorithm=algorithm,
                        seed=seed,
                        task_index=task_index,
                        variant="arrhythmia-pca95-track-a",
                        params={
                            **_benchmark_hyperparams(algorithm, seed),
                            "study": "arrhythmia_pca95",
                            "data_view": "pca95",
                        },
                        tags={"track": "a", "study": "arrhythmia_pca95"},
                    )
                )
                task_index += 1

        for dataset_id in PHASE4_DATASETS:
            catalog_obj.get(dataset_id)
            for algorithm in PHASE4_ALGORITHMS:
                for seed in DEFAULT_SEEDS:
                    for pca_variance in PHASE4_PCA_VARIANCE:
                        tasks.append(
                            ExperimentTask(
                                phase="phase4",
                                dataset_id=dataset_id,
                                algorithm=algorithm,
                                seed=seed,
                                task_index=task_index,
                                variant=f"pca-{pca_variance:.2f}",
                                params={
                                    "study": "dimensionality",
                                    "pca_variance": pca_variance,
                                    "apply_pca": True,
                                },
                                tags={"track": "a", "study": "dimensionality"},
                            )
                        )
                        task_index += 1
                    for algo_raw in ("LOF", "DBSCAN"):
                        tasks.append(
                            ExperimentTask(
                                phase="phase4",
                                dataset_id=dataset_id,
                                algorithm=algo_raw,
                                seed=seed,
                                task_index=task_index,
                                variant="dimensionality-raw-preprocess",
                                params={
                                    "study": "dimensionality",
                                    "apply_pca": False,
                                },
                                tags={"track": "a", "study": "dimensionality", "baseline": "raw"},
                            )
                        )
                        task_index += 1

        for dataset_id in PHASE4_TAXONOMY_DATASETS:
            catalog_obj.get(dataset_id)
            for algorithm in PHASE4_ALGORITHMS:
                for seed in DEFAULT_SEEDS:
                    tasks.append(
                        ExperimentTask(
                            phase="phase4",
                            dataset_id=dataset_id,
                            algorithm=algorithm,
                            seed=seed,
                            task_index=task_index,
                            variant="taxonomy",
                            params={"study": "taxonomy"},
                            tags={"track": "a", "study": "taxonomy"},
                        )
                    )
                    task_index += 1

    if include_oracle_pack:
        task_index = _append_oracle_track_b(catalog_obj, tasks, task_index)
        task_index = _append_sobol_and_bootstrap(catalog_obj, tasks, task_index)

    if include_phase5:
        for algorithm in PHASE5_ALGORITHMS:
            for seed in DEFAULT_SEEDS:
                tasks.append(
                    ExperimentTask(
                        phase="phase5",
                        dataset_id="test_data",
                        algorithm=algorithm,
                        seed=seed,
                        task_index=task_index,
                        variant="blind",
                        tags={"track": "blind"},
                    )
                )
                task_index += 1

    return ExperimentManifest(tasks=tuple(tasks))


def _normalize_phase(value: str) -> str:
    """Normalize CLI phase identifiers to canonical lowercase tokens.

    Args:
        value: Raw user-provided phase name.

    Returns:
        Canonical phase key among ``phase4``, ``phase5``, ``oracle``, ``all``.

    Raises:
        ValueError: When ``value`` is not a supported phase.
    """
    normalized = value.strip().lower()
    if normalized not in {"phase4", "phase5", "oracle", "all"}:
        raise ValueError(
            f"Unsupported phase '{value}'. Expected one of: phase4, phase5, oracle, all."
        )
    return normalized
