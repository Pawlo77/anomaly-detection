"""End-to-end dataset build runner (download→canonical→preprocess→validate)."""

import json
import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict
from functools import partial

import pandas as pd
from tqdm import tqdm

from ..config import DatasetSettings
from .canonicalize import Canonicalizer
from .catalog import DatasetCatalog, build_default_catalog
from .downloaders import DatasetDownloader
from .loader import DatasetLoader
from .pca import pca_by_variance_with_metadata
from .preprocess import preprocess_arrhythmia_with_report, robust_scale_features
from .validate import validate_canonical_artifacts, write_validation_report

RAW_METADATA_SUFFIX = ".meta.json"
"""Suffix for raw artifact metadata."""

CANONICAL_METADATA_SUFFIX = ".canonical.json"
"""Suffix for canonicalization metadata."""

LABEL_COLUMN_NAME = "label"
"""Canonical binary label column name."""

ARRHYTHMIA_DATASET_ID = "arrhythmia"
"""Dataset id for arrhythmia-specific preprocessing branch."""

LOGGER = logging.getLogger(__name__)
"""Module logger for datasets pipeline status reporting."""


def run_datasets_pipeline(
    settings: DatasetSettings | None = None,
    catalog: DatasetCatalog | None = None,
) -> None:
    """Execute canonical download, preprocessing, PCA, subsample, and QA stages.

    Args:
        settings: Optional settings override; defaults to ``DatasetSettings.from_env``.
        catalog: Optional catalog override, mostly exercised from unit tests.
    """
    cfg = settings or DatasetSettings.from_env()
    active_catalog = catalog or build_default_catalog()
    downloader = DatasetDownloader(settings=cfg)
    n_ds = len(active_catalog.specs)
    LOGGER.info("Starting datasets pipeline for %d datasets", n_ds)

    _create_directories(cfg)
    LOGGER.info("Directories prepared")

    stages: list[tuple[str, Callable[[], None]]] = [
        ("download", partial(_stage_download, active_catalog, cfg, downloader)),
        ("canonicalize", partial(_stage_canonicalize, active_catalog, cfg)),
        ("preprocess_pca", partial(_stage_preprocess_artifacts, active_catalog, cfg)),
        ("subsamples", partial(_stage_subsamples, active_catalog, cfg)),
        ("descriptive_stats", partial(_stage_descriptive_stats, cfg, active_catalog)),
        ("validate", partial(_stage_validate, cfg, active_catalog)),
    ]
    for stage_name, run_stage in tqdm(
        stages,
        desc="Datasets pipeline",
        unit="stage",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        LOGGER.info("Stage start: %s (%d datasets)", stage_name, n_ds)
        run_stage()
        LOGGER.info("Stage done: %s", stage_name)

    LOGGER.info("Datasets pipeline finished successfully")


def _create_directories(settings: DatasetSettings) -> None:
    """Create required directories for all stage outputs."""
    settings.resolve(settings.raw_dir).mkdir(parents=True, exist_ok=True)
    settings.resolve(settings.canonical_dir).mkdir(parents=True, exist_ok=True)
    settings.resolve(settings.stats_dir).mkdir(parents=True, exist_ok=True)
    settings.resolve(settings.subsamples_dir).mkdir(parents=True, exist_ok=True)
    settings.resolve(settings.pca_dir).mkdir(parents=True, exist_ok=True)
    settings.resolve(settings.preprocess_reports_dir).mkdir(parents=True, exist_ok=True)


def _stage_download(
    catalog: DatasetCatalog, settings: DatasetSettings, downloader: DatasetDownloader
) -> None:
    """Download raw datasets for each catalog spec."""
    for spec in _progress(catalog.specs, description="Download"):
        result = downloader.download(spec, settings.resolve(settings.raw_dir))
        metadata_path = settings.resolve(
            settings.raw_dir / f"{spec.dataset_id.replace('/', '__')}{RAW_METADATA_SUFFIX}"
        )
        metadata_path.write_text(
            json.dumps(
                {
                    "dataset_id": spec.dataset_id,
                    "source_type": spec.source_type,
                    "source_ref": spec.source_ref,
                    "raw_artifact": str(result.path),
                    "download_backend": result.backend_used,
                    "sha256": result.sha256,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _stage_canonicalize(catalog: DatasetCatalog, settings: DatasetSettings) -> None:
    """Materialize canonical artifacts from raw files."""
    canonicalizer = Canonicalizer()
    for spec in _progress(catalog.specs, description="Canonicalize"):
        raw_path = settings.resolve(settings.raw_dir / f"{spec.dataset_id.replace('/', '__')}.raw")
        output_path = settings.resolve(
            settings.canonical_dir / f"{spec.dataset_id.replace('/', '__')}.csv"
        )
        result = canonicalizer.canonicalize(spec=spec, raw_path=raw_path, output_path=output_path)
        meta_path = settings.resolve(
            settings.canonical_dir
            / f"{spec.dataset_id.replace('/', '__')}{CANONICAL_METADATA_SUFFIX}"
        )
        meta_path.write_text(
            json.dumps(
                {
                    "dataset_id": result.dataset_id,
                    "output_path": str(result.output_path),
                    "rows": result.rows,
                    "features": result.features,
                    "parser_used": result.parser_used,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _stage_subsamples(catalog: DatasetCatalog, settings: DatasetSettings) -> None:
    """Create preprocessed stratified subsamples for n^2 algorithms."""
    loader = DatasetLoader(settings=settings, catalog=catalog)
    for spec in _progress(catalog.specs, description="Subsamples"):
        if not spec.stratified_subsample_for_n2:
            continue
        for algorithm in settings.n2_algorithm_caps:
            for seed in settings.random_seeds:
                bundle = loader.load_subsample(spec.dataset_id, algorithm=algorithm, seed=seed)
                frame = bundle.X.copy()
                frame[LABEL_COLUMN_NAME] = bundle.y
                out = (
                    settings.resolve(settings.subsamples_dir)
                    / f"{spec.dataset_id.replace('/', '__')}__{algorithm.lower()}__seed{seed}.csv"
                )
                frame.to_csv(out, index=False)


def _stage_preprocess_artifacts(catalog: DatasetCatalog, settings: DatasetSettings) -> None:
    """Persist arrhythmia report and PCA ladder artifacts."""
    for spec in _progress(catalog.specs, description="Preprocess/PCA"):
        source = settings.resolve(
            settings.canonical_dir / f"{spec.dataset_id.replace('/', '__')}.csv"
        )
        if not source.exists():
            continue
        frame = pd.read_csv(source)
        if LABEL_COLUMN_NAME not in frame.columns:
            continue
        x_values = frame.drop(columns=[LABEL_COLUMN_NAME])
        y = frame[LABEL_COLUMN_NAME].astype(int)
        if spec.dataset_id == ARRHYTHMIA_DATASET_ID:
            transformed, report = preprocess_arrhythmia_with_report(x_values)
            report_path = settings.resolve(
                settings.preprocess_reports_dir
                / f"{spec.dataset_id.replace('/', '__')}__preprocess.json"
            )
            report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
            base = transformed
        else:
            base = robust_scale_features(x_values)
        variance_targets = list(settings.pca_variance_targets)
        if spec.dataset_id == ARRHYTHMIA_DATASET_ID and not any(
            abs(target - settings.arrhythmia_pca_variance) < 1e-12 for target in variance_targets
        ):
            variance_targets.append(settings.arrhythmia_pca_variance)
        for variance in variance_targets:
            _persist_pca_artifact(
                settings=settings, dataset_id=spec.dataset_id, base=base, y=y, variance=variance
            )


def _persist_pca_artifact(
    settings: DatasetSettings,
    dataset_id: str,
    base: pd.DataFrame,
    y: pd.Series,
    variance: float,
) -> None:
    """Persist one PCA projection CSV and JSON summary."""
    pca_result = pca_by_variance_with_metadata(base, variance_ratio=variance, seed=42)
    out = (
        settings.resolve(settings.pca_dir)
        / f"{dataset_id.replace('/', '__')}__pca_{int(variance * 100):02d}.csv"
    )
    table = pca_result.transformed.copy()
    table[LABEL_COLUMN_NAME] = y.values
    table.to_csv(out, index=False)
    summary_path = (
        settings.resolve(settings.pca_dir)
        / f"{dataset_id.replace('/', '__')}__pca_{int(variance * 100):02d}.json"
    )
    summary_path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "variance_target": variance,
                "n_components": pca_result.n_components,
                "explained_variance_ratio_sum": pca_result.explained_variance_ratio_sum,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _stage_descriptive_stats(settings: DatasetSettings, catalog: DatasetCatalog) -> None:
    """Compute and persist descriptive stats for report."""
    loader = DatasetLoader(settings=settings, catalog=catalog)
    stats_rows: list[dict[str, float | int | str]] = []
    for dataset_id in _progress(loader.list_datasets(), description="Descriptive stats"):
        summary = loader.stats(dataset_id)
        stats_rows.append(
            {
                "dataset_id": dataset_id,
                "n_rows": summary.n_rows,
                "n_features": summary.n_features,
                "contamination": summary.contamination,
                "missing_fraction": summary.missing_fraction,
                "max_feature_missing_fraction": summary.max_feature_missing_fraction,
                "outlier_count": summary.outlier_count,
                "inlier_count": summary.inlier_count,
                "mean_abs_correlation": summary.mean_abs_correlation,
                "distance_concentration_ratio": summary.distance_concentration_ratio,
            }
        )
    pd.DataFrame(stats_rows).to_csv(
        settings.resolve(settings.stats_dir / "descriptive_stats.csv"), index=False
    )


def _stage_validate(settings: DatasetSettings, catalog: DatasetCatalog) -> None:
    """Validate canonical artifacts and emit quality report."""
    rows, failures = validate_canonical_artifacts(settings=settings, catalog=catalog)
    # Keep both paths for backward compatibility: legacy consumers read
    # outputs/validation_report.csv, while newer reporting uses outputs/stats/.
    write_validation_report(rows, settings.resolve(settings.stats_dir / "validation_report.csv"))
    write_validation_report(
        rows, settings.resolve(settings.data_dir.parent / "outputs/validation_report.csv")
    )
    if failures and settings.strict_validate:
        raise ValueError(f"Validation failed for datasets: {', '.join(failures)}")


def _progress[ItemT](items: Iterable[ItemT], description: str) -> Iterator[ItemT]:
    """Wrap iterable with tqdm progress bar."""
    return tqdm(
        items,
        desc=description,
        unit="dataset",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.3,
    )
