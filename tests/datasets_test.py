import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anomaly_detection.config import DatasetSettings
from anomaly_detection.datasets.canonicalize import Canonicalizer, _normalize_label
from anomaly_detection.datasets.catalog import DatasetCatalog, DatasetSpec
from anomaly_detection.datasets.downloaders import DownloadError
from anomaly_detection.datasets.loader import DatasetLoader
from anomaly_detection.datasets.pipeline import run_datasets_pipeline
from anomaly_detection.datasets.sampling import stratified_subsample


def test_stratified_subsample_preserves_labels() -> None:
    frame = pd.DataFrame(
        {
            "f0": list(range(100)),
            "label": [0] * 90 + [1] * 10,
        }
    )
    sampled = stratified_subsample(frame, label_column="label", n_samples=40, seed=42)
    assert len(sampled) == 40
    ratio = sampled["label"].mean()
    assert 0.05 <= ratio <= 0.2


def _mini_catalog() -> DatasetCatalog:
    return DatasetCatalog(
        specs=(
            DatasetSpec(
                dataset_id="arrhythmia",
                source_type="requests",
                source_ref="arrhythmia",
                expected_rows=10,
                expected_dim=2,
                contamination=0.2,
            ),
            DatasetSpec(
                dataset_id="cover",
                source_type="requests",
                source_ref="cover",
                expected_rows=10,
                expected_dim=2,
                contamination=0.2,
                stratified_subsample_for_n2=True,
            ),
        )
    )


def _write_raw_zip(path: Path, frame: pd.DataFrame, filename: str = "data.csv") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_csv(index=False).encode("utf-8")
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(filename, payload)


def test_dataset_pipeline_generates_stats(tmp_path: Path) -> None:
    settings = DatasetSettings(
        root_dir=tmp_path,
        data_dir=Path("data"),
        raw_dir=Path("data/raw"),
        canonical_dir=Path("data/canonical"),
        processed_dir=Path("data/processed"),
        stats_dir=Path("data/stats"),
    )
    catalog = _mini_catalog()
    raw_dir = tmp_path / "data/raw"
    sample = pd.DataFrame(
        {"f0": range(10), "f1": range(10, 20), "y": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]}
    )
    _write_raw_zip(raw_dir / "arrhythmia.raw", sample)
    _write_raw_zip(raw_dir / "cover.raw", sample)
    run_datasets_pipeline(settings=settings, catalog=catalog)
    stats_file = tmp_path / "data/stats/descriptive_stats.csv"
    assert stats_file.exists()
    table = pd.read_csv(stats_file)
    assert not table.empty
    assert {"dataset_id", "n_rows", "n_features"}.issubset(table.columns)
    assert {"outlier_count", "inlier_count"}.issubset(table.columns)
    assert {"mean_abs_correlation", "distance_concentration_ratio"}.issubset(table.columns)


def test_loader_can_read_placeholder_artifacts(tmp_path: Path) -> None:
    canonical = tmp_path / "data/canonical"
    canonical.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"f0": [1.0, 2.0], "f1": [2.0, 3.0], "label": [0, 1]}).to_csv(
        canonical / "arrhythmia.csv", index=False
    )
    settings = DatasetSettings(root_dir=tmp_path, canonical_dir=Path("data/canonical"))
    loader = DatasetLoader(settings=settings)
    bundle = loader.load("arrhythmia", view="preprocessed")
    assert bundle.X.shape[0] == 2
    assert set(bundle.y.unique()).issubset({0, 1})


def test_pipeline_fails_download_without_backends(tmp_path: Path) -> None:
    settings = DatasetSettings(root_dir=tmp_path)
    failing_catalog = DatasetCatalog(
        specs=(
            DatasetSpec(
                dataset_id="broken",
                source_type="requests",
                source_ref="not-a-url",
                expected_rows=10,
                expected_dim=2,
                contamination=0.1,
            ),
        )
    )
    with pytest.raises(DownloadError):
        run_datasets_pipeline(settings=settings, catalog=failing_catalog)


def test_pipeline_writes_subsamples(tmp_path: Path) -> None:
    settings = DatasetSettings(root_dir=tmp_path)
    catalog = _mini_catalog()
    sample = pd.DataFrame(
        {"f0": range(10), "f1": range(10, 20), "y": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]}
    )
    _write_raw_zip(tmp_path / "data/raw/arrhythmia.raw", sample)
    _write_raw_zip(tmp_path / "data/raw/cover.raw", sample)
    run_datasets_pipeline(settings=settings, catalog=catalog)
    subsamples_dir = tmp_path / "data/processed/subsamples"
    files = list(subsamples_dir.glob("*.csv"))
    assert files


def test_pipeline_writes_validation_report(tmp_path: Path) -> None:
    settings = DatasetSettings(root_dir=tmp_path, strict_validate=False)
    catalog = _mini_catalog()
    sample = pd.DataFrame(
        {"f0": range(10), "f1": range(10, 20), "y": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]}
    )
    _write_raw_zip(tmp_path / "data/raw/arrhythmia.raw", sample)
    _write_raw_zip(tmp_path / "data/raw/cover.raw", sample)
    run_datasets_pipeline(settings=settings, catalog=catalog)
    report = tmp_path / "outputs/validation_report.csv"
    assert report.exists()
    table = pd.read_csv(report)
    assert not table.empty
    assert {"dataset_id", "ok", "contamination_abs_diff"}.issubset(table.columns)


def test_label_normalization_auto_inverts_when_needed() -> None:
    frame = pd.DataFrame(
        {
            "f0": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "y": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        }
    )
    spec = DatasetSpec(
        dataset_id="arrhythmia",
        source_type="requests",
        source_ref="arrhythmia",
        expected_rows=10,
        expected_dim=1,
        contamination=0.8,
    )
    normalized = _normalize_label(frame, spec)
    assert "y" not in normalized.columns
    assert "label" in normalized.columns
    assert float(normalized["label"].mean()) == 0.8


def test_label_normalization_keeps_direct_orientation() -> None:
    frame = pd.DataFrame(
        {
            "f0": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "class": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        }
    )
    spec = DatasetSpec(
        dataset_id="arrhythmia",
        source_type="requests",
        source_ref="arrhythmia",
        expected_rows=10,
        expected_dim=1,
        contamination=0.2,
    )
    normalized = _normalize_label(frame, spec)
    assert float(normalized["label"].mean()) == 0.2


def test_label_normalization_multiclass_uses_expected_contamination() -> None:
    frame = pd.DataFrame(
        {
            "f0": list(range(20)),
            "class": [0] * 16 + [1] * 3 + [2] * 1,
        }
    )
    spec = DatasetSpec(
        dataset_id="wut/x2",
        source_type="requests",
        source_ref="https://example.com/x2.zip",
        expected_rows=20,
        expected_dim=1,
        contamination=0.05,
    )
    normalized = _normalize_label(frame, spec)
    assert float(normalized["label"].mean()) == 0.05


def test_label_normalization_zero_contamination_forces_inliers() -> None:
    frame = pd.DataFrame(
        {
            "f0": list(range(10)),
            "class": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
        }
    )
    spec = DatasetSpec(
        dataset_id="wut/smile",
        source_type="requests",
        source_ref="https://example.com/smile.zip",
        expected_rows=10,
        expected_dim=1,
        contamination=0.0,
    )
    normalized = _normalize_label(frame, spec)
    assert float(normalized["label"].mean()) == 0.0


def test_preprocess_stage_persists_reports_and_pca(tmp_path: Path) -> None:
    settings = DatasetSettings(root_dir=tmp_path, stats_dir=Path("outputs"))
    catalog = _mini_catalog()
    sample = pd.DataFrame(
        {"f0": range(10), "f1": range(10, 20), "y": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]}
    )
    _write_raw_zip(tmp_path / "data/raw/arrhythmia.raw", sample)
    _write_raw_zip(tmp_path / "data/raw/cover.raw", sample)
    run_datasets_pipeline(settings=settings, catalog=catalog)
    report_file = tmp_path / "outputs/preprocess/arrhythmia__preprocess.json"
    pca_file = tmp_path / "data/processed/pca/arrhythmia__pca_95.csv"
    assert report_file.exists()
    assert pca_file.exists()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert "imputed_cell_fraction" in report
    assert "imputed_fraction_by_feature" in report


def test_canonicalizer_parses_npz_payload_with_raw_suffix(tmp_path: Path) -> None:
    raw_path = tmp_path / "annthyroid.raw"
    with raw_path.open("wb") as handle:
        np.savez(handle, X=np.array([[1.0, 2.0], [2.0, 3.0]]), y=np.array([0, 1]))
    spec = DatasetSpec(
        dataset_id="annthyroid",
        source_type="requests",
        source_ref="https://example.com/annthyroid.npz",
        expected_rows=2,
        expected_dim=2,
        contamination=0.5,
    )
    output_path = tmp_path / "annthyroid.csv"
    result = Canonicalizer().canonicalize(spec=spec, raw_path=raw_path, output_path=output_path)
    assert result.parser_used == "npz"
    table = pd.read_csv(output_path)
    assert list(table.columns) == ["f0", "f1", "label"]
    assert set(table["label"].tolist()) == {0, 1}


def test_canonicalizer_parses_csv_payload_with_raw_suffix(tmp_path: Path) -> None:
    raw_path = tmp_path / "arrhythmia.raw"
    pd.DataFrame({"f0": [1.0, 2.0], "f1": [2.0, 3.0], "y": [0, 1]}).to_csv(raw_path, index=False)
    spec = DatasetSpec(
        dataset_id="arrhythmia",
        source_type="requests",
        source_ref="https://example.com/arrhythmia.csv",
        expected_rows=2,
        expected_dim=2,
        contamination=0.5,
    )
    output_path = tmp_path / "arrhythmia.csv"
    result = Canonicalizer().canonicalize(spec=spec, raw_path=raw_path, output_path=output_path)
    assert result.parser_used == "csv"
    table = pd.read_csv(output_path)
    assert list(table.columns) == ["f0", "f1", "label"]
    assert set(table["label"].tolist()) == {0, 1}
