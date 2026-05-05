"""Typed configuration models for experiment and datasets pipelines."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Self

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRACKING_URI_MLRUNS_ALIAS = "mlruns"
"""Backward-compatible shorthand for local MLflow storage."""

TRACKING_URI_SQLITE_DEFAULT = "sqlite:///mlruns.db"
"""Default SQLite-backed MLflow tracking URI."""

TRACKING_EXPERIMENT_DEFAULT = "speech-recognition"
"""Default MLflow experiment name."""

DATASETS_ENV_PREFIX = "ANOMALY_DATASETS_"
"""Environment prefix for dataset settings."""

DEFAULT_DATASET_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 42)
"""Default random seeds used in dataset experiments."""

DEFAULT_DATASET_ALGORITHM_CAPS: dict[str, int] = {
    "OCSVM": 20_000,
    "LOF": 20_000,
    "DBSCAN": 20_000,
}
"""Maximum sample size for n^2-ish models."""

DEFAULT_DATASET_PCA_TARGETS: tuple[float, ...] = (
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.99,
)
"""Default PCA variance ladder for persisted dataset artifacts."""

DEFAULT_DATASET_MIN_ROW_RATIO = 0.9
"""Minimum acceptable ratio of actual rows to expected rows."""

DEFAULT_DATASET_MIN_FEATURE_RATIO = 0.9
"""Minimum acceptable ratio of actual features to expected features."""


class ConfigValidationError(ValueError):
    """Raised when a generic configuration value is invalid."""


class DatasetConfigError(ValueError):
    """Raised when dataset configuration values are invalid."""


def _require(condition: bool, message: str) -> None:
    """Raise a config validation error when condition is false."""
    if not condition:
        raise ConfigValidationError(message)


def _require_str(name: str, value: str) -> None:
    """Validate that a value is a non-empty string."""
    _require(isinstance(value, str), f"{name} must be a string.")
    _require(bool(value.strip()), f"{name} must not be empty.")


def _serialize(value: Any) -> Any:
    """Convert nested dataclasses and containers into JSON-friendly values."""
    if is_dataclass(value):
        return {field.name: _serialize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


def _extract_mapping(data: Mapping[str, Any] | Any, name: str) -> Mapping[str, Any]:
    """Validate and return mapping input for `from_dict` helpers."""
    _require(isinstance(data, Mapping), f"{name} must be a mapping.")
    return data


def _parse_bool(value: str) -> bool:
    """Parse strict boolean from environment string."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise DatasetConfigError(f"Cannot parse boolean from: {value}")


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    """Parse comma-separated integer list into tuple."""
    if not value.strip():
        return ()
    return tuple(int(token.strip()) for token in value.split(",") if token.strip())


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    """Parse comma-separated float list into tuple."""
    if not value.strip():
        return ()
    return tuple(float(token.strip()) for token in value.split(",") if token.strip())


def _parse_caps(value: str) -> dict[str, int]:
    """Parse algorithm cap map from `name:value,name:value` string."""
    caps: dict[str, int] = {}
    if not value.strip():
        return caps
    for part in value.split(","):
        if not part.strip():
            continue
        key, raw = part.split(":", 1)
        caps[key.strip()] = int(raw.strip())
    return caps


@dataclass(frozen=True, slots=True)
class MLflowTrackingConfig:
    """Local MLflow tracking configuration.

    Attributes:
        enabled: Whether MLflow integration is enabled.
        tracking_uri: MLflow tracking backend URI.
        experiment_name: MLflow experiment name.
        run_name: Optional explicit run name.
        log_params: Whether to log run parameters.
        log_metrics: Whether to log metrics.
        log_artifacts: Whether to upload artifacts.
        retain_local_checkpoints: Whether to keep checkpoints after upload.
    """

    enabled: bool = True
    tracking_uri: str = TRACKING_URI_SQLITE_DEFAULT
    experiment_name: str = TRACKING_EXPERIMENT_DEFAULT
    run_name: str | None = None
    log_params: bool = True
    log_metrics: bool = True
    log_artifacts: bool = True
    retain_local_checkpoints: bool = False

    def __post_init__(self) -> None:
        if self.enabled:
            _require_str("tracking_uri", self.tracking_uri)
        _require_str("experiment_name", self.experiment_name)
        if self.run_name is not None:
            _require_str("run_name", self.run_name)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of tracking config."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build tracking config from a mapping."""
        mapping = _extract_mapping(data, name="MLflowTrackingConfig")
        if mapping.get("tracking_uri") == TRACKING_URI_MLRUNS_ALIAS:
            mapping = {**mapping, "tracking_uri": TRACKING_URI_SQLITE_DEFAULT}
        return cls(**dict(mapping))


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Top-level experiment configuration.

    Attributes:
        random_seed: Base random seed used across the project.
        tracking: Nested MLflow tracking configuration.
    """

    random_seed: int = 42
    tracking: MLflowTrackingConfig = field(default_factory=MLflowTrackingConfig)

    def __post_init__(self) -> None:
        _require(self.random_seed >= 0, "random_seed must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of experiment config."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build experiment config from a mapping."""
        mapping = _extract_mapping(data, name="ExperimentConfig")
        payload = dict(mapping)
        tracking_raw = payload.get("tracking")
        if isinstance(tracking_raw, Mapping):
            payload["tracking"] = MLflowTrackingConfig.from_dict(tracking_raw)
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    """Configuration for datasets pipeline.

    Attributes:
        root_dir: Workspace root used to resolve relative paths.
        data_dir: Top-level data directory.
        raw_dir: Directory with downloaded raw artifacts.
        canonical_dir: Directory with canonicalized CSV datasets.
        processed_dir: Directory with processed datasets.
        subsamples_dir: Directory with stratified subsample artifacts.
        pca_dir: Directory with PCA-transformed datasets.
        preprocess_reports_dir: Directory with preprocessing reports.
        stats_dir: Directory with descriptive statistics outputs.
        random_seeds: Seeds used for reproducible subsampling and stats.
        n2_algorithm_caps: Max sample caps for n^2-style algorithms.
        workers: Number of parallel workers for pipeline steps.
        strict_validate: Whether validation failures should stop execution.
        arrhythmia_pca_variance: PCA variance target for arrhythmia flow.
        download_timeout_seconds: Per-download HTTP timeout.
        contamination_tolerance: Allowed contamination delta in validation.
        pca_variance_targets: Variance ladder for persisted PCA artifacts.
        min_row_ratio: Minimum acceptable row ratio for validation.
        min_feature_ratio: Minimum acceptable feature ratio for validation.
    """

    root_dir: Path = field(default_factory=Path.cwd)
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    canonical_dir: Path = Path("data/canonical")
    processed_dir: Path = Path("data/processed")
    subsamples_dir: Path = Path("data/processed/subsamples")
    pca_dir: Path = Path("data/processed/pca")
    preprocess_reports_dir: Path = Path("outputs/preprocess")
    stats_dir: Path = Path("outputs")
    random_seeds: tuple[int, ...] = DEFAULT_DATASET_SEEDS
    n2_algorithm_caps: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_DATASET_ALGORITHM_CAPS)
    )
    workers: int = 1
    strict_validate: bool = True
    arrhythmia_pca_variance: float = 0.95
    download_timeout_seconds: int = 120
    contamination_tolerance: float = 0.05
    pca_variance_targets: tuple[float, ...] = DEFAULT_DATASET_PCA_TARGETS
    min_row_ratio: float = DEFAULT_DATASET_MIN_ROW_RATIO
    min_feature_ratio: float = DEFAULT_DATASET_MIN_FEATURE_RATIO

    def __post_init__(self) -> None:
        if self.workers <= 0:
            raise DatasetConfigError("workers must be > 0")
        if self.download_timeout_seconds <= 0:
            raise DatasetConfigError("download_timeout_seconds must be > 0")
        if not (0.0 < self.arrhythmia_pca_variance <= 1.0):
            raise DatasetConfigError("arrhythmia_pca_variance must be in (0, 1]")
        if not (0.0 <= self.contamination_tolerance <= 1.0):
            raise DatasetConfigError("contamination_tolerance must be in [0, 1]")
        if not self.random_seeds:
            raise DatasetConfigError("random_seeds must not be empty")
        if not self.pca_variance_targets:
            raise DatasetConfigError("pca_variance_targets must not be empty")
        for value in self.pca_variance_targets:
            if not (0.0 < value <= 1.0):
                raise DatasetConfigError("pca_variance_targets values must be in (0, 1]")
        if not (0.0 < self.min_row_ratio <= 1.0):
            raise DatasetConfigError("min_row_ratio must be in (0, 1]")
        if not (0.0 < self.min_feature_ratio <= 1.0):
            raise DatasetConfigError("min_feature_ratio must be in (0, 1]")

    @classmethod
    def from_env(cls) -> "DatasetSettings":
        """Create dataset settings from environment variables."""
        prefix = DATASETS_ENV_PREFIX
        values: dict[str, object] = {}
        env = os.environ

        path_fields = {
            "ROOT_DIR": "root_dir",
            "DATA_DIR": "data_dir",
            "RAW_DIR": "raw_dir",
            "CANONICAL_DIR": "canonical_dir",
            "PROCESSED_DIR": "processed_dir",
            "SUBSAMPLES_DIR": "subsamples_dir",
            "PCA_DIR": "pca_dir",
            "PREPROCESS_REPORTS_DIR": "preprocess_reports_dir",
            "STATS_DIR": "stats_dir",
        }
        for env_name, field_name in path_fields.items():
            raw = env.get(f"{prefix}{env_name}")
            if raw is not None:
                values[field_name] = Path(raw)

        int_fields = {
            "WORKERS": "workers",
            "DOWNLOAD_TIMEOUT_SECONDS": "download_timeout_seconds",
        }
        for env_name, field_name in int_fields.items():
            raw = env.get(f"{prefix}{env_name}")
            if raw is not None:
                values[field_name] = int(raw)

        bool_raw = env.get(f"{prefix}STRICT_VALIDATE")
        if bool_raw is not None:
            values["strict_validate"] = _parse_bool(bool_raw)

        float_fields = {
            "ARRHYTHMIA_PCA_VARIANCE": "arrhythmia_pca_variance",
            "CONTAMINATION_TOLERANCE": "contamination_tolerance",
            "MIN_ROW_RATIO": "min_row_ratio",
            "MIN_FEATURE_RATIO": "min_feature_ratio",
        }
        for env_name, field_name in float_fields.items():
            raw = env.get(f"{prefix}{env_name}")
            if raw is not None:
                values[field_name] = float(raw)

        seeds_raw = env.get(f"{prefix}RANDOM_SEEDS")
        if seeds_raw is not None:
            values["random_seeds"] = _parse_int_tuple(seeds_raw)

        caps_raw = env.get(f"{prefix}N2_ALGORITHM_CAPS")
        if caps_raw is not None:
            values["n2_algorithm_caps"] = _parse_caps(caps_raw)

        pca_targets_raw = env.get(f"{prefix}PCA_VARIANCE_TARGETS")
        if pca_targets_raw is not None:
            values["pca_variance_targets"] = _parse_float_tuple(pca_targets_raw)

        return cls(**values)

    def resolve(self, target: Path) -> Path:
        """Resolve path relative to root directory."""
        if target.is_absolute():
            return target
        return (self.root_dir / target).resolve()
