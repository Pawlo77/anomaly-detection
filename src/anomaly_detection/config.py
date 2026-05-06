"""Typed configuration models for experiment and datasets pipelines."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

TRACKING_URI_MLRUNS_ALIAS = "mlruns"
"""Backward-compatible shorthand for local MLflow storage."""

TRACKING_URI_SQLITE_DEFAULT = "sqlite:///mlruns.db"
"""Default SQLite-backed MLflow tracking URI."""

TRACKING_EXPERIMENT_DEFAULT = "anomaly_detection"
"""Default MLflow experiment name when settings do not override it."""

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

ORCHESTRATION_ENV_PREFIX = "ANOMALY_ORCH_"
"""Environment-variable prefix for orchestration runtime settings."""

MLFLOW_ENV_PREFIX = "ANOMALY_MLFLOW_"
"""Environment-variable prefix for MLflow runtime settings."""


class ConfigValidationError(ValueError):
    """Raised when a generic configuration value is invalid."""


class DatasetConfigError(ValueError):
    """Raised when dataset configuration values are invalid."""


def _require(condition: bool, message: str) -> None:
    """Assert predicate for configuration payloads.

    Args:
        condition: Expression that must evaluate true.
        message: Human-readable failure detail.

    Raises:
        ConfigValidationError: If ``condition`` is false.
    """
    if not condition:
        raise ConfigValidationError(message)


def _require_str(name: str, value: str) -> None:
    """Ensure a configuration string is syntactically usable.

    Args:
        name: Field label used in errors.
        value: Candidate string value.

    Raises:
        ConfigValidationError: When empty or non-string after trimming.
    """
    _require(isinstance(value, str), f"{name} must be a string.")
    _require(bool(value.strip()), f"{name} must not be empty.")


def _serialize(value: Any) -> Any:
    """Convert nested dataclasses and containers into JSON-friendly values.

    Args:
        value: Arbitrary nested structure rooted in primitives or dataclasses.

    Returns:
        Recursive structure using only lists, dicts, and primitives.
    """
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
    """Validate structured input for pydantic/from_dict loaders.

    Args:
        data: Candidate mapping body.
        name: Configuration object label for diagnostics.

    Returns:
        The same mapping view after validation.

    Raises:
        ConfigValidationError: If ``data`` is not a mapping.
    """
    _require(isinstance(data, Mapping), f"{name} must be a mapping.")
    return data


def _parse_bool(value: str) -> bool:
    """Parse a strict boolean literal from environment text.

    Args:
        value: Raw string token from env configuration.

    Returns:
        Parsed boolean.

    Raises:
        DatasetConfigError: If no supported keyword matches ``value``.
    """
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise DatasetConfigError(f"Cannot parse boolean from: {value}")


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    """Parse comma-separated integers into an immutable tuple.

    Args:
        value: Possibly empty comma-separated numeric list.

    Returns:
        Parsed tuple, empty when ``value`` is blank.
    """
    if not value.strip():
        return ()
    return tuple(int(token.strip()) for token in value.split(",") if token.strip())


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    """Parse comma-separated floats into an immutable tuple.

    Args:
        value: Possibly empty comma-separated float list.

    Returns:
        Parsed tuple, empty when ``value`` is blank.
    """
    if not value.strip():
        return ()
    return tuple(float(token.strip()) for token in value.split(",") if token.strip())


def _parse_caps(value: str) -> dict[str, int]:
    """Parse algorithm throughput caps serialized as comma pairs.

    Args:
        value: String shaped like ``"OCSVM:20000,LOF:20000"``.

    Returns:
        Mapping algorithm name to nonnegative integer caps.
    """
    caps: dict[str, int] = {}
    if not value.strip():
        return caps
    for part in value.split(","):
        if not part.strip():
            continue
        key, raw = part.split(":", 1)
        caps[key.strip()] = int(raw.strip())
    return caps


class MlflowSettings(BaseSettings):
    """Unified MLflow tracking settings.

    Attributes:
        enabled: Whether MLflow integration is enabled.
        tracking_uri: MLflow tracking backend URI.
        experiment_name: Default MLflow experiment name.
        experiment_name_phase4: MLflow experiment name for phase-4 tasks.
        experiment_name_phase5: MLflow experiment name for phase-5 tasks.
        run_name: Optional explicit run name.
        log_params: Whether to log run parameters.
        log_metrics: Whether to log metrics.
        log_artifacts: Whether to upload artifacts.
        retain_local_checkpoints: Whether to keep checkpoints after upload.
        heartbeat_metric_key: Metric key used for scheduler heartbeat writes.
        delete_incomplete_runs: Whether failed/killed runs are soft-deleted.
    """

    model_config = SettingsConfigDict(
        env_prefix=MLFLOW_ENV_PREFIX,
        extra="ignore",
        frozen=True,
    )

    enabled: bool = True
    tracking_uri: str = TRACKING_URI_SQLITE_DEFAULT
    experiment_name: str = TRACKING_EXPERIMENT_DEFAULT
    experiment_name_phase4: str = "anomaly_detection/phase4"
    experiment_name_phase5: str = "anomaly_detection/phase5"
    run_name: str | None = None
    log_params: bool = True
    log_metrics: bool = True
    log_artifacts: bool = True
    retain_local_checkpoints: bool = False
    heartbeat_metric_key: str = "system.heartbeat_unix"
    delete_incomplete_runs: bool = True

    @model_validator(mode="after")
    def validate_settings(self) -> "MlflowSettings":
        """Validate coherence of mutually dependent MLflow fields.

        Returns:
            The unchanged settings instance when validation succeeds.
        """
        if self.enabled:
            _require_str("tracking_uri", self.tracking_uri)
        _require_str("experiment_name", self.experiment_name)
        if self.run_name is not None:
            _require_str("run_name", self.run_name)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize settings for JSON/logging consumers.

        Returns:
            Primitive mapping equivalent to pydantic dump mode ``json``.
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Hydrate MLflow configuration from unstructured input.

        Args:
            data: Mapping containing pydantic-compatible keys.

        Returns:
            Validated immutable settings snapshot.
        """
        mapping = _extract_mapping(data, name="MlflowSettings")
        if mapping.get("tracking_uri") == TRACKING_URI_MLRUNS_ALIAS:
            mapping = {**mapping, "tracking_uri": TRACKING_URI_SQLITE_DEFAULT}
        return cls.model_validate(dict(mapping))


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Top-level experiment configuration.

    Attributes:
        random_seed: Base random seed used across the project.
        tracking: Nested MLflow settings.
    """

    random_seed: int = 42
    tracking: MlflowSettings = field(default_factory=MlflowSettings)

    def __post_init__(self) -> None:
        _require(self.random_seed >= 0, "random_seed must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize experiment-wide settings for bookkeeping.

        Returns:
            Recursive JSON-compatible mapping.
        """
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Construct experiment configuration from dictionaries.

        Args:
            data: Mapping containing nested ``tracking`` payloads when needed.

        Returns:
            Parsed dataclass wired with nested pydantic MLflow objects.
        """
        mapping = _extract_mapping(data, name="ExperimentConfig")
        payload = dict(mapping)
        tracking_raw = payload.get("tracking")
        if isinstance(tracking_raw, Mapping):
            payload["tracking"] = MlflowSettings.from_dict(tracking_raw)
        return cls(**payload)


class OrchestrationSettings(BaseSettings):
    """Runtime settings for experiment orchestration.

    Attributes:
        phase: Requested orchestration phase (`phase4`, `oracle`, `phase5`, or `all`).
        jobs: Maximum number of concurrent worker processes.
        stale_ttl_seconds: Heartbeat TTL for stale running-run detection.
        heavy_max_concurrent: Maximum number of heavy tasks run concurrently.
        fail_on_duplicate_running: Whether duplicate RUNNING runs should raise.
    """

    model_config = SettingsConfigDict(
        env_prefix=ORCHESTRATION_ENV_PREFIX,
        extra="ignore",
        frozen=True,
    )

    phase: str = Field(default="all")
    jobs: int = Field(default=4, ge=1)
    stale_ttl_seconds: int = Field(default=600, ge=5)
    heavy_max_concurrent: int = Field(default=2, ge=1)
    fail_on_duplicate_running: bool = Field(default=True)


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

    root_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data") / "raw"
    canonical_dir: Path = Path("data") / "canonical"
    processed_dir: Path = Path("data") / "processed"
    subsamples_dir: Path = Path("data") / "processed" / "subsamples"
    pca_dir: Path = Path("data") / "processed" / "pca"
    preprocess_reports_dir: Path = Path("outputs") / "preprocess"
    stats_dir: Path = Path("outputs") / "stats"
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
        """Construct dataset defaults reading ``ANOMALY_DATASETS_*`` environment variables.

        Returns:
            Hydrated immutable settings respecting present environment keys.
        """
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
        """Resolve possibly relative filesystem paths against ``root_dir``.

        Args:
            target: Candidate filesystem path literal.

        Returns:
            Absolute path rooted at ``root_dir`` when ``target`` is relative.
        """
        if target.is_absolute():
            return target
        return (self.root_dir / target).resolve()
