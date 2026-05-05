"""Normalize heterogeneous raw downloads into unified CSV schemas with labels."""

import gzip
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import DatasetSpec

LABEL_COLUMN_NAME = "label"
"""Canonical binary label column name."""

RAW_DATA_SUFFIX = ".data.gz"
"""Gagolewski data file suffix inside archive."""

RAW_LABEL_SUFFIXES: tuple[str, ...] = (".labels1.gz", ".labels0.gz")
"""Preferred label files order inside clustering archive."""

ODDS_LABEL_COLUMN_BY_DATASET: dict[str, tuple[str, ...]] = {
    "annthyroid": ("y", "label", "class"),
    "arrhythmia": ("y", "label", "class"),
    "breastw": ("y", "label", "class"),
    "cardio": ("y", "label", "class"),
    "cover": ("y", "label", "class"),
    "glass": ("y", "label", "class"),
    "http": ("y", "label", "class"),
    "ionosphere": ("y", "label", "class"),
    "letter": ("y", "label", "class"),
    "lympho": ("y", "label", "class"),
    "mammography": ("y", "label", "class"),
    "mnist": ("y", "label", "class"),
    "musk": ("y", "label", "class"),
    "optdigits": ("y", "label", "class"),
    "pendigits": ("y", "label", "class"),
    "pima": ("y", "label", "class"),
    "satellite": ("y", "label", "class"),
    "satimage-2": ("y", "label", "class"),
    "shuttle": ("y", "label", "class"),
    "smtp": ("y", "label", "class"),
    "speech": ("y", "label", "class"),
    "thyroid": ("y", "label", "class"),
    "vertebral": ("y", "label", "class"),
    "vowels": ("y", "label", "class"),
    "wbc": ("y", "label", "class"),
    "wine": ("y", "label", "class"),
}
"""Dataset-specific label column priority map for ODDS-style datasets."""

GLOBAL_LABEL_COLUMNS: tuple[str, ...] = ("label", "y", "class", "target", "is_outlier")
"""Fallback label column priority for unknown dataset layout."""


@dataclass(frozen=True, slots=True)
class CanonicalizationResult:
    """Result payload for canonicalization step.

    Attributes:
        dataset_id: Dataset identifier.
        output_path: Path to canonicalized CSV output.
        rows: Number of rows in canonicalized output.
        features: Number of features (excluding label) in canonicalized output.
        parser_used: Name of parser that successfully processed raw artifact.
    """

    dataset_id: str
    output_path: Path
    rows: int
    features: int
    parser_used: str


class Canonicalizer:
    """Stateful adapter dispatching parsers per dataset lineage.

    The resulting tables always expose the ``label`` binary column enforced by loaders.
    """

    def __init__(self) -> None:
        """Instantiate dispatcher with lazily populated parser registry lookups."""

    def canonicalize(
        self, spec: DatasetSpec, raw_path: Path, output_path: Path
    ) -> CanonicalizationResult:
        """Parse ``raw_path`` streams and emit normalized CSV shards.

        Args:
            spec: Catalog row describing downloader targets and quirks.
            raw_path: On-disk footprint produced by ingest stage.
            output_path: Final canonical CSV destination.

        Returns:
            Lightweight summary describing artifact dimensions and parsers used.

        Raises:
            FileNotFoundError: When ``raw_path`` is unexpectedly missing locally.
            ValueError: When no parser understands the artifact bytes.
        """
        parser_name, frame = self._parse_raw(spec, raw_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        return CanonicalizationResult(
            dataset_id=spec.dataset_id,
            output_path=output_path,
            rows=frame.shape[0],
            features=frame.shape[1] - 1,
            parser_used=parser_name,
        )

    def _parse_raw(self, spec: DatasetSpec, raw_path: Path) -> tuple[str, pd.DataFrame]:
        """Parse raw artifact using registered parsers and normalize label column."""
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw artifact: {raw_path}")

        for parser_name, parser in _parser_registry(raw_path):
            parsed = parser(raw_path, spec)
            if parsed is not None:
                return parser_name, _normalize_label(parsed, spec)

        raise ValueError(f"Cannot parse raw artifact for {spec.dataset_id}: {raw_path}")


def _read_first_csv_from_zip(path: Path, _: DatasetSpec) -> pd.DataFrame | None:
    """Read first CSV file from ZIP archive, or return None if no CSV found."""
    with zipfile.ZipFile(path) as archive:
        csv_files = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_files:
            return None
        with archive.open(csv_files[0]) as handle:
            payload = handle.read()
            return pd.read_csv(io.BytesIO(payload))


def _read_csv(path: Path, _: DatasetSpec) -> pd.DataFrame | None:
    """Read CSV payload if file content is parseable as tabular text."""
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if frame.empty or frame.shape[1] < 2:
        return None
    return frame


def _read_npz(path: Path, _: DatasetSpec) -> pd.DataFrame | None:
    """Read ADBench-style npz artifact with X/y arrays."""
    try:
        with np.load(path, allow_pickle=False) as data:
            if "X" not in data or "y" not in data:
                return None
            features = np.asarray(data["X"])
            y = np.asarray(data["y"]).reshape(-1)
    except Exception:
        return None
    frame = pd.DataFrame(features, columns=[f"f{idx}" for idx in range(features.shape[1])])
    frame[LABEL_COLUMN_NAME] = y
    return frame


def _read_clustering_archive(path: Path, spec: DatasetSpec) -> pd.DataFrame | None:
    """Read one dataset from clustering-data-v1 archive (.data.gz + labels)."""
    if not zipfile.is_zipfile(path) or "/" not in spec.dataset_id:
        return None
    data_suffix = f"/{spec.dataset_id}{RAW_DATA_SUFFIX}"
    label_suffixes = [f"/{spec.dataset_id}{suffix}" for suffix in RAW_LABEL_SUFFIXES]
    with zipfile.ZipFile(path) as archive:
        data_member = next(
            (name for name in archive.namelist() if name.endswith(data_suffix)), None
        )
        if data_member is None:
            return None
        with archive.open(data_member) as handle:
            features = np.loadtxt(io.BytesIO(gzip.decompress(handle.read())))
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        y = np.zeros(features.shape[0], dtype=int)
        label_member = None
        for suffix in label_suffixes:
            label_member = next(
                (name for name in archive.namelist() if name.endswith(suffix)), None
            )
            if label_member is not None:
                break
        if label_member is not None:
            with archive.open(label_member) as handle:
                y = np.asarray(np.loadtxt(io.BytesIO(gzip.decompress(handle.read())))).reshape(-1)
        frame = pd.DataFrame(features, columns=[f"f{idx}" for idx in range(features.shape[1])])
        frame[LABEL_COLUMN_NAME] = y
        return frame


def _parser_registry(raw_path: Path) -> tuple[tuple[str, callable], ...]:
    """Registry of parsers to attempt for raw artifact, in order of priority."""
    parsers: list[tuple[str, callable]] = [
        ("npz", _read_npz),
        ("csv", _read_csv),
        ("clustering-archive", _read_clustering_archive),
    ]
    if zipfile.is_zipfile(raw_path):
        parsers.append(("zip-csv", _read_first_csv_from_zip))
    return tuple(parsers)


def _normalize_label(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Normalize raw label to binary 0/1 column named `label`."""
    label_column = _resolve_label_column(frame, spec.dataset_id)
    if label_column is not None:
        output = frame.copy()
        raw_labels = output[label_column]
        output = output.drop(columns=[label_column])
        output[LABEL_COLUMN_NAME] = _binarize_labels(
            raw_labels=raw_labels, expected_contamination=spec.contamination
        )
        return output
    raise ValueError(f"No label column found for dataset {spec.dataset_id}")


def _resolve_label_column(frame: pd.DataFrame, dataset_id: str) -> str | None:
    """Resolve label column name based on dataset-specific and global priority."""
    dataset_priority = ODDS_LABEL_COLUMN_BY_DATASET.get(dataset_id, ())
    candidates = (*dataset_priority, *GLOBAL_LABEL_COLUMNS)
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _binarize_labels(raw_labels: pd.Series, expected_contamination: float) -> pd.Series:
    """Binarize raw label values to 0/1 based on expected contamination and value distribution."""
    if expected_contamination <= 0.0:
        return pd.Series(np.zeros(len(raw_labels), dtype=int), index=raw_labels.index)

    values = pd.Series(raw_labels).dropna()
    if values.empty:
        return pd.Series(np.zeros(len(raw_labels), dtype=int), index=raw_labels.index)
    unique_values = sorted(values.unique().tolist())
    if len(unique_values) == 2:
        low, high = unique_values[0], unique_values[1]
        direct = (raw_labels == high).astype(int)
        inverted = (raw_labels == low).astype(int)
        direct_diff = abs(float(direct.mean()) - expected_contamination)
        inverted_diff = abs(float(inverted.mean()) - expected_contamination)
        return direct if direct_diff <= inverted_diff else inverted

    candidate_masks = [(raw_labels == value).astype(int) for value in unique_values]
    return min(
        candidate_masks,
        key=lambda candidate: abs(float(candidate.mean()) - expected_contamination),
    )
