"""Validation routines for canonical dataset artifacts."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import DatasetSettings
from .catalog import DatasetCatalog

LABEL_COLUMN_NAME = "label"
"""Canonical binary label column name."""


@dataclass(frozen=True, slots=True)
class ValidationRow:
    """Validation output for one dataset.

    Attributes:
        dataset_id: Dataset identifier.
        row_count: Number of rows in artifact.
        feature_count: Number of features in artifact (excluding label).
        contamination_actual: Contamination fraction in artifact.
        contamination_expected: Expected contamination from catalog.
        contamination_abs_diff: Absolute difference between actual and expected contamination.
        row_ratio: Ratio of actual rows to expected rows.
        feature_ratio: Ratio of actual features to expected features.
        ok: Whether artifact meets all validation criteria.
    """

    dataset_id: str
    row_count: int
    feature_count: int
    contamination_actual: float
    contamination_expected: float
    contamination_abs_diff: float
    row_ratio: float
    feature_ratio: float
    ok: bool


def validate_canonical_artifacts(
    settings: DatasetSettings,
    catalog: DatasetCatalog,
) -> tuple[list[ValidationRow], list[str]]:
    """Validate canonical files against catalog expectations.

    Args:
        settings: Runtime settings.
        catalog: Dataset specification catalog.

    Returns:
        Tuple of validation rows and list of failing dataset ids.
    """
    rows: list[ValidationRow] = []
    failures: list[str] = []
    for spec in catalog.specs:
        path = settings.resolve(
            settings.canonical_dir / f"{spec.dataset_id.replace('/', '__')}.csv"
        )
        if not path.exists():
            failures.append(spec.dataset_id)
            continue
        frame = pd.read_csv(path)
        if LABEL_COLUMN_NAME not in frame.columns:
            failures.append(spec.dataset_id)
            continue
        row_count = int(frame.shape[0])
        feature_count = int(frame.shape[1] - 1)
        contamination_actual = float(frame[LABEL_COLUMN_NAME].mean())
        contamination_abs_diff = abs(contamination_actual - spec.contamination)
        row_ratio = row_count / float(spec.expected_rows)
        feature_ratio = feature_count / float(spec.expected_dim)
        is_ok = (
            contamination_abs_diff <= settings.contamination_tolerance
            and row_ratio >= settings.min_row_ratio
            and feature_ratio >= settings.min_feature_ratio
            and set(frame[LABEL_COLUMN_NAME].dropna().unique()).issubset({0, 1})
        )
        if not is_ok:
            failures.append(spec.dataset_id)
        rows.append(
            ValidationRow(
                dataset_id=spec.dataset_id,
                row_count=row_count,
                feature_count=feature_count,
                contamination_actual=contamination_actual,
                contamination_expected=spec.contamination,
                contamination_abs_diff=contamination_abs_diff,
                row_ratio=row_ratio,
                feature_ratio=feature_ratio,
                ok=is_ok,
            )
        )
    return rows, failures


def write_validation_report(rows: list[ValidationRow], output_path: Path) -> None:
    """Write validation rows to CSV file."""
    table = pd.DataFrame(
        [
            {
                "dataset_id": row.dataset_id,
                "row_count": row.row_count,
                "feature_count": row.feature_count,
                "contamination_actual": row.contamination_actual,
                "contamination_expected": row.contamination_expected,
                "contamination_abs_diff": row.contamination_abs_diff,
                "row_ratio": row.row_ratio,
                "feature_ratio": row.feature_ratio,
                "ok": row.ok,
            }
            for row in rows
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
