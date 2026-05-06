"""Phase-6 artifact generation for score-redundancy and elbow validation analyses."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.loader import DatasetLoader
from ..models import DBSCANModel, ECODModel, HBOSModel, IForestModel, LOFModel, OCSVMModel
from ..models.params import (
    DBSCANParams,
    ECODParams,
    HBOSParams,
    IForestParams,
    LOFParams,
    OCSVMParams,
)
from .ensemble import normalize_to_rank, weighted_borda
from .phase5 import elbow_contamination_from_scores

SCORES_FILENAME = "outputs/phase6_scores.csv.gz"
"""Filename for the exported score vectors, compressed as CSV for efficient storage and loading."""
ELBOW_VALIDATION_FILENAME = "outputs/phase6_elbow_validation.csv"
"""Filename for the exported elbow-contamination parity table,
sorted by absolute error for easy analysis."""
HEAVY_ALGORITHMS = frozenset({"OCSVM", "LOF", "DBSCAN"})
"""Set of algorithms with heavier computational profiles,
used for logging and potential future filtering."""


@dataclass(frozen=True, slots=True)
class Phase6ArtifactsReport:
    """Materialized artifact paths with row-level export counts."""

    scores_path: Path
    elbow_validation_path: Path
    exported_score_rows: int
    datasets_processed: int


def _build_model(algorithm: str) -> object:
    """Factory function to instantiate models with consistent parameters for phase-6 scoring."""
    if algorithm == "OCSVM":
        return OCSVMModel(params=OCSVMParams())
    if algorithm == "IForest":
        return IForestModel(params=IForestParams(n_jobs=-1))
    if algorithm == "LOF":
        return LOFModel(params=LOFParams(n_jobs=-1))
    if algorithm == "DBSCAN":
        return DBSCANModel(params=DBSCANParams(n_jobs=-1))
    if algorithm == "ECOD":
        return ECODModel(params=ECODParams())
    if algorithm == "HBOS":
        return HBOSModel(params=HBOSParams())
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _phase6_datasets(loader: DatasetLoader) -> list[str]:
    """Identify datasets with positive contamination for phase-6 analysis."""
    dataset_ids: list[str] = []
    for dataset_id in loader.catalog.ids():
        spec = loader.catalog.get(dataset_id)
        if spec.contamination <= 0.0:
            continue
        dataset_ids.append(dataset_id)
    return dataset_ids


def _load_aligned_matrix(loader: DatasetLoader, dataset_id: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return matrix/labels aligned across all algorithms for Kendall comparisons."""
    spec = loader.catalog.get(dataset_id)
    if spec.stratified_subsample_for_n2:
        # Use one shared stratified draw for all methods to keep row-wise comparability.
        bundle = loader.load_subsample(
            dataset_id=dataset_id,
            algorithm="OCSVM",
            seed=42,
            view="preprocessed",
        )
    else:
        bundle = loader.load(dataset_id=dataset_id, view="preprocessed")
    x = bundle.X.reset_index(drop=True)
    y = bundle.y.reset_index(drop=True)
    return x, y


def export_phase6_artifacts(
    scores_path: Path | None = None,
    elbow_validation_path: Path | None = None,
) -> Phase6ArtifactsReport:
    """Export score vectors and elbow-contamination parity table for phase-6 analysis."""
    loader = DatasetLoader()
    algorithms = ("OCSVM", "IForest", "LOF", "DBSCAN", "ECOD", "HBOS")
    target_scores = scores_path or Path(SCORES_FILENAME)
    target_elbow = elbow_validation_path or Path(ELBOW_VALIDATION_FILENAME)
    target_scores.parent.mkdir(parents=True, exist_ok=True)
    target_elbow.parent.mkdir(parents=True, exist_ok=True)

    score_rows: list[dict[str, float | int | str]] = []
    elbow_rows: list[dict[str, float | str | int]] = []

    for dataset_id in _phase6_datasets(loader):
        x_frame, y_series = _load_aligned_matrix(loader, dataset_id)
        x = np.asarray(x_frame.to_numpy(), dtype=np.float64)
        y = np.asarray(y_series.to_numpy(), dtype=np.int64)
        ranked_scores: dict[str, np.ndarray] = {}
        raw_scores: dict[str, np.ndarray] = {}

        for algorithm in algorithms:
            model = _build_model(algorithm)
            model.fit(x)
            raw = np.asarray(model.score_samples(x), dtype=np.float64)
            ranked = normalize_to_rank(raw)
            raw_scores[algorithm] = raw
            ranked_scores[algorithm] = ranked
            for row_index, (raw_value, rank_value, label) in enumerate(
                zip(raw, ranked, y, strict=True)
            ):
                score_rows.append(
                    {
                        "dataset": dataset_id,
                        "algorithm": algorithm,
                        "row_index": row_index,
                        "score_raw": float(raw_value),
                        "score_rank": float(rank_value),
                        "label": int(label),
                    }
                )

        ensemble = weighted_borda(ranked_scores)
        predicted = elbow_contamination_from_scores(ensemble)
        actual = float(y.mean())
        elbow_rows.append(
            {
                "dataset": dataset_id,
                "rows_used": int(x.shape[0]),
                "features_used": int(x.shape[1]),
                "contamination_true": actual,
                "contamination_elbow": float(predicted),
                "abs_error": float(abs(actual - predicted)),
            }
        )

    pd.DataFrame(score_rows).to_csv(target_scores, index=False)
    pd.DataFrame(elbow_rows).sort_values("abs_error").to_csv(target_elbow, index=False)
    return Phase6ArtifactsReport(
        scores_path=target_scores,
        elbow_validation_path=target_elbow,
        exported_score_rows=len(score_rows),
        datasets_processed=len(elbow_rows),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build a command-line argument parser for phase-6 artifact generation."""
    parser = argparse.ArgumentParser(description="Export phase-6 analysis artifacts.")
    parser.add_argument("--scores-out", type=Path, default=Path(SCORES_FILENAME))
    parser.add_argument("--elbow-out", type=Path, default=Path(ELBOW_VALIDATION_FILENAME))
    return parser


def main() -> None:
    """Main entry point for phase-6 artifact generation."""
    args = build_parser().parse_args()
    report = export_phase6_artifacts(
        scores_path=args.scores_out,
        elbow_validation_path=args.elbow_out,
    )
    print(
        "phase6 artifacts exported:",
        f"scores={report.scores_path}",
        f"elbow={report.elbow_validation_path}",
        f"score_rows={report.exported_score_rows}",
        f"datasets={report.datasets_processed}",
    )


if __name__ == "__main__":
    main()
