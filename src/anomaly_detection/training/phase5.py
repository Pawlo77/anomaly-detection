"""Phase-five blind ensemble exporter producing ``test_labels.csv`` submissions."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.catalog import FULL_DATASET_SPECS, DatasetSpec
from ..geometry import distance_concentration_ratio
from ..models import DBSCANModel, ECODModel, HBOSModel, IForestModel, LOFModel, OCSVMModel
from ..models.params import (
    DBSCANParams,
    ECODParams,
    HBOSParams,
    IForestParams,
    LOFParams,
    OCSVMParams,
)
from .blind_audit import write_blind_audit_json
from .blind_features import prepare_blind_feature_matrix
from .ensemble import consensus_labels, normalize_to_rank, weighted_borda

TEST_DATA_FILENAME = "test_data.csv"
"""Input CSV file used for blind verification."""

TEST_LABELS_FILENAME = "test_labels.csv"
"""Default output filename for phase-5 predictions."""

HIGH_DIMENSION_THRESHOLD = 50
"""Feature-count threshold for high-dimensional weighting policy."""

DISTANCE_CONCENTRATION_THRESHOLD = 0.1
"""Distance concentration ratio threshold for weighting policy."""

STRUCTURAL_DISTANCE_THRESHOLD = 2.0
"""Z-style distance threshold: when no ODDS profile is sufficiently close,
use elbow on the aggregated ensemble ranking (plan §5.3 fallback).
"""


def _nearest_odds_contamination(rows: int, features: int) -> tuple[float, float]:
    """Estimate contamination by nearest ODDS catalog neighbors in row/feature space.

    Args:
        rows: Observed row count after blind preprocessing.
        features: Observed feature dimensionality.

    Returns:
        Tuple ``(mean contamination, profile distance)`` where distance is normalized
        Euclidean distance in standardized row/feature coordinates.
    """
    odds_specs = [spec for spec in FULL_DATASET_SPECS if "/" not in spec.dataset_id]
    if not odds_specs:
        raise ValueError("No ODDS dataset specs configured.")
    ratios = np.array([float(s.expected_rows) for s in odds_specs], dtype=np.float64)
    dims = np.array([float(s.expected_dim) for s in odds_specs], dtype=np.float64)
    sigma_r = max(float(np.std(ratios)), 1.0)
    sigma_f = max(float(np.std(dims)), 1.0)

    def metric(spec: DatasetSpec) -> float:
        return float(
            np.hypot(
                (rows - float(spec.expected_rows)) / sigma_r,
                (features - float(spec.expected_dim)) / sigma_f,
            )
        )

    nearest = sorted(odds_specs, key=lambda spec: (metric(spec), spec.dataset_id))[:3]
    c_hat = float(np.mean([s.contamination for s in nearest]))
    return c_hat, float(metric(nearest[0]))


def elbow_contamination_from_scores(scores: np.ndarray) -> float:
    """Infer contamination from dominant jump in descending ensemble rankings (§5.3).

    Args:
        scores: Aggregated ensemble scores used for rank-based heuristics.

    Returns:
        Clipped contamination ratio suitable for ``consensus_labels``.
    """
    s_desc = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    n = int(s_desc.shape[0])
    if n < 12:
        return float(np.clip(5.5 / float(n), 0.015, 0.45))
    deltas = np.diff(s_desc)
    k_idx = min(n - 1, max(1, int(np.argmax(deltas)) + 1))
    return float(np.clip(float(k_idx) / float(n), 3.6 / float(n), 0.49))


@dataclass(frozen=True, slots=True)
class Phase5ExportReport:
    """Summary metadata for generated phase-5 export.

    Attributes:
        output_path: Filesystem path where submission CSV is written.
        rows: Number of exported rows.
        outliers: Count of rows labeled as outliers.
        contamination_estimate: Estimated contamination ratio used for thresholding.
        distance_concentration_ratio: Distance concentration ratio on prepared matrix.
    """

    output_path: Path
    rows: int
    outliers: int
    contamination_estimate: float
    distance_concentration_ratio: float


def export_phase5_labels(output_path: Path | None = None) -> Phase5ExportReport:
    """Generate ``test_labels.csv`` using the planned rank-aggregation protocol.

    Args:
        output_path: Optional override for submission CSV location.

    Returns:
        ``Phase5ExportReport`` summarizing rows, outliers, and geometry stats.

    Raises:
        FileNotFoundError: When ``test_data.csv`` is absent from the working tree.
    """
    source_path = Path(TEST_DATA_FILENAME)
    if not source_path.exists():
        raise FileNotFoundError(f"{TEST_DATA_FILENAME} not found.")
    output = output_path or Path(TEST_LABELS_FILENAME)

    table = pd.read_csv(source_path)
    write_blind_audit_json(table, output.with_name("test_data_profile.json"))
    x = prepare_blind_feature_matrix(table)
    n_rows, n_features = x.shape
    cr = distance_concentration_ratio(x, pool_max_rows=None)
    c_odds, profile_distance = _nearest_odds_contamination(n_rows=n_rows, n_features=n_features)

    models = {
        "OCSVM": OCSVMModel(params=OCSVMParams()),
        "IForest": IForestModel(params=IForestParams(n_jobs=-1)),
        "LOF": LOFModel(params=LOFParams(n_jobs=-1)),
        "DBSCAN": DBSCANModel(params=DBSCANParams(n_jobs=-1)),
        "ECOD": ECODModel(params=ECODParams()),
        "HBOS": HBOSModel(params=HBOSParams()),
    }
    ranked: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(x)
        ranked[name] = normalize_to_rank(model.score_samples(x))

    if n_features > HIGH_DIMENSION_THRESHOLD or cr < DISTANCE_CONCENTRATION_THRESHOLD:
        weights = {
            "OCSVM": 0.5,
            "LOF": 0.5,
            "DBSCAN": 0.5,
            "IForest": 1.0,
            "ECOD": 1.5,
            "HBOS": 1.5,
        }
    else:
        weights = dict.fromkeys(models, 1.0)

    ensemble_score = weighted_borda(ranked, weights=weights)
    if profile_distance > STRUCTURAL_DISTANCE_THRESHOLD:
        c_hat = elbow_contamination_from_scores(ensemble_score)
    else:
        c_hat = c_odds
    c_eff = float(np.clip(c_hat, max(15.0 / float(max(n_rows, 1)), 1e-6), 0.489))
    labels = consensus_labels(
        ranked_scores=ranked, ensemble_score=ensemble_score, contamination_estimate=c_eff
    )

    submission = pd.DataFrame({"class": labels.astype(int)})
    assert submission.columns.tolist() == ["class"]
    assert set(submission["class"].unique()).issubset({0, 1})
    assert len(submission) == len(table)
    submission.to_csv(output, index=False)
    return Phase5ExportReport(
        output_path=output,
        rows=len(submission),
        outliers=int(submission["class"].sum()),
        contamination_estimate=float(c_eff),
        distance_concentration_ratio=float(cr),
    )
