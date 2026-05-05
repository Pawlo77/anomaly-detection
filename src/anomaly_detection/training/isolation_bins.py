"""Isolation-by-distance deciles for ODDS benchmarking proxies (§4.2)."""

import numpy as np
import numpy.typing as npt
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from ..metrics import MetricThresholdConfig, labels_from_scores

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]


def odds_isolation_decile_metrics(
    x: FloatArray,
    y: IntArray,
    scores: FloatArray,
    contamination: float,
    max_geom_samples: int = 512,
) -> dict[str, float]:
    """Bin true outliers by nearest-inlier normalized distance deciles.

    Args:
        x: Feature matrix defining geometry for neighbor queries.
        y: Binary labels with ``1`` marking ground-truth anomalies.
        scores: Model anomaly scores paired row-wise with ``x``.
        contamination: Budget forwarded to deterministic thresholding.
        max_geom_samples: Row cap controlling pairwise distance approximation cost.

    Returns:
        Metrics keyed ``idb_*`` mixing counts/recalls across ten buckets. Empty
        dict when either class lacks support for stratification.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64)
    outliers = np.where(y_arr == 1)[0]
    inliers = np.where(y_arr == 0)[0]
    if outliers.size == 0 or inliers.size == 0:
        return {}

    neighbor = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
    neighbor.fit(x_arr[inliers])
    nearest_dist, _ = neighbor.kneighbors(x_arr[outliers])
    nearest_dist = nearest_dist[:, 0].astype(np.float64)

    sample_n = min(max_geom_samples, x_arr.shape[0])
    geom = x_arr[:sample_n]
    matrix = pairwise_distances(geom, metric="euclidean")
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    diameter = float(np.max(upper)) if upper.size else float(np.max(nearest_dist))
    diameter = max(diameter, 1e-12)
    norm = nearest_dist / diameter

    y_pred = labels_from_scores(scores, MetricThresholdConfig(contamination=contamination))
    edges = np.percentile(norm, np.linspace(0.0, 100.0, 11))

    metrics: dict[str, float] = {}
    for bucket in range(10):
        low = float(edges[bucket])
        high = float(edges[bucket + 1])
        mask = (norm >= low) & (norm < high) if bucket < 9 else (norm >= low) & (norm <= high)
        count = int(np.sum(mask))
        metrics[f"idb_count_d{bucket}"] = float(count)
        if count == 0:
            metrics[f"idb_recall_d{bucket}"] = 0.0
            continue
        idx = outliers[mask]
        metrics[f"idb_recall_d{bucket}"] = float(np.mean(y_pred[idx] == 1))
    return metrics
