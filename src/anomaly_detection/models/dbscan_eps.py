"""Heuristic DBSCAN radius selection via sorted k-distance knee (§2.4)."""

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors

EPS_FLOOR = 1e-6
"""Numerical floor for DBSCAN eps to prevent underflow and degeneracy."""


def compute_dbscan_eps_knee(
    x: npt.NDArray[np.floating],
    min_samples: int,
    metric: str,
) -> float:
    """Return ``eps_knee`` from curvature of sorted k-distance graph.

    Args:
        x: Training features used to approximate local density shells.
        min_samples: ``min_samples`` hyperparameter aligning ``k`` with DBSCAN rule.
        metric: Passed through to sklearn ``NearestNeighbors``.

    Returns:
        Positive ``eps`` estimate clipped against numerical floors. Tiny or nearly
        empty inputs fall back to scale-aware medians rather than throwing.
    """
    xx = np.ascontiguousarray(x, dtype=np.float64)
    n_samples = int(xx.shape[0])
    k_target = max(1, int(min_samples) - 1)
    if n_samples <= k_target + 1:
        fallback = np.median(np.linalg.norm(xx - xx.mean(axis=0), axis=1))
        return float(np.clip(max(fallback * 0.1, EPS_FLOOR), EPS_FLOOR, 100.0))
    nn = int(min(k_target, n_samples - 1))
    neighbors = NearestNeighbors(n_neighbors=nn, metric=metric, algorithm="auto", n_jobs=-1)
    neighbors.fit(xx)
    dist, _ = neighbors.kneighbors(xx)
    spread = np.sort(dist[:, -1])
    if spread.size <= 5:
        return float(np.clip(max(np.median(spread), EPS_FLOOR), EPS_FLOOR, 100.0))
    y = spread / float(np.maximum(np.percentile(spread, 95), EPS_FLOOR))
    curvature = np.abs(np.gradient(np.gradient(y)))
    margin = max(5, spread.size // 40)
    band = curvature[margin : spread.size - margin]
    knee_local = margin + int(np.argmax(band)) if band.size else spread.size // 2
    return float(np.clip(max(spread[knee_local], EPS_FLOOR), EPS_FLOOR, 100.0))
