"""Distance-based geometric summaries shared by training and dataset reporting.

Distance concentration CR(d) follows plan §4.1: Euclidean distances on uniformly
random row pairs independent of pairwise matrix materialization costs.
"""

import numpy as np
import numpy.typing as npt

DEFAULT_PAIR_COUNT = 500
"""Number of sampled pairwise distances in CR(d) estimates (plan §4.1)."""


def distance_concentration_ratio(
    x: npt.ArrayLike,
    pair_count: int = DEFAULT_PAIR_COUNT,
    pool_max_rows: int | None = None,
    random_state: int = 42,
) -> float:
    """Compute (d_max - d_min) / d_mean from Euclidean distances on random row pairs.

    Draws indices ``i,j`` uniformly with replacement from ``{0...m-1}`` excluding ``i=j``,
    where ``m = min(pool_max_rows, n)`` when ``pool_max_rows`` is set or ``n`` otherwise.

    Args:
        x: Feature matrix of shape ``(n_samples, n_features)``.
        pair_count: Cardinality of unordered distance multiset (distinct index pairs).
        pool_max_rows: Optional cap aligning CR with chronological prefix subsets.
        random_state: NumPy RNG seed for reproducibility.

    Returns:
        Concentration statistic, or ``0.0`` when degenerate data yield no spread.
    """
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 2 or xx.shape[0] < 2 or pair_count < 1:
        return 0.0

    pool_n_full = int(xx.shape[0])
    upper = pool_n_full if pool_max_rows is None else min(int(pool_max_rows), pool_n_full)
    if upper < 2:
        return 0.0

    pool = xx[:upper]

    rng = np.random.default_rng(seed=random_state)
    i_idx = rng.integers(0, upper, size=pair_count)
    offsets = rng.integers(1, upper, size=pair_count)
    j_idx = (i_idx + offsets) % upper

    diffs = pool[i_idx] - pool[j_idx]
    distances = np.linalg.norm(diffs, axis=1)
    d_mean = float(np.mean(distances))
    if d_mean <= 0.0:
        return 0.0
    return float((float(np.max(distances)) - float(np.min(distances))) / d_mean)
