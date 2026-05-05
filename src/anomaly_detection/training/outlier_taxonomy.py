"""Plan §4.2 Gagolewski local/global anomaly typing via nearest cluster structure."""

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]

MIN_INLIERS_KMEANS = 12
"""Minimum inliers required to attempt KMeans clustering for local/global typing."""

MAX_K = 12
"""Maximum KMeans clusters to consider when partitioning inliers for local/global typing."""

MIN_SAMPLES_PER_CLUSTER = 5
"""Minimum inliers per cluster to consider when partitioning inliers for local/global typing."""


def classify_positives_global_vs_localclusters(
    x: FloatArray,
    y: IntArray,
    random_state: int = 42,
) -> tuple[IntArray, IntArray, FloatArray]:
    """Split positive indices into Type-G vs Type-L using nearest inlier-cluster geometry.

    Inliers (:math:`y=0`) are partitioned with ``KMeans``; :math:`k` is picked by silhouette
    on a bounded grid. Each cluster has spread :math:`\\sigma_c` (std of member distances to
    centroid). A positive sample is Type **G** (global) if its distance to the nearest cluster
    centroid exceeds :math:`2\\sigma_{c^{*}}`.

    Falls back to centroid-of-all-inliers when clustering is degenerate.

    Args:
        x: Feature matrix aligned with labels.
        y: Binary labels; ``1`` marks anomalies/noise probes.
        random_state: ``KMeans`` reproducibility seed.

    Returns:
        Triple ``type_g_positions, type_l_positions, centroid_distances Positives)``
        relative to indices in ``positives``.
    """
    positives = np.where(y == 1)[0]
    if positives.size == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.zeros(shape=(0,), dtype=np.float64)

    inliers = np.where(y == 0)[0]
    if inliers.size < MIN_INLIERS_KMEANS:
        return _fallback_global_centroid_rules(x, y, positives)

    x_in = np.asarray(x[inliers], dtype=np.float64)

    upper_k = min(MAX_K, max(2, inliers.size // MIN_SAMPLES_PER_CLUSTER))
    upper_k = min(upper_k, inliers.size - 1)
    if upper_k < 2:
        return _fallback_global_centroid_rules(x, y, positives)

    rng = random_state + inliers.size
    best_score = float("-inf")
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None

    for k in range(2, upper_k + 1):
        estimator = KMeans(
            n_clusters=k,
            random_state=rng,
            n_init="auto",
        )
        labels = estimator.fit_predict(x_in)
        if np.unique(labels).size < 2:
            continue
        try:
            sil = silhouette_score(x_in, labels)
        except ValueError:
            continue
        if sil > best_score:
            best_score = sil
            best_labels = labels
            best_centers = estimator.cluster_centers_.astype(np.float64)

    if best_centers is None or best_labels is None:
        return _fallback_global_centroid_rules(x, y, positives)

    centroid_stds = _cluster_centroid_distance_stds(x_in, best_labels, best_centers)
    x_pos = np.asarray(x[positives], dtype=np.float64)
    centroid_dists = _pairwise_centroid_distance_matrix(x_pos, best_centers)
    nearest = np.argmin(centroid_dists, axis=1)
    d_nearest = centroid_dists[np.arange(centroid_dists.shape[0]), nearest]

    sigma = centroid_stds[nearest]
    type_g_local = np.where(d_nearest > 2.0 * sigma)[0]
    type_l_local = np.where(d_nearest <= 2.0 * sigma)[0]
    type_g_positions = positives[type_g_local]
    type_l_positions = positives[type_l_local]
    return type_g_positions, type_l_positions, d_nearest


def _fallback_global_centroid_rules(
    x: FloatArray, y: IntArray, positives: np.ndarray
) -> tuple[IntArray, IntArray, FloatArray]:
    """Plan §4.2 global-centroid heuristic when clustering is unreliable."""
    if np.any(y == 0):
        centroid = np.mean(x[y == 0], axis=0)
        distances = np.linalg.norm(x - centroid, axis=1)
        within_std = float(np.std(distances[y == 0], ddof=0))
    else:
        centroid = np.mean(x, axis=0)
        distances = np.linalg.norm(x - centroid, axis=0)
        within_std = float(np.std(distances, ddof=0))
    within_std = max(within_std, 1e-9)
    threshold = 2.0 * within_std
    type_g_positions = positives[distances[positives] > threshold]
    type_l_positions = positives[distances[positives] <= threshold]
    member_dists = np.asarray(distances[positives], dtype=np.float64)
    return type_g_positions, type_l_positions, member_dists


def _cluster_centroid_distance_stds(
    x_in: FloatArray, labels: np.ndarray, centers: FloatArray
) -> FloatArray:
    """Spread of Euclidean distances inside each fitted cluster."""
    k = centers.shape[0]
    spreads = np.ones(shape=(k,), dtype=np.float64)
    for c in range(k):
        memb = labels == c
        if np.sum(memb) < 2:
            spreads[c] = 1e-9
            continue
        dists = np.linalg.norm(x_in[memb] - centers[c], axis=1)
        spreads[c] = max(float(np.std(dists, ddof=0)), 1e-9)
    return spreads


def _pairwise_centroid_distance_matrix(points: FloatArray, centers: FloatArray) -> FloatArray:
    """Return shape ``(n_points, k)`` pairwise Euclidean distances."""
    diff = points[:, None, :] - centers[None, :, :]
    return np.linalg.norm(diff, axis=2)
