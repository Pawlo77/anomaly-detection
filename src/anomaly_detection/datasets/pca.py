"""PCA transformation utilities."""

from dataclasses import dataclass

import pandas as pd
from sklearn.decomposition import PCA

PCA_COMPONENT_PREFIX = "pc_"
"""Prefix for generated PCA component names."""


@dataclass(frozen=True, slots=True)
class PcaResult:
    """PCA result payload with metadata.

    Attributes:
        transformed: PCA-transformed data frame.
        n_components: Number of PCA components retained.
        explained_variance_ratio_sum: Total variance ratio explained by retained components.
    """

    transformed: pd.DataFrame
    n_components: int
    explained_variance_ratio_sum: float


def pca_by_variance(frame: pd.DataFrame, variance_ratio: float, seed: int) -> pd.DataFrame:
    """Project features with variance-retaining PCA.

    Args:
        frame: Input feature matrix.
        variance_ratio: Fraction of variance to retain in (0, 1].
        seed: Random seed used by solver.

    Returns:
        PCA-transformed data frame with generated component names.
    """
    return pca_by_variance_with_metadata(
        frame=frame, variance_ratio=variance_ratio, seed=seed
    ).transformed


def pca_by_variance_with_metadata(
    frame: pd.DataFrame, variance_ratio: float, seed: int
) -> PcaResult:
    """Project features with PCA while surfacing explanatory variance statistics.

    Args:
        frame: Numeric feature-only ``DataFrame``.
        variance_ratio: Target cumulative variance in ``(0, 1]`` passed to sklearn.
        seed: Random seed delegated to deterministic SVD backends.

    Returns:
        Structured ``PcaResult`` exposing transformed coordinates plus metadata counts.
    """
    pca = PCA(n_components=variance_ratio, svd_solver="full", random_state=seed)
    transformed = pca.fit_transform(frame)
    columns = [f"{PCA_COMPONENT_PREFIX}{idx}" for idx in range(transformed.shape[1])]
    table = pd.DataFrame(transformed, columns=columns, index=frame.index)
    return PcaResult(
        transformed=table,
        n_components=table.shape[1],
        explained_variance_ratio_sum=float(pca.explained_variance_ratio_.sum()),
    )
