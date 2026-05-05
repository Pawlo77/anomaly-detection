"""DatasetLoader facade materializing canonical CSV views and subsamples."""

from pathlib import Path

import pandas as pd

from ..config import DatasetSettings
from .catalog import DatasetCatalog, build_default_catalog
from .pca import pca_by_variance
from .preprocess import preprocess_arrhythmia, robust_scale_features
from .sampling import stratified_subsample
from .stats import DatasetStats, compute_descriptive_stats
from .types import DatasetBundle

LABEL_COLUMN_NAME = "label"
"""Canonical binary label column name."""


class DatasetLoader:
    """Hydrate dataframe bundles backed by persisted canonical CSV artifacts.

    Attributes:
        settings: Resolved path roots and stochastic knobs influencing PCA/sampling.
        catalog: Specification registry guarding dataset identifiers.
    """

    def __init__(
        self, settings: DatasetSettings | None = None, catalog: DatasetCatalog | None = None
    ):
        """Attach optional settings/catalog overrides commonly used by tests."""
        self.settings = settings or DatasetSettings()
        self.catalog = catalog or build_default_catalog()

    def list_datasets(self) -> list[str]:
        """Return identifiers known to the bound catalog ordering."""
        return self.catalog.ids()

    def load(
        self,
        dataset_id: str,
        view: str = "raw",
        pca_random_state: int | None = None,
    ) -> DatasetBundle:
        """Load dataset view from canonical artifact store.

        Args:
            dataset_id: Dataset identifier.
            view: View key in ``raw``, ``preprocessed``, or ``pca95``.
            pca_random_state: Random seed for PCA when ``view=pca95`` (defaults to 42).

        Returns:
            Dataset bundle with features and labels.
        """
        path = self._artifact_path(dataset_id)
        table = pd.read_csv(path)
        y = table[LABEL_COLUMN_NAME].astype(int)
        x_values = table.drop(columns=[LABEL_COLUMN_NAME])
        if view == "preprocessed":
            x_values = (
                preprocess_arrhythmia(x_values)
                if dataset_id == "arrhythmia"
                else robust_scale_features(x_values)
            )
        elif view == "pca95":
            base = (
                preprocess_arrhythmia(x_values)
                if dataset_id == "arrhythmia"
                else robust_scale_features(x_values)
            )
            rng = int(pca_random_state) if pca_random_state is not None else 42
            x_values = pca_by_variance(base, self.settings.arrhythmia_pca_variance, seed=rng)
        return DatasetBundle(dataset_id=dataset_id, view=view, X=x_values, y=y, source_path=path)

    def load_subsample(
        self,
        dataset_id: str,
        algorithm: str,
        seed: int,
        view: str = "preprocessed",
        pca_random_state: int | None = None,
    ) -> DatasetBundle:
        """Load stratified subsample for expensive algorithm.

        Args:
            dataset_id: Dataset identifier.
            algorithm: Algorithm name controlling cap.
            seed: Sampling seed.

        Returns:
            Subsampled dataset bundle.
        """
        bundle = self.load(dataset_id, view=view, pca_random_state=pca_random_state)
        cap = self.settings.n2_algorithm_caps.get(algorithm)
        if cap is None:
            return bundle
        merged = bundle.X.copy()
        merged[LABEL_COLUMN_NAME] = bundle.y
        sampled = stratified_subsample(
            merged, label_column=LABEL_COLUMN_NAME, n_samples=cap, seed=seed
        )
        return DatasetBundle(
            dataset_id=dataset_id,
            view=f"subsample_{algorithm}",
            X=sampled.drop(columns=[LABEL_COLUMN_NAME]),
            y=sampled[LABEL_COLUMN_NAME].astype(int),
            source_path=bundle.source_path,
        )

    def stats(self, dataset_id: str) -> DatasetStats:
        """Compute descriptive statistics for the raw canonical slice.

        Args:
            dataset_id: Catalog-backed identifier.

        Returns:
            Structured ``DatasetStats`` over merged feature/label frames.
        """
        bundle = self.load(dataset_id, view="raw")
        merged = bundle.X.copy()
        merged[LABEL_COLUMN_NAME] = bundle.y
        return compute_descriptive_stats(merged, label_column=LABEL_COLUMN_NAME)

    def _artifact_path(self, dataset_id: str) -> Path:
        """Resolve filesystem location for flattened canonical CSV shards.

        Args:
            dataset_id: Logical dataset key possibly containing slashes.

        Returns:
            Absolute path constrained by ``DatasetSettings``.
        """
        return self.settings.resolve(
            self.settings.canonical_dir / f"{dataset_id.replace('/', '__')}.csv"
        )
