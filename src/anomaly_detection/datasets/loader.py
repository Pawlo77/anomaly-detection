"""High-level dataset loading API."""

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
    """Entry point for retrieving prepared dataset views."""

    def __init__(
        self, settings: DatasetSettings | None = None, catalog: DatasetCatalog | None = None
    ):
        """Initialize loader with optional settings overrides."""
        self.settings = settings or DatasetSettings()
        self.catalog = catalog or build_default_catalog()

    def list_datasets(self) -> list[str]:
        """Return all registered dataset IDs."""
        return self.catalog.ids()

    def load(self, dataset_id: str, view: str = "raw") -> DatasetBundle:
        """Load dataset view from canonical artifact store.

        Args:
            dataset_id: Dataset identifier.
            view: View key in `raw|preprocessed|pca95`.

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
            x_values = pca_by_variance(base, self.settings.arrhythmia_pca_variance, seed=42)
        return DatasetBundle(dataset_id=dataset_id, view=view, X=x_values, y=y, source_path=path)

    def load_subsample(self, dataset_id: str, algorithm: str, seed: int) -> DatasetBundle:
        """Load stratified subsample for expensive algorithm.

        Args:
            dataset_id: Dataset identifier.
            algorithm: Algorithm name controlling cap.
            seed: Sampling seed.

        Returns:
            Subsampled dataset bundle.
        """
        bundle = self.load(dataset_id, view="preprocessed")
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
        """Compute descriptive stats for raw dataset."""
        bundle = self.load(dataset_id, view="raw")
        merged = bundle.X.copy()
        merged[LABEL_COLUMN_NAME] = bundle.y
        return compute_descriptive_stats(merged, label_column=LABEL_COLUMN_NAME)

    def _artifact_path(self, dataset_id: str) -> Path:
        """Build canonical artifact path."""
        return self.settings.resolve(
            self.settings.canonical_dir / f"{dataset_id.replace('/', '__')}.csv"
        )
