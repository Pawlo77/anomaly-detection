"""Composable protocol for anomaly detector wrappers."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
FitCallable = Callable[[FloatArray], None]
ScoreCallable = Callable[[FloatArray], FloatArray]


class ThresholdConfig(BaseModel):
    """Thresholding config for binary labels from continuous scores.

    Attributes:
        contamination: Fraction of samples to mark as outliers.
    """

    contamination: float = Field(default=0.1, gt=0.0, le=1.0)


@dataclass(slots=True)
class ModelProtocol:
    """Model protocol implemented through composition.

    This class centralizes the shared interface used in experiments:
    `fit`, `score_samples`, and `predict`.

    Attributes:
        name: Human-readable model name.
        fit_fn: Function used to fit wrapped backend.
        score_fn: Function that returns anomaly scores (higher = more anomalous).
    """

    name: str
    fit_fn: FitCallable
    score_fn: ScoreCallable
    is_fitted: bool = False

    def fit(self, x: FloatArray) -> None:
        """Fit backend model.

        Args:
            x: Feature matrix of shape `(n_samples, n_features)`.
        """
        self.fit_fn(x)
        self.is_fitted = True

    def score_samples(self, x: FloatArray) -> FloatArray:
        """Return continuous anomaly scores.

        Args:
            x: Feature matrix of shape `(n_samples, n_features)`.

        Returns:
            Scores where larger means more anomalous.

        Raises:
            RuntimeError: If called before fitting.
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted. Call fit() first.")
        return self.score_fn(x)

    def predict(self, x: FloatArray, contamination: float = 0.1) -> IntArray:
        """Return binary anomaly labels using quantile thresholding.

        Args:
            x: Feature matrix of shape `(n_samples, n_features)`.
            contamination: Fraction of highest-scoring samples labeled outliers.

        Returns:
            Array of labels, where `1` means outlier and `0` means inlier.

        Raises:
            RuntimeError: If ``fit`` was never called successfully.
        """
        cfg = ThresholdConfig(contamination=contamination)
        scores = self.score_samples(x)
        threshold = float(np.quantile(scores, 1.0 - cfg.contamination))
        return (scores >= threshold).astype(np.int64)
