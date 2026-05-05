"""Concrete anomaly detectors with unified protocol interface."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

from .params import (
    DBSCANParams,
    ECODParams,
    HBOSParams,
    IForestParams,
    LOFParams,
    OCSVMParams,
)
from .protocol import ModelProtocol

type FloatArray = npt.NDArray[np.float64]


def _to_float_array(x: npt.ArrayLike) -> FloatArray:
    """Cast array-like input to contiguous float64 matrix."""
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


@dataclass(slots=True)
class OCSVMModel:
    """One-Class SVM wrapper with unified detector interface."""

    PARAM_GRID: ClassVar[dict[str, list[Any]]] = {
        "kernel": ["rbf", "poly", "sigmoid", "linear"],
        "nu": [0.01, 0.05, 0.10, 0.20, 0.35, 0.50],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0, 10.0],
        "degree": [2, 3, 4],
        "coef0": [0.0, 1.0],
    }
    params: OCSVMParams = field(default_factory=OCSVMParams)
    _estimator: OneClassSVM = field(init=False)
    protocol: ModelProtocol = field(init=False)

    def __post_init__(self) -> None:
        self._estimator = OneClassSVM(**self.params.model_dump())
        self.protocol = ModelProtocol("OCSVM", self._fit_impl, self._score_impl)

    def _fit_impl(self, x: FloatArray) -> None:
        self._estimator.fit(x)

    def _score_impl(self, x: FloatArray) -> FloatArray:
        return -self._estimator.decision_function(x)

    def fit(self, x: npt.ArrayLike) -> None:
        """Fit model on feature matrix."""
        self.protocol.fit(_to_float_array(x))

    def score_samples(self, x: npt.ArrayLike) -> FloatArray:
        """Return anomaly scores (higher means more anomalous)."""
        return self.protocol.score_samples(_to_float_array(x))

    def predict(self, x: npt.ArrayLike, contamination: float = 0.1) -> npt.NDArray[np.int64]:
        """Return binary labels by contamination quantile."""
        return self.protocol.predict(_to_float_array(x), contamination=contamination)


@dataclass(slots=True)
class IForestModel:
    """Isolation Forest wrapper with unified detector interface."""

    PARAM_GRID: ClassVar[dict[str, list[Any]]] = {
        "n_estimators": [50, 100, 200, 500],
        "max_samples": ["auto", 32, 64, 128, 256, 512],
        "contamination": [0.01, 0.05, 0.10, 0.20, 0.35, 0.50],
        "max_features": [0.5, 0.75, 1.0],
        "bootstrap": [True, False],
        "random_state": [42],
    }
    params: IForestParams = field(default_factory=IForestParams)
    _estimator: IsolationForest = field(init=False)
    protocol: ModelProtocol = field(init=False)

    def __post_init__(self) -> None:
        self._estimator = IsolationForest(**self.params.model_dump())
        self.protocol = ModelProtocol("IForest", self._fit_impl, self._score_impl)

    def _fit_impl(self, x: FloatArray) -> None:
        self._estimator.fit(x)

    def _score_impl(self, x: FloatArray) -> FloatArray:
        return -self._estimator.score_samples(x)

    def fit(self, x: npt.ArrayLike) -> None:
        """Fit model on feature matrix."""
        self.protocol.fit(_to_float_array(x))

    def score_samples(self, x: npt.ArrayLike) -> FloatArray:
        """Return anomaly scores (higher means more anomalous)."""
        return self.protocol.score_samples(_to_float_array(x))

    def predict(self, x: npt.ArrayLike, contamination: float = 0.1) -> npt.NDArray[np.int64]:
        """Return binary labels by contamination quantile."""
        return self.protocol.predict(_to_float_array(x), contamination=contamination)


@dataclass(slots=True)
class LOFModel:
    """Local Outlier Factor wrapper with unified detector interface."""

    PARAM_GRID: ClassVar[dict[str, list[Any]]] = {
        "n_neighbors": [5, 10, 15, 20, 30, 50, 75, 100],
        "metric": ["euclidean", "manhattan", "minkowski", "cosine"],
        "p": [1, 2, 3],
        "contamination": [0.01, 0.05, 0.10, 0.20, 0.35],
        "novelty": [False],
    }
    params: LOFParams = field(default_factory=LOFParams)
    _estimator: LocalOutlierFactor = field(init=False)
    _train_x: FloatArray | None = field(init=False, default=None)
    _train_scores: FloatArray | None = field(init=False, default=None)
    protocol: ModelProtocol = field(init=False)

    def __post_init__(self) -> None:
        self._estimator = LocalOutlierFactor(**self.params.model_dump())
        self.protocol = ModelProtocol("LOF", self._fit_impl, self._score_impl)

    def _fit_impl(self, x: FloatArray) -> None:
        self._estimator.fit(x)
        self._train_x = x.copy()
        self._train_scores = -self._estimator.negative_outlier_factor_.astype(np.float64)

    def _score_impl(self, x: FloatArray) -> FloatArray:
        if self.params.novelty:
            return -self._estimator.score_samples(x)
        if self._train_x is None or self._train_scores is None:
            msg = "LOF backend is not fitted."
            raise RuntimeError(msg)
        if x.shape != self._train_x.shape or not np.array_equal(x, self._train_x):
            msg = "LOF with novelty=False supports score_samples only on fit data."
            raise ValueError(msg)
        return self._train_scores

    def fit(self, x: npt.ArrayLike) -> None:
        """Fit model on feature matrix."""
        self.protocol.fit(_to_float_array(x))

    def score_samples(self, x: npt.ArrayLike) -> FloatArray:
        """Return anomaly scores (higher means more anomalous)."""
        return self.protocol.score_samples(_to_float_array(x))

    def predict(self, x: npt.ArrayLike, contamination: float = 0.1) -> npt.NDArray[np.int64]:
        """Return binary labels by contamination quantile."""
        return self.protocol.predict(_to_float_array(x), contamination=contamination)


@dataclass(slots=True)
class DBSCANModel:
    """DBSCAN wrapper with continuous score conversion."""

    DBSCAN_EPS_MULTIPLIERS: ClassVar[list[float]] = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    PARAM_GRID: ClassVar[dict[str, list[Any]]] = {
        "eps": ["data-driven: eps_knee * multiplier"],
        "min_samples": [3, 5, 10, 15, 20, 30],
        "metric": ["euclidean", "manhattan"],
        "algorithm": ["auto"],
    }
    params: DBSCANParams = field(default_factory=DBSCANParams)
    _estimator: DBSCAN = field(init=False)
    _core_neighbors: NearestNeighbors | None = field(init=False, default=None)
    _train_x: FloatArray | None = field(init=False, default=None)
    _train_scores: FloatArray | None = field(init=False, default=None)
    protocol: ModelProtocol = field(init=False)

    def __post_init__(self) -> None:
        self._estimator = DBSCAN(**self.params.model_dump())
        self.protocol = ModelProtocol("DBSCAN", self._fit_impl, self._score_impl)

    def _fit_impl(self, x: FloatArray) -> None:
        self._estimator.fit(x)
        self._train_x = x.copy()
        if self._estimator.core_sample_indices_.size == 0:
            self._core_neighbors = None
        else:
            core_samples = x[self._estimator.core_sample_indices_]
            self._core_neighbors = NearestNeighbors(n_neighbors=1, metric=self.params.metric)
            self._core_neighbors.fit(core_samples)
        labels = self._estimator.labels_
        distances = self._distance_to_core(x)
        base = distances / self.params.eps
        scores = base.copy()
        scores[labels == -1] = 1.0 + base[labels == -1]
        scores[labels != -1] = np.minimum(scores[labels != -1], 1.0)
        self._train_scores = scores

    def _distance_to_core(self, x: FloatArray) -> FloatArray:
        if self._core_neighbors is None:
            return np.full(shape=(x.shape[0],), fill_value=self.params.eps * 2.0)
        distances, _ = self._core_neighbors.kneighbors(x, return_distance=True)
        return distances[:, 0]

    def _score_impl(self, x: FloatArray) -> FloatArray:
        if self._train_x is None or self._train_scores is None:
            msg = "DBSCAN backend is not fitted."
            raise RuntimeError(msg)
        if x.shape != self._train_x.shape or not np.array_equal(x, self._train_x):
            msg = "DBSCAN score_samples supports fit data only (transductive mode)."
            raise ValueError(msg)
        return self._train_scores

    def fit(self, x: npt.ArrayLike) -> None:
        """Fit model on feature matrix."""
        self.protocol.fit(_to_float_array(x))

    def score_samples(self, x: npt.ArrayLike) -> FloatArray:
        """Return anomaly scores (higher means more anomalous)."""
        return self.protocol.score_samples(_to_float_array(x))

    def predict(self, x: npt.ArrayLike, contamination: float = 0.1) -> npt.NDArray[np.int64]:
        """Return binary labels by contamination quantile."""
        return self.protocol.predict(_to_float_array(x), contamination=contamination)


@dataclass(slots=True)
class ECODModel:
    """ECOD wrapper backed by PyOD with unified detector interface."""

    PARAM_GRID: ClassVar[dict[str, list[Any]]] = {
        "contamination": [0.01, 0.05, 0.10, 0.20, 0.35, 0.50]
    }
    params: ECODParams = field(default_factory=ECODParams)
    _estimator: ECOD = field(init=False)
    protocol: ModelProtocol = field(init=False)

    def __post_init__(self) -> None:
        self._estimator = ECOD(**self.params.model_dump())
        self.protocol = ModelProtocol("ECOD", self._fit_impl, self._score_impl)

    def _fit_impl(self, x: FloatArray) -> None:
        self._estimator.fit(x)

    def _score_impl(self, x: FloatArray) -> FloatArray:
        return np.asarray(self._estimator.decision_function(x), dtype=np.float64)

    def fit(self, x: npt.ArrayLike) -> None:
        """Fit model on feature matrix."""
        self.protocol.fit(_to_float_array(x))

    def score_samples(self, x: npt.ArrayLike) -> FloatArray:
        """Return anomaly scores (higher means more anomalous)."""
        return self.protocol.score_samples(_to_float_array(x))

    def predict(self, x: npt.ArrayLike, contamination: float = 0.1) -> npt.NDArray[np.int64]:
        """Return binary labels by contamination quantile."""
        return self.protocol.predict(_to_float_array(x), contamination=contamination)


@dataclass(slots=True)
class HBOSModel:
    """HBOS wrapper backed by PyOD with unified detector interface."""

    PARAM_GRID: ClassVar[dict[str, list[Any]]] = {
        "n_bins": [5, 10, 20, 30, 50, "auto"],
        "alpha": [0.0, 0.1, 0.2, 0.5],
        "tol": [0.1, 0.5],
        "contamination": [0.01, 0.05, 0.10, 0.20, 0.35],
    }
    params: HBOSParams = field(default_factory=HBOSParams)
    _estimator: HBOS = field(init=False)
    protocol: ModelProtocol = field(init=False)

    def __post_init__(self) -> None:
        self.protocol = ModelProtocol("HBOS", self._fit_impl, self._score_impl)

    def _resolve_bins(self, n_samples: int) -> int:
        if self.params.n_bins == "auto":
            return max(2, int(np.sqrt(n_samples)))
        return int(self.params.n_bins)

    def _fit_impl(self, x: FloatArray) -> None:
        if self.params.n_bins == "auto":
            n_bins = self._resolve_bins(x.shape[0])
        else:
            n_bins = self.params.n_bins
        self._estimator = HBOS(
            n_bins=n_bins,
            alpha=self.params.alpha,
            tol=self.params.tol,
            contamination=self.params.contamination,
        )
        self._estimator.fit(x)

    def _score_impl(self, x: FloatArray) -> FloatArray:
        return np.asarray(self._estimator.decision_function(x), dtype=np.float64)

    def fit(self, x: npt.ArrayLike) -> None:
        """Fit model on feature matrix."""
        self.protocol.fit(_to_float_array(x))

    def score_samples(self, x: npt.ArrayLike) -> FloatArray:
        """Return anomaly scores (higher means more anomalous)."""
        return self.protocol.score_samples(_to_float_array(x))

    def predict(self, x: npt.ArrayLike, contamination: float = 0.1) -> npt.NDArray[np.int64]:
        """Return binary labels by contamination quantile."""
        return self.protocol.predict(_to_float_array(x), contamination=contamination)
