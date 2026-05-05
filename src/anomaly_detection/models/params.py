"""Parameter models for anomaly detector wrappers."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseParams(BaseModel):
    """Shared pydantic guarantees (forbid unknown fields, immutable instances)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class OCSVMParams(BaseParams):
    """Validated parameters for One-Class SVM wrapper.

    Attributes:
        kernel: Kernel type to use in SVM.
        nu: An upper bound on the fraction of training errors and
            a lower bound of the fraction of support vectors.
        gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
        degree: Degree of the polynomial kernel function ('poly'). Ignored by other kernels.
        coef0: Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
    """

    kernel: str = "rbf"
    nu: float = Field(default=0.1, gt=0.0, le=0.5)
    gamma: str | float = "scale"
    degree: int = Field(default=3, ge=1)
    coef0: float = 0.0


class IForestParams(BaseParams):
    """Validated parameters for Isolation Forest wrapper.

    Attributes:
        n_estimators: The number of base estimators in the ensemble.
        max_samples: The number of samples to draw from the training data to train
            each base estimator. If "auto", then max_samples=min(256, n_samples).
        contamination: The amount of contamination of the data set, i.e. the proportion
            of outliers in the data set. Used when fitting to define the threshold on the decision
            function. If "auto", the threshold is determined as in the original paper.
        max_features: The number of features to draw from the training
            data to train each base estimator.
        bootstrap: Whether samples are drawn with replacement.
            If False, sampling without replacement is performed.
        random_state: Controls the randomness of the estimator.
    """

    n_estimators: int = Field(default=200, ge=1)
    max_samples: str | int = "auto"
    contamination: str | float = "auto"
    max_features: float = Field(default=1.0, gt=0.0, le=1.0)
    bootstrap: bool = False
    random_state: int = 42
    n_jobs: int = -1


class LOFParams(BaseParams):
    """Validated parameters for Local Outlier Factor wrapper.

    Attributes:
        n_neighbors: Number of neighbors to use by default for `kneighbors` queries.
        metric: The distance metric to use for the tree. The default metric is minkowski,
            and with p=2 is equivalent to the standard Euclidean metric.
        p: The power parameter for the Minkowski metric. When p = 1, this is equivalent
            to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2.
            For arbitrary p, minkowski_distance (l_p) is used.
        contamination: The amount of contamination of the data set, i.e. the proportion
            of outliers in the data set. Used when fitting to define the threshold on
            the decision function. If "auto", the threshold is determined as in the original paper.
        novelty: Whether to use LOF for novelty detection (predicting on new data) or
            for outlier detection (predicting on training data). In novelty detection,
            the LOF algorithm does not use the samples in `fit` to compute the local density,
            and therefore can be used to predict on new unseen data. In outlier detection,
            the LOF algorithm uses the samples in `fit` to compute the local density,
            and therefore can only be used to predict on
    """

    n_neighbors: int = Field(default=20, ge=2)
    metric: str = "minkowski"
    p: int = Field(default=2, ge=1)
    contamination: float = Field(default=0.1, gt=0.0, le=0.5)
    novelty: bool = False
    n_jobs: int = -1


class DBSCANParams(BaseParams):
    """Validated parameters for DBSCAN wrapper.

    Attributes:
        eps: Neighborhood radius used when eps_mode=fixed or as fallback when knee fails.
        eps_mode: When ``knee``, fit uses sorted k-distance knee per plan §2.4.
        eps_knee_multiplier: Plan §2.4 multipliers swept around eps_knee; Track A uses 1.0.
        min_samples: The number of samples (or total weight) in a neighborhood for a point
            to be considered as a core point. This includes the point itself.
        metric: The metric to use when calculating distance between instances in a feature array.
        algorithm: The algorithm to be used by the NearestNeighbors module to compute
            pointwise distances and find nearest neighbors.
    """

    eps: float = Field(default=0.5, gt=0.0)
    eps_mode: Literal["fixed", "knee"] = "knee"
    eps_knee_multiplier: float = Field(default=1.0, gt=0.0, le=10.0)
    min_samples: int = Field(default=5, ge=1)
    metric: str = "euclidean"
    algorithm: str = "auto"
    n_jobs: int = -1


class ECODParams(BaseParams):
    """Validated parameters for ECOD wrapper.

    Attributes:
        contamination: The amount of contamination of the data set, i.e. the proportion
            of outliers in the data set. Used when fitting to define the threshold on the decision
            function.
    """

    contamination: float = Field(default=0.1, gt=0.0, le=0.5)


class HBOSParams(BaseParams):
    """Validated parameters for HBOS wrapper.

    Attributes:
        n_bins: The number of bins to use for histogram estimation. If "auto", the number of bins is
            determined using the Freedman-Diaconis rule.
        alpha: Smoothing parameter added to each bin count to avoid zero probabilities.
        tol: Threshold for numerical stability when computing log probabilities.
    """

    n_bins: int | str = 10
    alpha: float = Field(default=0.1, ge=0.0, lt=1.0)
    tol: float = Field(default=0.5, ge=0.0)
    contamination: float = Field(default=0.1, gt=0.0, le=0.5)
