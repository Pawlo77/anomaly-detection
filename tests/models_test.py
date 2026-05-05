"""Tests for anomaly model wrappers and protocol behavior."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_detection.models import (
    DBSCANModel,
    ECODModel,
    HBOSModel,
    IForestModel,
    LOFModel,
    OCSVMModel,
)
from anomaly_detection.models.protocol import ModelProtocol


def _toy_data() -> np.ndarray:
    """Create deterministic toy matrix for model smoke tests."""
    rng = np.random.default_rng(42)
    normal = rng.normal(loc=0.0, scale=1.0, size=(64, 4))
    outliers = rng.normal(loc=5.0, scale=0.5, size=(8, 4))
    return np.vstack([normal, outliers])


@pytest.mark.parametrize(
    ("model_cls", "required_keys"),
    [
        (OCSVMModel, {"kernel", "nu", "gamma"}),
        (IForestModel, {"n_estimators", "max_samples", "contamination"}),
        (LOFModel, {"n_neighbors", "metric", "contamination"}),
        (DBSCANModel, {"eps", "min_samples", "metric"}),
        (ECODModel, {"contamination"}),
        (HBOSModel, {"n_bins", "alpha", "tol", "contamination"}),
    ],
)
def test_param_grid_is_defined(model_cls: type, required_keys: set[str]) -> None:
    """Each wrapper exposes class-level hyperparameter grid."""
    assert required_keys.issubset(set(model_cls.PARAM_GRID))


@pytest.mark.parametrize(
    "model",
    [
        OCSVMModel(),
        IForestModel(),
        LOFModel(),
        DBSCANModel(),
        ECODModel(),
        HBOSModel(),
    ],
)
def test_model_protocol_fit_score_predict_contract(model: object) -> None:
    """All wrappers support unified fit/score/predict interface."""
    x = _toy_data()
    model.fit(x)
    scores = model.score_samples(x)
    labels = model.predict(x, contamination=0.1)

    assert isinstance(model.protocol, ModelProtocol)
    assert scores.shape == (x.shape[0],)
    assert labels.shape == (x.shape[0],)
    assert set(np.unique(labels)).issubset({0, 1})
    assert labels.sum() >= 1


def test_protocol_raises_when_scoring_before_fit() -> None:
    """Protocol blocks score calls before fitting."""
    model = ECODModel()
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = model.score_samples(_toy_data())


def test_transductive_models_reject_unseen_data_scoring() -> None:
    """LOF/DBSCAN in transductive mode should reject unseen matrices."""
    train_x = _toy_data()
    new_x = train_x + 0.001

    lof = LOFModel()
    lof.fit(train_x)
    with pytest.raises(ValueError, match="novelty=False"):
        _ = lof.score_samples(new_x)

    dbscan = DBSCANModel()
    dbscan.fit(train_x)
    with pytest.raises(ValueError, match="fit data only"):
        _ = dbscan.score_samples(new_x)
