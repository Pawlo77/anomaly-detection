"""Tests for blind test-data feature preparation."""

import numpy as np
import pandas as pd
import pytest

from anomaly_detection.training.blind_features import prepare_blind_feature_matrix


def test_prepare_blind_median_imputes_nan_before_scaling() -> None:
    """Missing numeric cells become finite after median fill (§5.1 robustness hook)."""
    table = pd.DataFrame({"f0": [1.0, np.nan, 4.5, -2.0], "f1": [10.0, 11.0, 12.0, 13.0]})
    x = prepare_blind_feature_matrix(table)
    assert x.shape == (4, 2)
    assert np.isfinite(x).all()


def test_prepare_blind_drops_supervised_columns() -> None:
    """Ground-truth columns must never enter the anomaly feature matrix."""
    table = pd.DataFrame({"f": [1.0, 2.0], "class": [0, 1]})
    x = prepare_blind_feature_matrix(table)
    assert x.shape == (2, 1)


def test_prepare_blind_rejects_only_non_numeric() -> None:
    """Bare string columns indicate a schema problem — fail loudly."""
    table = pd.DataFrame({"text": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="No numeric"):
        _ = prepare_blind_feature_matrix(table)
