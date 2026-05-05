"""Shared numeric feature preparation pipeline for blind test CSV inputs (§5.1)."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

LOW_VARIANCE_STD_THRESHOLD = 0.01
"""Minimum feature standard deviation after scaling."""

HIGH_CORRELATION_THRESHOLD = 0.95
"""Correlation threshold used for redundant feature pruning."""


def prepare_blind_feature_matrix(table: pd.DataFrame) -> np.ndarray:
    """Drop labels, keep numeric columns, median-impute NaNs, scale, prune weak columns.

    Order matches the staged pipeline in plan §5.1-§5.2: handle missing values before
    ``RobustScaler``, then strip near-zero-variance columns, then prune near-duplicate
    features by correlation.

    Args:
        table: Parsed blind-test dataframe possibly containing stray label columns.

    Returns:
        ``float64`` matrix ready for multi-model ensembles.

    Raises:
        ValueError: When no usable numeric columns remain after filtering.
    """
    feature_table = table.copy()
    for label_col in ("class", "label"):
        if label_col in feature_table.columns:
            feature_table = feature_table.drop(columns=[label_col])

    numeric = feature_table.select_dtypes(include=[np.number, "bool"]).copy()
    if numeric.shape[1] == 0:
        msg = (
            "No numeric or boolean columns available for blind verification after "
            "dropping labels - check dtypes in the input CSV."
        )
        raise ValueError(msg)
    bool_cols = [c for c in numeric.columns if pd.api.types.is_bool_dtype(numeric[c])]
    for col in bool_cols:
        numeric[col] = numeric[col].astype(np.float64)

    if numeric.isna().any().any():
        imputed = SimpleImputer(strategy="median").fit_transform(numeric)
        numeric = pd.DataFrame(imputed, columns=numeric.columns, index=numeric.index)

    scaled = pd.DataFrame(
        RobustScaler().fit_transform(numeric.to_numpy(dtype=np.float64)),
        columns=numeric.columns,
        index=numeric.index,
    )
    low_variance = [
        column
        for column in scaled.columns
        if float(scaled[column].std(ddof=0)) < LOW_VARIANCE_STD_THRESHOLD
    ]
    if low_variance:
        scaled = scaled.drop(columns=low_variance)

    if scaled.shape[1] == 0:
        msg = "Every column was dropped as near-zero-variance after scaling."
        raise ValueError(msg)

    corr = scaled.corr(numeric_only=True).abs()
    drop_cols: set[str] = set()
    columns = list(corr.columns)
    for i, _left in enumerate(columns):
        for j in range(i + 1, len(columns)):
            if corr.iloc[i, j] > HIGH_CORRELATION_THRESHOLD:
                drop_cols.add(columns[j])
    if drop_cols:
        scaled = scaled.drop(columns=sorted(drop_cols))
    return np.asarray(scaled.to_numpy(dtype=np.float64), dtype=np.float64)
