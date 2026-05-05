"""Preprocessing pipelines for datasets."""

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

DEFAULT_PREPROCESS_SEED = 42
"""Default random seed for deterministic preprocessing components."""

ARRHYTHMIA_IMPUTER_MAX_ITER = 10
"""Maximum iterations for arrhythmia iterative imputer."""

ARRHYTHMIA_CATEGORICAL_MAX_UNIQUE = 20
"""Max unique count for treating integer-like arrhythmia columns as nominal."""

ARRHYTHMIA_CATEGORICAL_MAX_RATIO = 0.05
"""Max unique ratio for treating integer-like arrhythmia columns as nominal."""


@dataclass(frozen=True, slots=True)
class ArrhythmiaPreprocessReport:
    """Report payload for arrhythmia preprocessing run.

    Attributes:
        rows: Number of rows in input data frame.
        input_dim: Number of columns in input data frame.
        output_dim: Number of columns in output data frame after preprocessing.
        numeric_feature_count: Number of numeric features in input data frame.
        categorical_feature_count: Number of categorical features in input data frame.
        imputed_cell_fraction: Fraction of missing cells in numeric block before imputation.
        imputed_fraction_by_feature: Missing-value fraction by numeric feature before imputation.
    """

    rows: int
    input_dim: int
    output_dim: int
    numeric_feature_count: int
    categorical_feature_count: int
    imputed_cell_fraction: float
    imputed_fraction_by_feature: dict[str, float]


def _split_arrhythmia_feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer categorical and numeric columns for arrhythmia preprocessing.

    Object/category/bool columns are categorical. Integer-like columns with low cardinality
    are also treated as nominal to handle numerically encoded categories.

    Args:
        frame: Raw ``arrhythmia`` feature table excluding labels.

    Returns:
        Pair ``(categorical_columns, numeric_columns)``.
    """
    categorical: list[str] = []
    numeric: list[str] = []
    row_count = max(frame.shape[0], 1)
    for column in frame.columns:
        series = frame[column]
        if (
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series)
        ):
            categorical.append(column)
            continue
        if pd.api.types.is_integer_dtype(series):
            unique_count = int(series.nunique(dropna=True))
            unique_ratio = unique_count / float(row_count)
            if (
                unique_count <= ARRHYTHMIA_CATEGORICAL_MAX_UNIQUE
                and unique_ratio <= ARRHYTHMIA_CATEGORICAL_MAX_RATIO
            ):
                categorical.append(column)
                continue
        numeric.append(column)
    return categorical, numeric


def robust_scale_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply robust scaling on numeric feature table.

    Args:
        frame: Numeric feature matrix.

    Returns:
        Scaled numeric matrix as data frame.
    """
    scaler = RobustScaler()
    values = scaler.fit_transform(frame)
    return pd.DataFrame(values, columns=list(frame.columns), index=frame.index)


def preprocess_arrhythmia(frame: pd.DataFrame) -> pd.DataFrame:
    """Run plan-specific arrhythmia preprocessing branch.

    Args:
        frame: Raw arrhythmia feature matrix.

    Returns:
        Processed matrix with imputation and categorical encoding.
    """
    categorical, numeric = _split_arrhythmia_feature_types(frame)
    transformer = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        (
                            "imputer",
                            IterativeImputer(
                                estimator=BayesianRidge(),
                                max_iter=ARRHYTHMIA_IMPUTER_MAX_ITER,
                                random_state=DEFAULT_PREPROCESS_SEED,
                            ),
                        ),
                        ("scaler", RobustScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
    )
    transformed = transformer.fit_transform(frame)
    return pd.DataFrame(transformed, index=frame.index)


def preprocess_arrhythmia_with_report(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, ArrhythmiaPreprocessReport]:
    """Run arrhythmia preprocessing alongside missingness bookkeeping.

    Args:
        frame: Raw feature-rich arrhythmia table prior to iterative imputation.

    Returns:
        Tuple of engineered numeric matrix plus ``ArrhythmiaPreprocessReport``.
    """
    output = preprocess_arrhythmia(frame)
    categorical, numeric = _split_arrhythmia_feature_types(frame)
    if numeric:
        missing_by_feature = frame[numeric].isna().mean()
        imputed_fraction_by_feature = {col: float(missing_by_feature[col]) for col in numeric}
        imputed_cell_fraction = float(frame[numeric].isna().sum().sum()) / float(
            frame.shape[0] * len(numeric)
        )
    else:
        imputed_fraction_by_feature = {}
        imputed_cell_fraction = 0.0
    report = ArrhythmiaPreprocessReport(
        rows=frame.shape[0],
        input_dim=frame.shape[1],
        output_dim=output.shape[1],
        numeric_feature_count=len(numeric),
        categorical_feature_count=len(categorical),
        imputed_cell_fraction=imputed_cell_fraction,
        imputed_fraction_by_feature=imputed_fraction_by_feature,
    )
    return output, report
