"""This module contains helper functions and constants for the analysis notebooks."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
"""This module contains helper functions and constants for the analysis notebooks."""

ALGORITHM_ORDER: list[str] = ["OCSVM", "IForest", "LOF", "DBSCAN", "ECOD", "HBOS"]
"""Defines the order of algorithms for consistent plotting and analysis."""

ALGORITHM_PALETTE: dict[str, str] = {
    "OCSVM": "#4C78A8",
    "IForest": "#F58518",
    "LOF": "#54A24B",
    "DBSCAN": "#E45756",
    "ECOD": "#72B7B2",
    "HBOS": "#B279A2",
}
"""Defines a color palette for the algorithms for consistent plotting and analysis."""

HEATMAP_CMAP = "mako"
"""Shared colormap for all heatmaps across analysis notebooks."""

FIGURES_DIR = OUTPUTS_DIR / "figures"
"""Directory for saving analysis figures, created if it does not exist."""
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(filename: str) -> Path:
    """Save current matplotlib figure in a consistent location and format."""
    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path


def annotate_figure(fig: plt.Figure, text: str, x: float = 0.01, y: float = 0.01) -> None:
    """Attach a small setup note at the bottom of a figure."""
    fig.text(x, y, f"Setup: {text}", ha="left", va="bottom", fontsize=9, alpha=0.85)


def setup_theme() -> None:
    """Set up a consistent visual theme for all analysis plots."""
    sns.set_theme(style="whitegrid", context="talk")
    sns.set_palette([ALGORITHM_PALETTE[alg] for alg in ALGORITHM_ORDER])
    plt.rcParams.update(
        {
            "figure.figsize": (11, 6),
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.frameon": False,
        }
    )


def load_phase4_summary() -> pd.DataFrame:
    """Load the phase 4 summary results and ensure numeric columns are properly typed."""
    frame = pd.read_csv(OUTPUTS_DIR / "phase4_summary.csv")
    numeric_columns = [
        "pr_auc",
        "roc_auc",
        "mcc",
        "f1",
        "distance_concentration_ratio",
        "n_features",
        "pca_variance",
        "recall_type_g",
        "recall_type_l",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_descriptive_stats() -> pd.DataFrame:
    """Load the descriptive statistics and ensure numeric columns are properly typed."""
    frame = pd.read_csv(OUTPUTS_DIR / "descriptive_stats.csv")
    numeric_columns = [
        "n_rows",
        "n_features",
        "contamination",
        "outlier_count",
        "inlier_count",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def split_quantitative_cohorts(descriptive: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split datasets into valid quantitative and visual-only cohorts.

    Quantitative cohort requires at least one labeled outlier. Datasets with zero outliers
    are retained for visual/qualitative analyses only because thresholded positive-class
    metrics are undefined there.
    """
    if "outlier_count" not in descriptive.columns:
        raise KeyError("descriptive stats must include 'outlier_count'")
    quantitative = descriptive[descriptive["outlier_count"] > 0].copy()
    visual_only = descriptive[descriptive["outlier_count"] == 0].copy()
    return quantitative, visual_only


def contamination_stratum(value: float) -> str:
    """Categorize contamination levels into strata for analysis."""
    if value < 0.02:
        return "sparse (<2%)"
    if value < 0.1:
        return "low (2-10%)"
    if value < 0.3:
        return "mid (10-30%)"
    return "high (>=30%)"


def dimensionality_stratum(n_features: float) -> str:
    """Categorize dimensionality levels into strata for analysis."""
    if n_features < 15:
        return "low-d (<15)"
    if n_features < 80:
        return "mid-d (15-79)"
    return "high-d (>=80)"
