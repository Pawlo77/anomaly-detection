"""Pull finished MLflow runs into analyst-friendly CSV aggregates."""

from pathlib import Path

import pandas as pd
from mlflow.tracking import MlflowClient

from ..config import MlflowSettings

PHASE4_SUMMARY_FILENAME = "phase4_summary.csv"
"""Default CSV filename for aggregated phase-4 metrics."""


def export_phase4_summary(settings: MlflowSettings, output_path: Path | None = None) -> Path | None:
    """Materialize phase-4 metrics as a flattened CSV keyed by tracked tags.

    Args:
        settings: MLflow routing information for locating phase-four experiment.
        output_path: Optional explicit destination CSV path.

    Returns:
        Saved CSV path after successful export, otherwise ``None`` when there is
        no experiment or FINISHED payload to summarize.
    """
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    experiment = client.get_experiment_by_name(settings.experiment_name_phase4)
    if experiment is None:
        return None
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=50_000,
    )
    rows: list[dict[str, object]] = []
    for run in runs:
        tags = run.data.tags
        metrics = run.data.metrics
        rows.append(
            {
                "run_name": tags.get("run_name", tags.get("run_key", run.info.run_name)),
                "dataset": tags.get("dataset"),
                "algorithm": tags.get("algorithm"),
                "variant": tags.get("variant", "default"),
                "seed": tags.get("seed"),
                "pr_auc": metrics.get("pr_auc"),
                "roc_auc": metrics.get("roc_auc"),
                "mcc": metrics.get("mcc"),
                "f1": metrics.get("f1"),
                "distance_concentration_ratio": metrics.get("distance_concentration_ratio"),
                "n_features": metrics.get("n_features"),
                "pca_variance": metrics.get("pca_variance"),
                "recall_type_g": metrics.get("recall_type_g"),
                "recall_type_l": metrics.get("recall_type_l"),
            }
        )
    if not rows:
        return None
    table = pd.DataFrame(rows)
    target = output_path or Path(PHASE4_SUMMARY_FILENAME)
    table.to_csv(target, index=False)
    return target
