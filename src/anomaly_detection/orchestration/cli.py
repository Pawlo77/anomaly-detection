"""CLI entrypoint wiring manifest reconciliation, scheduling, and exports."""

import argparse
import logging
from pathlib import Path

from mlflow.tracking import MlflowClient

from ..config import MlflowSettings, OrchestrationSettings
from ..training.phase5 import export_phase5_labels
from ..warnings_filters import apply_known_sklearn_experiment_warnings
from .manifest import build_manifest
from .reconcile import reconcile_manifest
from .reporting import export_phase4_summary
from .scheduler import execute_tasks

PHASE5_EXPORT_FILENAME = "test_labels.csv"
"""Default output filename for phase-5 aggregate export."""

PHASE5_AGGREGATE_RUN_NAME = "phase5__aggregate__final_export"
"""MLflow run name for phase-5 aggregate export artifact."""

PHASE4_SUMMARY_RUN_NAME = "phase4__aggregate__summary"
"""MLflow run name for phase-4 aggregate summary artifact."""

AGGREGATE_TASK_TYPE = "aggregate"
"""Common task type tag value for aggregate MLflow runs."""

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
"""Structured console logging for orchestration CLI."""

_ORCH_LOGGER = logging.getLogger("anomaly_detection.orchestration")


def build_parser() -> argparse.ArgumentParser:
    """Configure ``argparse`` for phase selection and parallelism overrides.

    Returns:
        Parser ready for ``parse_args`` without executing side effects yet.
    """
    parser = argparse.ArgumentParser(description="Run/resume anomaly detection experiments.")
    parser.add_argument(
        "--phase",
        type=str,
        default=None,
        help="Phase: phase4, oracle, phase5, or all (default from settings).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel worker count (default from settings, usually 4).",
    )
    return parser


def main() -> None:
    """Parse CLI arguments, reconcile, schedule tasks, and emit summaries.

    Writes phase-4 aggregate CSV artifacts and optionally logs companion MLflow runs
    alongside phase-five blind-export predictions when executions succeed cleanly.
    """
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    apply_known_sklearn_experiment_warnings()

    parser = build_parser()
    args = parser.parse_args()

    orch_settings = OrchestrationSettings()
    mlflow_settings = MlflowSettings()

    phase = args.phase if args.phase is not None else orch_settings.phase
    jobs = args.jobs if args.jobs is not None else orch_settings.jobs

    manifest = build_manifest(phase=phase)
    _ORCH_LOGGER.info(
        "Manifest built: phase=%s tasks=%d",
        phase,
        len(manifest.tasks),
    )
    reconciled = reconcile_manifest(
        tasks=manifest.tasks,
        settings=mlflow_settings,
        stale_ttl_seconds=orch_settings.stale_ttl_seconds,
        fail_on_duplicate_running=orch_settings.fail_on_duplicate_running,
    )
    _ORCH_LOGGER.info(
        "Reconciled: pending=%d already_finished=%d",
        len(reconciled.pending_tasks),
        len(reconciled.finished_run_keys),
    )

    summary = execute_tasks(
        tasks=reconciled.pending_tasks,
        jobs=jobs,
        heavy_max_concurrent=orch_settings.heavy_max_concurrent,
        mlflow_settings=mlflow_settings,
    )
    if phase in {"phase4", "oracle", "all"} and summary.failed == 0:
        summary_path = export_phase4_summary(settings=mlflow_settings)
        if summary_path is not None:
            client = MlflowClient(tracking_uri=mlflow_settings.tracking_uri)
            experiment = client.get_experiment_by_name(mlflow_settings.experiment_name_phase4)
            if experiment is not None:
                run = client.create_run(
                    experiment_id=experiment.experiment_id,
                    tags={
                        "mlflow.runName": PHASE4_SUMMARY_RUN_NAME,
                        "phase": "phase4",
                        "task_type": AGGREGATE_TASK_TYPE,
                    },
                )
                client.log_artifact(run.info.run_id, str(summary_path.resolve()))
                client.set_terminated(run.info.run_id, status="FINISHED")
    if phase in {"phase5", "all"} and summary.failed == 0:
        report = export_phase5_labels(output_path=Path(PHASE5_EXPORT_FILENAME))
        client = MlflowClient(tracking_uri=mlflow_settings.tracking_uri)
        experiment = client.get_experiment_by_name(mlflow_settings.experiment_name_phase5)
        if experiment is not None:
            run = client.create_run(
                experiment_id=experiment.experiment_id,
                tags={
                    "mlflow.runName": PHASE5_AGGREGATE_RUN_NAME,
                    "phase": "phase5",
                    "task_type": AGGREGATE_TASK_TYPE,
                },
            )
            client.log_metric(run.info.run_id, "rows", float(report.rows))
            client.log_metric(run.info.run_id, "outliers", float(report.outliers))
            client.log_metric(
                run.info.run_id, "contamination_estimate", report.contamination_estimate
            )
            client.log_metric(
                run.info.run_id,
                "distance_concentration_ratio",
                report.distance_concentration_ratio,
            )
            client.log_artifact(run.info.run_id, str(report.output_path.resolve()))
            client.set_terminated(run.info.run_id, status="FINISHED")
    print(
        "experiments done:",
        f"completed={summary.completed}",
        f"failed={summary.failed}",
        f"killed={summary.killed}",
        f"already_finished={len(reconciled.finished_run_keys)}",
    )


if __name__ == "__main__":
    main()
