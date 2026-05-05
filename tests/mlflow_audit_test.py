"""Tests for MLflow metric contract auditing."""

from pathlib import Path
from time import time

import pytest
from mlflow.tracking import MlflowClient

from anomaly_detection.config import MlflowSettings
from anomaly_detection.orchestration.mlflow_audit import (
    audit_phase4_task_runs,
    format_report,
    required_metric_keys,
)


@pytest.fixture
def isolated_mlflow_settings(tmp_path: Path) -> MlflowSettings:
    """SQLite backing store local to the test."""
    db = tmp_path / "audit.sqlite"
    uri = f"sqlite:///{db.resolve().as_posix()}"
    return MlflowSettings(tracking_uri=uri)


def test_required_keys_align_with_metrics_report() -> None:
    """Core keys must stay synchronized with ``MetricsReport`` fields."""
    settings = MlflowSettings()
    req = required_metric_keys(settings)
    assert "pr_auc" in req
    assert "mcc" in req
    assert "runtime_seconds" in req
    assert settings.heartbeat_metric_key in req
    assert "subsampled" not in req


def test_audit_empty_experiment(isolated_mlflow_settings: MlflowSettings) -> None:
    """Missing experiment yields zero violations (nothing to check)."""
    report = audit_phase4_task_runs(isolated_mlflow_settings)
    assert not report.experiment_found
    assert report.task_runs_audited == 0
    assert report.ok


def test_audit_passes_when_all_metrics_logged(isolated_mlflow_settings: MlflowSettings) -> None:
    client = MlflowClient(isolated_mlflow_settings.tracking_uri)
    exp_id = client.create_experiment(isolated_mlflow_settings.experiment_name_phase4)
    run = client.create_run(
        exp_id,
        tags={
            "task_type": "default",
            "dataset": "annthyroid",
            "algorithm": "IForest",
            "run_name": "smoke",
        },
    )
    rid = run.info.run_id
    for key in required_metric_keys(isolated_mlflow_settings):
        val = float(time()) if key == isolated_mlflow_settings.heartbeat_metric_key else 0.42
        client.log_metric(rid, key, val)
    client.set_terminated(rid, "FINISHED")

    report = audit_phase4_task_runs(isolated_mlflow_settings)
    assert report.experiment_found
    assert report.task_runs_audited == 1
    assert report.ok
    assert "All audited" in format_report(report)


def test_audit_flags_missing_metrics(isolated_mlflow_settings: MlflowSettings) -> None:
    client = MlflowClient(isolated_mlflow_settings.tracking_uri)
    exp_id = client.create_experiment(isolated_mlflow_settings.experiment_name_phase4)
    run = client.create_run(exp_id, tags={"task_type": "default"})
    rid = run.info.run_id
    client.log_metric(rid, "pr_auc", 0.1)
    client.set_terminated(rid, "FINISHED")

    report = audit_phase4_task_runs(isolated_mlflow_settings)
    assert not report.ok
    assert len(report.issues) == 1
    assert "missing_metrics" in report.issues[0].problems[0]


def test_aggregate_runs_skipped(isolated_mlflow_settings: MlflowSettings) -> None:
    client = MlflowClient(isolated_mlflow_settings.tracking_uri)
    exp_id = client.create_experiment(isolated_mlflow_settings.experiment_name_phase4)
    run = client.create_run(
        exp_id,
        tags={"task_type": "aggregate", "phase": "phase4"},
    )
    rid = run.info.run_id
    client.log_metric(rid, "foo", 1.0)
    client.set_terminated(rid, "FINISHED")

    report = audit_phase4_task_runs(isolated_mlflow_settings)
    assert report.ok
    assert report.task_runs_audited == 0
    assert report.aggregate_runs_skipped == 1
