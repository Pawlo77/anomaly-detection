"""Tests for orchestration manifest and reconciliation primitives."""

from dataclasses import replace
from types import SimpleNamespace

from anomaly_detection.config import MlflowSettings
from anomaly_detection.orchestration.manifest import build_manifest
from anomaly_detection.orchestration.reconcile import ReconcileResult, reconcile_manifest
from anomaly_detection.training.tasks import ExperimentTask


def test_track_a_benchmark_skips_arrhythmia_lof_dbscan_without_pca() -> None:
    """Plan §1.3 keeps LOF/DBSCAN arrhythmia runs on PCA view only."""
    manifest = build_manifest("phase4")
    offending = [
        task
        for task in manifest.tasks
        if task.dataset_id == "arrhythmia"
        and task.algorithm in {"LOF", "DBSCAN"}
        and task.params.get("study") == "benchmark"
    ]
    assert not offending


def test_oracle_phase_manifest_is_non_empty() -> None:
    """Oracle pack should schedule Track B grids plus sensitivity aggregates."""
    manifest = build_manifest("oracle")
    studies = {task.params.get("study") for task in manifest.tasks}
    assert "sobol_iforest" in studies
    assert "sobol_ocsvm" in studies
    assert "lhs_ocsvm" in studies
    assert "bootstrap_stable_iforest" in studies
    assert "oracle_ocsvm_rbf" in studies


def test_manifest_is_deterministic_and_unique() -> None:
    """Manifest generation should be stable and run keys unique."""
    manifest_a = build_manifest("phase4")
    manifest_b = build_manifest("phase4")
    assert len(manifest_a.tasks) > 0
    assert [task.run_key for task in manifest_a.tasks] == [
        task.run_key for task in manifest_b.tasks
    ]
    assert len({task.run_key for task in manifest_a.tasks}) == len(manifest_a.tasks)


def test_reconcile_marks_finished_and_pending(monkeypatch) -> None:
    """Reconciliation should skip FINISHED runs and keep missing runs pending."""
    task_done = ExperimentTask("phase4", "arrhythmia", "IForest", 42, 1)
    task_pending = replace(task_done, task_index=2, seed=0)
    run_done = SimpleNamespace(
        info=SimpleNamespace(run_id="done", status="FINISHED"),
        data=SimpleNamespace(metrics={}, tags={"run_key": task_done.run_key}),
    )

    def _fake_collect_runs(*_: object, **__: object) -> dict[str, tuple[SimpleNamespace, ...]]:
        return {task_done.run_key: (run_done,)}

    monkeypatch.setattr(
        "anomaly_detection.orchestration.reconcile._collect_runs", _fake_collect_runs
    )
    result: ReconcileResult = reconcile_manifest(
        tasks=(task_done, task_pending),
        settings=MlflowSettings(),
        stale_ttl_seconds=600,
        fail_on_duplicate_running=True,
    )
    assert task_done.run_key in result.finished_run_keys
    assert result.pending_tasks == (task_pending,)
