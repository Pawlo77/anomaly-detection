"""Tests for orchestration lifecycle policies and aggregate logging."""

from types import SimpleNamespace

from anomaly_detection.config import MlflowSettings
from anomaly_detection.orchestration import cli as orch_cli
from anomaly_detection.orchestration import reconcile as orch_reconcile
from anomaly_detection.orchestration import scheduler as orch_scheduler


class _FakeClient:
    """Minimal MLflow client spy used for lifecycle tests."""

    def __init__(self, *_: object, **__: object) -> None:
        self.deleted: list[str] = []
        self.terminated: list[tuple[str, str]] = []
        self.tags: list[tuple[str, str, str]] = []

    def set_terminated(self, run_id: str, status: str) -> None:
        self.terminated.append((run_id, status))

    def delete_run(self, run_id: str) -> None:
        self.deleted.append(run_id)

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self.tags.append((run_id, key, value))

    def log_metric(self, *_: object, **__: object) -> None:
        return None

    def log_artifact(self, *_: object, **__: object) -> None:
        return None


def test_scheduler_delete_policy_toggle(monkeypatch) -> None:
    """Fail/kill handlers should respect delete_incomplete_runs setting."""
    fake = _FakeClient()
    monkeypatch.setattr(orch_scheduler, "MlflowClient", lambda **_: fake)

    keep_settings = MlflowSettings(delete_incomplete_runs=False)
    orch_scheduler._fail_run("run-1", RuntimeError("boom"), keep_settings)
    orch_scheduler._kill_run("run-2", keep_settings)
    assert fake.deleted == []

    drop_settings = MlflowSettings(delete_incomplete_runs=True)
    orch_scheduler._fail_run("run-3", RuntimeError("boom"), drop_settings)
    orch_scheduler._kill_run("run-4", drop_settings)
    assert "run-3" in fake.deleted
    assert "run-4" in fake.deleted


def test_reconcile_stale_policy_toggle() -> None:
    """Stale-running reconciliation should optionally soft-delete runs."""
    fake = _FakeClient()
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="run-x"),
        data=SimpleNamespace(metrics={"system.heartbeat_unix": 0.0}),
    )
    orch_reconcile._kill_if_stale(  # type: ignore[attr-defined]
        client=fake,
        run=run,
        heartbeat_key="system.heartbeat_unix",
        now=10_000.0,
        stale_ttl_seconds=1,
        delete_incomplete_runs=False,
    )
    assert fake.deleted == []

    orch_reconcile._kill_if_stale(  # type: ignore[attr-defined]
        client=fake,
        run=run,
        heartbeat_key="system.heartbeat_unix",
        now=10_000.0,
        stale_ttl_seconds=1,
        delete_incomplete_runs=True,
    )
    assert fake.deleted == ["run-x"]


def test_cli_aggregate_tag_constant() -> None:
    """CLI should expose single aggregate task-type constant for consistency."""
    assert orch_cli.AGGREGATE_TASK_TYPE == "aggregate"
