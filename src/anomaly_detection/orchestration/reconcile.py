"""MLflow run reconciliation for resumable orchestration."""

from dataclasses import dataclass
from time import time

from mlflow import MlflowClient
from mlflow.entities import Run

from ..config import MlflowSettings
from ..training.tasks import ExperimentTask

TERMINAL_DONE = {"FINISHED"}
"""MLflow statuses treated as completed for resume logic."""


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    """Result of manifest-vs-MLflow reconciliation.

    Attributes:
        finished_run_keys: Run keys already completed in MLflow.
        pending_tasks: Tasks still requiring execution after reconciliation.
    """

    finished_run_keys: frozenset[str]
    pending_tasks: tuple[ExperimentTask, ...]


def reconcile_manifest(
    tasks: tuple[ExperimentTask, ...],
    settings: MlflowSettings,
    stale_ttl_seconds: int,
    fail_on_duplicate_running: bool,
) -> ReconcileResult:
    """Filter manifest tasks against MLflow run history for crash-safe resumes.

    Args:
        tasks: Planned tasks from manifest generation.
        settings: Tracking configuration locating experiments and URIs.
        stale_ttl_seconds: Heartbeat TTL for deciding abandoned RUNNING rows.
        fail_on_duplicate_running: Whether duplicated RUNNING states raise.

    Returns:
        Keys known to be FINISHED versus tasks still awaiting execution.

    Raises:
        RuntimeError: When ``fail_on_duplicate_running`` and duplicates exist.
    """
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    run_index = _collect_runs(client, settings)
    now = time()

    finished: set[str] = set()
    pending: list[ExperimentTask] = []
    for task in tasks:
        runs = run_index.get(task.run_key, ())
        if not runs:
            pending.append(task)
            continue

        completed = False
        running_count = 0
        for run in runs:
            status = run.info.status
            if status in TERMINAL_DONE:
                completed = True
                break
            if status == "RUNNING":
                running_count += 1
                _kill_if_stale(
                    client=client,
                    run=run,
                    heartbeat_key=settings.heartbeat_metric_key,
                    now=now,
                    stale_ttl_seconds=stale_ttl_seconds,
                    delete_incomplete_runs=settings.delete_incomplete_runs,
                )

        if fail_on_duplicate_running and running_count > 1:
            raise RuntimeError(f"Duplicate RUNNING runs found for key={task.run_key}")

        if completed:
            finished.add(task.run_key)
        else:
            pending.append(task)

    return ReconcileResult(finished_run_keys=frozenset(finished), pending_tasks=tuple(pending))


def _collect_runs(client: MlflowClient, settings: MlflowSettings) -> dict[str, tuple[Run, ...]]:
    """Index searchable MLflow runs by deterministic ``run_key`` tag.

    Args:
        client: Active MLflow tracking client bound to backing store.
        settings: Experiment naming configuration for phases four and five.

    Returns:
        Mapping from normalized run keys to chronological run tuples per key.
    """
    grouped: dict[str, list[Run]] = {}
    for experiment_name in (settings.experiment_name_phase4, settings.experiment_name_phase5):
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            continue
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status != 'SCHEDULED'",
            max_results=50_000,
        )
        for run in runs:
            run_key = run.data.tags.get("run_key")
            if not run_key:
                continue
            grouped.setdefault(run_key, []).append(run)
    return {key: tuple(values) for key, values in grouped.items()}


def _kill_if_stale(
    client: MlflowClient,
    run: Run,
    heartbeat_key: str,
    now: float,
    stale_ttl_seconds: int,
    delete_incomplete_runs: bool,
) -> None:
    """Terminate stuck RUNNING entries based on heartbeat age.

    Args:
        client: MLflow tracking client with mutation permissions.
        run: Candidate run flagged RUNNING server-side.
        heartbeat_key: Metric name storing Unix heartbeat timestamps.
        now: Caller-provided clock for TTL comparison.
        stale_ttl_seconds: Elapsed seconds after which heartbeat is stale.
        delete_incomplete_runs: Optionally hard-delete orphaned runs after kill.
    """
    metrics = run.data.metrics
    heartbeat = metrics.get(heartbeat_key)
    if heartbeat is None:
        return
    if (now - float(heartbeat)) > stale_ttl_seconds:
        client.set_terminated(run.info.run_id, status="KILLED")
        if delete_incomplete_runs:
            client.delete_run(run.info.run_id)
