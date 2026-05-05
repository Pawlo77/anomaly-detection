"""Process-pool scheduler wiring ``run_task`` to durable MLflow logging."""

import json
import logging
import signal
import threading
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep, time
from typing import Any

from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient
from tqdm import tqdm

from ..config import MlflowSettings
from ..training.runner import TaskResult, run_task
from ..training.tasks import PROJECT_TAG_VALUE, ExperimentTask

HEAVY_ALGORITHMS = {"OCSVM", "LOF", "DBSCAN"}
"""Algorithms throttled by heavy-worker admission control."""

SCHEDULER_IDLE_SLEEP_SECONDS = 0.2
"""Sleep duration when scheduler queue is empty."""

SCHEDULER_WAIT_TIMEOUT_SECONDS = 0.5
"""Future wait timeout used in scheduler loop."""

MANIFEST_VERSION = "v1"
"""Manifest schema version tag logged to MLflow."""

_LOGGER = logging.getLogger(__name__)
"""Scheduler progress logger."""


@dataclass(frozen=True, slots=True)
class SchedulerSummary:
    """Execution summary for scheduler run.

    Attributes:
        completed: Number of tasks finished successfully.
        failed: Number of tasks finished with failure status.
        killed: Number of tasks terminated due to interrupt.
    """

    completed: int
    failed: int
    killed: int


class _InterruptState:
    """Latch shared between signal handlers and the scheduler loop."""

    def __init__(self) -> None:
        """Create an idle (not interrupted) coordination primitive."""
        self._event = threading.Event()

    def mark(self) -> None:
        """Record that graceful shutdown should begin."""
        self._event.set()

    @property
    def interrupted(self) -> bool:
        """Whether shutdown has been requested."""
        return self._event.is_set()


def execute_tasks(
    tasks: Iterable[ExperimentTask],
    jobs: int,
    heavy_max_concurrent: int,
    mlflow_settings: MlflowSettings,
) -> SchedulerSummary:
    """Execute workloads concurrently while persisting telemetry for every task.

    Args:
        tasks: Iterable of runnable tasks emitted by reconciliation.
        jobs: ``ProcessPoolExecutor`` worker saturation target.
        heavy_max_concurrent: Extra admission throttle for quadratic learners.
        mlflow_settings: Tracking backend configuration describing experiments.

    Returns:
        Aggregate counts distinguishing successful completions, failures,
        and runs cancelled due to interruption.
    """
    interrupt = _InterruptState()
    _install_signal_handlers(interrupt)

    pending = list(tasks)
    total_tasks = len(pending)
    active: dict[Future[TaskResult], tuple[ExperimentTask, str]] = {}
    completed = 0
    failed = 0
    killed = 0
    heavy_running = 0

    _LOGGER.info(
        "Starting scheduler: %d task(s), workers=%d, heavy_max_concurrent=%d",
        total_tasks,
        jobs,
        heavy_max_concurrent,
    )

    with (
        tqdm(
            total=total_tasks,
            desc="Experiments",
            unit="task",
            dynamic_ncols=True,
            miniters=1,
            mininterval=0.25,
        ) as pbar,
        ProcessPoolExecutor(max_workers=jobs) as pool,
    ):
        while pending or active:
            while pending and len(active) < jobs and not interrupt.interrupted:
                task = pending[0]
                is_heavy = task.algorithm in HEAVY_ALGORITHMS
                if is_heavy and heavy_running >= heavy_max_concurrent:
                    break
                pending.pop(0)
                run_id = _start_task_run(task, mlflow_settings)
                future = pool.submit(run_task, task)
                active[future] = (task, run_id)
                if is_heavy:
                    heavy_running += 1

            if not active:
                if interrupt.interrupted:
                    break
                sleep(SCHEDULER_IDLE_SLEEP_SECONDS)
                continue

            done, _ = wait(
                active.keys(),
                timeout=SCHEDULER_WAIT_TIMEOUT_SECONDS,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                _heartbeat_active(active.values(), mlflow_settings)
                continue

            for future in done:
                task, run_id = active.pop(future)
                if task.algorithm in HEAVY_ALGORITHMS:
                    heavy_running -= 1
                try:
                    result = future.result()
                except Exception as error:
                    failed += 1
                    _fail_run(run_id, error, mlflow_settings)
                    pbar.update(1)
                    pbar.set_postfix_str(f"{task.dataset_id}/{task.algorithm} FAIL", refresh=False)
                    continue
                completed += 1
                _finish_run(run_id, task, result, mlflow_settings)
                pbar.update(1)
                pbar.set_postfix_str(f"{task.dataset_id}/{task.algorithm}", refresh=False)

        if interrupt.interrupted:
            for future, (_, run_id) in list(active.items()):
                future.cancel()
                _kill_run(run_id, mlflow_settings)
                killed += 1
                pbar.update(1)
                pbar.set_postfix_str("interrupted", refresh=False)

    _LOGGER.info(
        "Scheduler finished: completed=%d failed=%d killed=%d",
        completed,
        failed,
        killed,
    )
    return SchedulerSummary(completed=completed, failed=failed, killed=killed)


def _start_task_run(task: ExperimentTask, settings: MlflowSettings) -> str:
    """Open MLflow bookkeeping for a spawned worker process.

    Args:
        task: Task metadata converted into tags/parameters.
        settings: Resolved tracking URI plus experiment segmentation.

    Returns:
       Opaque MLflow ``run_id`` string used for heartbeat + completion paths.
    """
    experiment_name = (
        settings.experiment_name_phase4
        if task.phase == "phase4"
        else settings.experiment_name_phase5
    )
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    tags_dict: dict[str, str] = {
        "mlflow.runName": task.run_name,
        "run_key": task.run_key,
        "run_name": task.run_name,
        "phase": task.phase,
        "dataset": task.dataset_id,
        "algorithm": task.algorithm,
        "seed": str(task.seed),
        "variant": task.variant,
        "task_type": task.task_type,
        "project": PROJECT_TAG_VALUE,
        "resumable": "true",
        "manifest_version": MANIFEST_VERSION,
        "config_hash": task.config_hash,
    }
    for key, value in sorted(task.tags.items()):
        tags_dict[key] = str(value)
    run = client.create_run(
        experiment_id,
        tags=tags_dict,
        run_name=task.run_name,
    )
    if task.params:
        client.log_batch(
            run_id=run.info.run_id,
            metrics=[Metric(settings.heartbeat_metric_key, float(time()), int(time() * 1000), 0)],
            params=[Param(key, str(value)) for key, value in sorted(task.params.items())],
            tags=[],
        )
    else:
        client.log_metric(run.info.run_id, settings.heartbeat_metric_key, float(time()))
    return run.info.run_id


def _finish_run(
    run_id: str,
    task: ExperimentTask,
    result: TaskResult,
    settings: MlflowSettings,
) -> None:
    """Write result payload and mark run finished."""
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    client.log_metric(run_id, settings.heartbeat_metric_key, float(time()))
    client.log_metric(run_id, "runtime_seconds", result.duration_seconds)
    for key, value in result.metrics.model_dump().items():
        client.log_metric(run_id, key, float(value))
    for key, value in sorted(result.extra_metrics.items()):
        client.log_metric(run_id, key, float(value))
    payload = {
        "run_key": result.run_key,
        "dataset_id": result.dataset_id,
        "algorithm": result.algorithm,
        "seed": result.seed,
        "phase": task.phase,
    }
    _log_json_artifact(client, run_id=run_id, filename="task_result.json", payload=payload)
    for artifact_name, artifact_payload in sorted(result.artifacts.items()):
        _log_json_artifact(
            client,
            run_id=run_id,
            filename=f"{artifact_name}.json",
            payload=artifact_payload,
        )
    client.set_terminated(run_id, status="FINISHED")


def _fail_run(run_id: str, error: Exception, settings: MlflowSettings) -> None:
    """Persist error details and mark run failed."""
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    client.log_metric(run_id, settings.heartbeat_metric_key, float(time()))
    payload = {"error_type": type(error).__name__, "message": str(error)}
    _log_json_artifact(client, run_id=run_id, filename="error.json", payload=payload)
    client.set_tag(run_id, "error.message", str(error)[:5000])
    client.set_terminated(run_id, status="FAILED")
    if settings.delete_incomplete_runs:
        client.delete_run(run_id)


def _kill_run(run_id: str, settings: MlflowSettings) -> None:
    """Mark run as killed then delete from active tracking view."""
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    client.set_terminated(run_id, status="KILLED")
    if settings.delete_incomplete_runs:
        client.delete_run(run_id)


def _heartbeat_active(
    active: Iterable[tuple[ExperimentTask, str]],
    settings: MlflowSettings,
) -> None:
    """Write heartbeat metric for all active runs."""
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    for _, run_id in active:
        client.log_metric(run_id, settings.heartbeat_metric_key, float(time()))


def _install_signal_handlers(interrupt: _InterruptState) -> None:
    """Install SIGINT/SIGTERM handlers for graceful shutdown."""

    def _handler(signum: int, _: Any) -> None:
        _ = signum
        interrupt.mark()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _log_json_artifact(
    client: MlflowClient,
    run_id: str,
    filename: str,
    payload: dict[str, Any],
) -> None:
    """Log JSON payload as run artifact with MLflow-client compatibility."""
    with TemporaryDirectory(prefix="anomaly-artifacts-") as temp_dir:
        target = Path(temp_dir) / filename
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        client.log_artifact(run_id=run_id, local_path=str(target))
