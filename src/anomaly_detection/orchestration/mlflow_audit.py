"""Validate MLflow persisted metrics against orchestration contracts.

The default tracking store is ``sqlite:///mlruns.db`` (not ``mlflow.db``); paths are
configurable via ``ANOMALY_MLFLOW_TRACKING_URI``.
"""

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

from mlflow.tracking import MlflowClient

from ..config import MlflowSettings
from ..metrics import MetricsReport

CORE_METRICS: frozenset[str] = frozenset(MetricsReport.model_fields.keys())
"""Metrics emitted from ``MetricsReport.model_dump()`` for every finished fit task."""

BASE_EXTRA_METRICS_PHASE4: frozenset[str] = frozenset(
    {
        "distance_concentration_ratio",
        "n_features",
        "n_samples_fit",
    }
)
"""Extras logged for phase-4 paths (Sobol/LHS/bootstrap).

See ``runner`` and ``sensitivity_execution``.
"""

SCHEDULER_METRICS: frozenset[str] = frozenset({"runtime_seconds"})
"""Always logged in ``_finish_run`` after a task completes."""

AGGREGATE_TASK_TYPES: frozenset[str] = frozenset({"aggregate"})
"""Runs created only for CSV exports / summaries — not scored detector tasks."""


def required_metric_keys(settings: MlflowSettings) -> frozenset[str]:
    """Return the full metric key set expected on a finished phase-4 *task* run."""
    return (
        CORE_METRICS
        | BASE_EXTRA_METRICS_PHASE4
        | SCHEDULER_METRICS
        | frozenset({settings.heartbeat_metric_key})
    )


@dataclass(frozen=True, slots=True)
class RunAuditIssue:
    """Single run-level finding."""

    run_id: str
    run_name: str | None
    problems: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Phase4AuditReport:
    """Summary of a phase-4 experiment audit."""

    tracking_uri: str
    experiment_name: str
    experiment_found: bool
    finished_runs_seen: int
    task_runs_audited: int
    aggregate_runs_skipped: int
    issues: tuple[RunAuditIssue, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """True when every audited task run satisfies the metric contract."""
        return len(self.issues) == 0


def audit_phase4_task_runs(settings: MlflowSettings | None = None) -> Phase4AuditReport:
    """Check FINISHED phase-4 runs for core metrics, extras, scheduler fields, finiteness.

    Skips aggregate export runs (``task_type=aggregate``), which only carry CSV artifacts.

    Args:
        settings: MLflow routing; defaults to environment-backed ``MlflowSettings``.

    Returns:
        Structured audit report (empty ``issues`` when valid).
    """
    cfg = settings or MlflowSettings()
    client = MlflowClient(tracking_uri=cfg.tracking_uri)
    required = required_metric_keys(cfg)

    experiment = client.get_experiment_by_name(cfg.experiment_name_phase4)
    if experiment is None:
        return Phase4AuditReport(
            tracking_uri=cfg.tracking_uri,
            experiment_name=cfg.experiment_name_phase4,
            experiment_found=False,
            finished_runs_seen=0,
            task_runs_audited=0,
            aggregate_runs_skipped=0,
            issues=(),
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=50_000,
    )

    issues: list[RunAuditIssue] = []
    aggregate_skipped = 0
    task_audited = 0

    for run in runs:
        tags = run.data.tags or {}
        task_type = tags.get("task_type", "default")
        if task_type in AGGREGATE_TASK_TYPES:
            aggregate_skipped += 1
            continue

        task_audited += 1
        metrics = dict(run.data.metrics or {})
        run_name = tags.get("run_name") or tags.get("run_key") or run.info.run_name
        problems: list[str] = []

        missing = sorted(required - metrics.keys())
        if missing:
            problems.append(f"missing_metrics={missing}")

        for key in sorted(required & metrics.keys()):
            val = metrics[key]
            try:
                fv = float(val)
            except (TypeError, ValueError):
                problems.append(f"non_numeric_metric:{key}={val!r}")
                continue
            if not math.isfinite(fv):
                problems.append(f"non_finite_metric:{key}={val!r}")

        if problems:
            issues.append(
                RunAuditIssue(
                    run_id=run.info.run_id,
                    run_name=run_name,
                    problems=tuple(problems),
                )
            )

    return Phase4AuditReport(
        tracking_uri=cfg.tracking_uri,
        experiment_name=cfg.experiment_name_phase4,
        experiment_found=True,
        finished_runs_seen=len(runs),
        task_runs_audited=task_audited,
        aggregate_runs_skipped=aggregate_skipped,
        issues=tuple(issues),
    )


def format_report(report: Phase4AuditReport) -> str:
    """Human-readable multi-line summary for CLI or logs."""
    lines = [
        f"tracking_uri={report.tracking_uri}",
        f"experiment={report.experiment_name!r} found={report.experiment_found}",
        f"finished_runs={report.finished_runs_seen} "
        f"task_runs_audited={report.task_runs_audited} "
        f"aggregate_skipped={report.aggregate_runs_skipped}",
    ]
    if not report.experiment_found:
        lines.append("No experiment — nothing to audit.")
        return "\n".join(lines)
    if report.ok:
        lines.append("All audited task runs have complete finite metrics.")
        return "\n".join(lines)
    lines.append(f"Issues ({len(report.issues)} run(s)):")
    for item in report.issues:
        lines.append(f"  run_id={item.run_id} name={item.run_name!r}")
        for p in item.problems:
            lines.append(f"    - {p}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: print audit and exit 1 if any run violates the metric contract."""
    parser = argparse.ArgumentParser(
        description="Audit MLflow phase-4 runs for expected detector metrics.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="Override MLflow tracking URI (default: MlflowSettings / env).",
    )
    args = parser.parse_args(argv)

    settings = (
        MlflowSettings(tracking_uri=args.tracking_uri) if args.tracking_uri else MlflowSettings()
    )
    db_hint = ""
    if settings.tracking_uri.startswith("sqlite:///"):
        raw = settings.tracking_uri.removeprefix("sqlite:///")
        if raw != ":memory:":
            db_hint = f" (SQLite file: {Path(raw).resolve()})"

    report = audit_phase4_task_runs(settings)
    print(format_report(report))
    if db_hint:
        print(db_hint)

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
