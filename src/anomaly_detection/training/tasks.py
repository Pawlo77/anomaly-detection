"""Typed task contracts for experiment execution."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

RUN_KEY_SEPARATOR = "__"
"""Separator token used in deterministic run-key generation."""

PROJECT_TAG_VALUE = "anomaly_detection"
"""Project tag value attached to all orchestrated MLflow runs."""


@dataclass(frozen=True, slots=True)
class ExperimentTask:
    """Single resumable model-fit task definition.

    Attributes:
        phase: Scheduling bucket (`phase4`, `phase5`). Oracle and sensitivity
            studies still advertise ``phase4`` but differentiate via ``params``.
        dataset_id: Dataset identifier from catalog.
        algorithm: Algorithm name matching model wrappers.
        seed: Deterministic random seed for the task.
        task_index: Stable index inside generated manifest.
        variant: Task variant identifier used in run naming and tags.
        task_type: Task category used in tracking (`fit` by default).
        params: Algorithm hyperparameters.
        tags: Additional run tags for tracking.
    """

    phase: str
    dataset_id: str
    algorithm: str
    seed: int
    task_index: int
    variant: str = "default"
    task_type: str = "fit"
    params: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def run_key(self) -> str:
        """Stable string reconciled against persisted MLflow ``run_key`` tags.

        Returns:
            Canonical identifier derived from dataset, estimator, seeds, variant.
        """
        return (
            f"{self.phase}{RUN_KEY_SEPARATOR}{self.dataset_id.replace('/', RUN_KEY_SEPARATOR)}"
            f"{RUN_KEY_SEPARATOR}{self.algorithm}{RUN_KEY_SEPARATOR}"
            f"seed-{self.seed}{RUN_KEY_SEPARATOR}task-{self.task_index:06d}"
        )

    @property
    def config_hash(self) -> str:
        """Short SHA-256 fingerprint of deterministic task JSON payload.

        Returns:
            Hex digest prefix used in MLflow run naming for configuration drift checks.
        """
        payload = {
            "phase": self.phase,
            "dataset_id": self.dataset_id,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "variant": self.variant,
            "task_type": self.task_type,
            "params": self.params,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:8]

    @property
    def run_name(self) -> str:
        """Human-visible MLflow run label combining dedupe key plus config slice.

        Returns:
            Concatenated ``run_key`` and ``cfg-<config_hash>`` token.
        """
        return f"{self.run_key}{RUN_KEY_SEPARATOR}cfg-{self.config_hash}"
