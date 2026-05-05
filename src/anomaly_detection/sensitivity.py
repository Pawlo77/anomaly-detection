"""Sensitivity analysis utilities for anomaly detection experiments.

This module implements the two sensitivity tracks from the experimental plan:

1. Sobol sensitivity indices over hyperparameter space.
2. Bootstrap stability analysis using PR-AUC dispersion.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .metrics import metric_pr_auc

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type TrainScoreCallable = Callable[[FloatArray], FloatArray]


class SensitivityError(RuntimeError):
    """Raised when sensitivity analysis cannot be completed."""


class HyperparameterSpec(BaseModel):
    """Typed hyperparameter specification for Sobol sampling.

    Attributes:
        name: Hyperparameter name.
        kind: Parameter type.
        lower: Lower bound for numeric parameters.
        upper: Upper bound for numeric parameters.
        choices: Allowed categorical values.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    kind: Literal["float", "int", "categorical"]
    lower: float | None = None
    upper: float | None = None
    choices: tuple[float | int | str, ...] | None = None

    @model_validator(mode="after")
    def validate_domain(self) -> "HyperparameterSpec":
        """Validate parameter domain coherence by parameter kind."""
        if self.kind in {"float", "int"}:
            if self.lower is None or self.upper is None:
                raise ValueError("Numeric hyperparameters require lower and upper bounds.")
            if self.lower >= self.upper:
                raise ValueError("Numeric hyperparameter requires lower < upper.")
        if self.kind == "categorical" and not self.choices:
            raise ValueError("Categorical hyperparameter requires non-empty choices.")
        return self

    def decode(self, sampled_value: float) -> float | int | str:
        """Decode Sobol sampled value into typed hyperparameter value.

        Args:
            sampled_value: Raw sampled value in configured bounds.

        Returns:
            Typed hyperparameter value.
        """
        if self.kind == "float":
            return float(sampled_value)
        if self.kind == "int":
            return round(sampled_value)

        assert self.choices is not None  # validated in model validator
        lower = float(self.lower if self.lower is not None else 0.0)
        upper = float(self.upper if self.upper is not None else len(self.choices) - 1)
        span = max(upper - lower, 1.0)
        normalized = (sampled_value - lower) / span
        index = int(np.clip(round(normalized * (len(self.choices) - 1)), 0, len(self.choices) - 1))
        return self.choices[index]

    @property
    def bounds(self) -> tuple[float, float]:
        """Numeric bounds tuples SALib consumes for continuous sampling axes.

        Returns:
            ``(low, high)`` pair; categorical knobs map to enumerated indices.
        """
        if self.kind == "categorical":
            assert self.choices is not None  # validated in model validator
            return (0.0, float(len(self.choices) - 1))
        assert self.lower is not None and self.upper is not None
        return (float(self.lower), float(self.upper))


class SobolSpace(BaseModel):
    """Sobol problem specification for model sensitivity.

    Attributes:
        parameters: Ordered hyperparameter definitions.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parameters: tuple[HyperparameterSpec, ...] = Field(min_length=1)

    @field_validator("parameters")
    @classmethod
    def validate_unique_names(
        cls, parameters: tuple[HyperparameterSpec, ...]
    ) -> tuple[HyperparameterSpec, ...]:
        """Validate Sobol axis labels remain injective for decoding.

        Returns:
            Echoed ``parameters`` tuple when uniqueness holds.

        Raises:
            ValueError: When duplicate textual names collide.
        """
        names = [parameter.name for parameter in parameters]
        if len(names) != len(set(names)):
            raise ValueError("Hyperparameter names must be unique.")
        return parameters

    @property
    def salib_problem(self) -> dict[str, Any]:
        """``problem`` dictionary expected by SALib sampling routines.

        Returns:
            Keys ``num_vars``, ``names``, and ``bounds`` for Saltelli tooling.
        """
        return {
            "num_vars": len(self.parameters),
            "names": [parameter.name for parameter in self.parameters],
            "bounds": [list(parameter.bounds) for parameter in self.parameters],
        }

    def decode_row(self, row: FloatArray) -> dict[str, float | int | str]:
        """Vectorize Sobol draws into estimator-ready keyword dictionaries.

        Args:
            row: Length-``num_vars`` float vector respecting parameter order.

        Returns:
            Mapping hyperparameter names to decoded scalars/strings.
        """
        values: dict[str, float | int | str] = {}
        for index, parameter in enumerate(self.parameters):
            values[parameter.name] = parameter.decode(float(row[index]))
        return values


class BootstrapStabilityReport(BaseModel):
    """Summary statistics for bootstrap PR-AUC stability.

    Attributes:
        mean_pr_auc: Mean PR-AUC across bootstrap runs.
        std_pr_auc: Standard deviation of PR-AUC.
        min_pr_auc: Minimum observed PR-AUC.
        max_pr_auc: Maximum observed PR-AUC.
        n_resamples: Number of bootstrap runs.
        sample_fraction: Fraction of data in each bootstrap sample.
        pr_auc_values: PR-AUC for each bootstrap run.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean_pr_auc: float = Field(ge=0.0, le=1.0)
    std_pr_auc: float = Field(ge=0.0, le=1.0)
    min_pr_auc: float = Field(ge=0.0, le=1.0)
    max_pr_auc: float = Field(ge=0.0, le=1.0)
    n_resamples: int = Field(ge=1)
    sample_fraction: float = Field(gt=0.0, le=1.0)
    pr_auc_values: tuple[float, ...] = Field(min_length=1)


class CandidateStability(BaseModel):
    """Per-configuration stability summary used for model selection.

    Attributes:
        params: Hyperparameter values for candidate.
        mean_pr_auc: Mean PR-AUC for candidate.
        std_pr_auc: PR-AUC standard deviation for candidate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    params: dict[str, float | int | str]
    mean_pr_auc: float = Field(ge=0.0, le=1.0)
    std_pr_auc: float = Field(ge=0.0, le=1.0)


@dataclass(slots=True, frozen=True)
class StabilityConstraint:
    """Constraint parameters for stable model selection.

    Attributes:
        max_std: Maximum acceptable PR-AUC standard deviation.
    """

    max_std: float = 0.03


def _require_salib() -> tuple[Any, Any]:
    """Import Saltelli helpers or raise actionable guidance.

    Returns:
        Tuple ``(saltelli_module, sobol_module)`` from SALib.

    Raises:
        SensitivityError: When SALib is not installed in the interpreter.
    """
    try:
        from SALib.analyze import sobol
        from SALib.sample import saltelli
    except ImportError as error:
        raise SensitivityError(
            "SALib is required for Sobol sensitivity analysis. Install dependency 'SALib'."
        ) from error
    return saltelli, sobol


def sobol_sample(
    space: SobolSpace, base_sample_size: int, calc_second_order: bool = False
) -> FloatArray:
    """Generate Sobol/Saltelli sample matrix for configured hyperparameter space.

    Args:
        space: Hyperparameter Sobol space.
        base_sample_size: Base sample count N for Saltelli sampling.
        calc_second_order: Include second-order effects in generated sample.

    Returns:
        Matrix of sampled numeric values.
    """
    if base_sample_size < 1:
        raise ValueError("base_sample_size must be >= 1.")
    saltelli, _ = _require_salib()
    samples = saltelli.sample(
        problem=space.salib_problem,
        N=base_sample_size,
        calc_second_order=calc_second_order,
    )
    return np.asarray(samples, dtype=np.float64)


def sobol_analyze(
    space: SobolSpace,
    objective_values: Sequence[float],
    calc_second_order: bool = False,
) -> dict[str, Any]:
    """Compute Sobol sensitivity indices for objective values.

    Args:
        space: Hyperparameter Sobol space.
        objective_values: Objective values aligned to Sobol sample order.
        calc_second_order: Compute second-order indices.

    Returns:
        SALib Sobol analysis result dictionary.
    """
    objective = np.asarray(objective_values, dtype=np.float64)
    if objective.ndim != 1:
        raise ValueError("objective_values must be one-dimensional.")
    _, sobol = _require_salib()
    return sobol.analyze(
        problem=space.salib_problem,
        Y=objective,
        calc_second_order=calc_second_order,
        print_to_console=False,
    )


def evaluate_sobol_objective(
    space: SobolSpace,
    sampled_values: FloatArray,
    objective: Callable[[dict[str, float | int | str]], float],
) -> FloatArray:
    """Evaluate objective over Sobol sample matrix.

    Args:
        space: Hyperparameter Sobol space.
        sampled_values: Numeric sample matrix.
        objective: Objective callable receiving decoded parameter dictionary.

    Returns:
        Objective vector aligned to sampled rows.
    """
    if sampled_values.ndim != 2:
        raise ValueError("sampled_values must be a two-dimensional matrix.")
    if sampled_values.shape[1] != len(space.parameters):
        raise ValueError("sampled_values columns must match Sobol parameter count.")

    values = np.empty(sampled_values.shape[0], dtype=np.float64)
    for index, row in enumerate(sampled_values):
        values[index] = float(objective(space.decode_row(row)))
    return values


def bootstrap_pr_auc_stability(
    x: npt.ArrayLike,
    y_true: npt.ArrayLike,
    score_fn: TrainScoreCallable,
    n_resamples: int = 10,
    sample_fraction: float = 0.9,
    random_state: int = 42,
) -> BootstrapStabilityReport:
    """Estimate PR-AUC stability via stratified bootstrap.

    This function fits/evaluates model scores inside each bootstrap resample,
    then computes PR-AUC for sampled data and reports dispersion.

    Args:
        x: Feature matrix ``(n_samples, n_features)``.
        y_true: Ground-truth labels ``(n_samples,)``.
        score_fn: Callable mapping sampled ``x`` to anomaly scores.
        n_resamples: Number of bootstrap runs.
        sample_fraction: Fraction of rows sampled in each run.
        random_state: Seed for reproducible resampling.

    Returns:
        Bootstrap stability report.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1.")
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError("sample_fraction must be in (0, 1].")

    x_matrix = np.asarray(x, dtype=np.float64)
    if x_matrix.ndim != 2:
        raise ValueError("x must be a two-dimensional matrix.")
    y_vector = np.asarray(y_true, dtype=np.int64)
    if y_vector.ndim != 1:
        raise ValueError("y_true must be a one-dimensional vector.")
    if x_matrix.shape[0] != y_vector.shape[0]:
        raise ValueError("x and y_true must have equal sample count.")

    rng = np.random.default_rng(random_state)
    target_size = max(1, round(x_matrix.shape[0] * sample_fraction))
    positive_idx = np.where(y_vector == 1)[0]
    negative_idx = np.where(y_vector == 0)[0]
    use_stratified = positive_idx.size > 0 and negative_idx.size > 0

    pr_auc_values: list[float] = []
    for _ in range(n_resamples):
        if use_stratified:
            positive_take = round(target_size * (positive_idx.size / y_vector.size))
            positive_take = int(np.clip(positive_take, 1, target_size - 1))
            negative_take = target_size - positive_take
            boot_positive = rng.choice(positive_idx, size=positive_take, replace=True)
            boot_negative = rng.choice(negative_idx, size=negative_take, replace=True)
            sampled_indices = np.concatenate((boot_positive, boot_negative))
            rng.shuffle(sampled_indices)
        else:
            sampled_indices = rng.choice(np.arange(y_vector.size), size=target_size, replace=True)

        x_sample = x_matrix[sampled_indices]
        y_sample = y_vector[sampled_indices]
        scores = np.asarray(score_fn(x_sample), dtype=np.float64)
        if scores.shape != (x_sample.shape[0],):
            raise ValueError("score_fn must return vector with shape (n_samples,).")
        pr_auc_values.append(metric_pr_auc(y_sample, scores))

    pr_auc_array = np.asarray(pr_auc_values, dtype=np.float64)
    return BootstrapStabilityReport(
        mean_pr_auc=float(np.mean(pr_auc_array)),
        std_pr_auc=float(np.std(pr_auc_array, ddof=0)),
        min_pr_auc=float(np.min(pr_auc_array)),
        max_pr_auc=float(np.max(pr_auc_array)),
        n_resamples=n_resamples,
        sample_fraction=sample_fraction,
        pr_auc_values=tuple(float(value) for value in pr_auc_array),
    )


def select_stable_best(
    candidates: Sequence[CandidateStability],
    constraint: StabilityConstraint | None = None,
) -> CandidateStability:
    """Select best candidate under PR-AUC stability constraint.

    Selection rule:
    1. Keep candidates with ``std_pr_auc <= max_std``.
    2. Pick candidate with highest mean PR-AUC.
    3. Break ties by lower ``std_pr_auc``.

    Args:
        candidates: Candidate stability summaries.
        constraint: Stability constraint values.

    Returns:
        Best stable candidate.

    Raises:
        ValueError: If candidates list is empty.
        SensitivityError: If no candidate satisfies stability constraint.
    """
    active_constraint = constraint or StabilityConstraint()
    if not candidates:
        raise ValueError("candidates cannot be empty.")
    stable = [
        candidate for candidate in candidates if candidate.std_pr_auc <= active_constraint.max_std
    ]
    if not stable:
        raise SensitivityError(
            f"No candidate satisfies stability constraint std <= {active_constraint.max_std:.4f}."
        )
    return max(stable, key=lambda item: (item.mean_pr_auc, -item.std_pr_auc))
