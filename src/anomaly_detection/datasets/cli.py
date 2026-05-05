"""CLI façade delegating dataset builds to ``run_datasets_pipeline``."""

import logging

from ..warnings_filters import apply_known_sklearn_experiment_warnings
from .pipeline import run_datasets_pipeline

CLI_COMMAND_NAME = "datasets"
"""Logical command name for datasets pipeline entrypoint."""

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
"""Default log format for datasets CLI."""


def main() -> None:
    """Configure logging verbosity then execute every pipeline stage serially."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    apply_known_sklearn_experiment_warnings()
    run_datasets_pipeline()


if __name__ == "__main__":
    main()
