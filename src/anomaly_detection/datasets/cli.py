"""CLI entry point for dataset pipeline."""

import logging

from .pipeline import run_datasets_pipeline

CLI_COMMAND_NAME = "datasets"
"""Logical command name for datasets pipeline entrypoint."""

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
"""Default log format for datasets CLI."""


def main() -> None:
    """Run full datasets pipeline end-to-end."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    run_datasets_pipeline()


if __name__ == "__main__":
    main()
