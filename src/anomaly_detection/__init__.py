"""Root package for tabular anomaly detection experiments.

Installing or importing this module configures a baseline ``logging`` formatter
suitable for scripts. Applications should normally configure logging
explicitly instead of relying on import side effects.
"""

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

__all__: list[str] = []
