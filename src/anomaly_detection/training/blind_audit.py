"""Structural audits for blind-phase CSV inputs ahead of preprocessing (§5.1)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def audit_blind_input_table(table: pd.DataFrame) -> dict[str, Any]:
    """Summarize dtypes, missingness, and numeric ranges prior to preprocessing.

    Args:
        table: Blind-test frame exactly as parsed from CSV.

    Returns:
        JSON-serializable dict describing coarse column-level summaries.
    """
    profile: dict[str, Any] = {
        "n_rows": len(table),
        "has_class": bool("class" in table.columns),
        "has_label": bool("label" in table.columns),
        "columns": [],
    }
    for column in table.columns:
        series = table[column]
        entry: dict[str, Any] = {
            "name": column,
            "dtype": str(series.dtype),
            "missing_fraction": float(series.isna().mean()),
        }
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            vc = numeric.dropna()
            entry["numeric_min"] = float(np.min(vc))
            entry["numeric_max"] = float(np.max(vc))
        profile["columns"].append(entry)
    return profile


def write_blind_audit_json(table: pd.DataFrame, destination: Path) -> Path:
    """Persist structured audit payloads for downstream provenance tooling.

    Args:
        table: Blind-test dataframe snapshot.
        destination: Absolute or relative filesystem target for JSON text.

    Returns:
        Resolved path after writing indented JSON artifacts.
    """
    destination.write_text(
        json.dumps(audit_blind_input_table(table), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination
