"""Tests for blind-input tabular audits."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from anomaly_detection.training.blind_audit import audit_blind_input_table, write_blind_audit_json


def test_audit_blind_input_table_records_numeric_range() -> None:
    table = pd.DataFrame({"a": [1.0, 2.0], "meta": ["x", "y"]})
    profile = audit_blind_input_table(table)
    assert profile["n_rows"] == 2
    names = {c["name"] for c in profile["columns"]}
    assert names == {"a", "meta"}
    a_entry = next(c for c in profile["columns"] if c["name"] == "a")
    assert "numeric_min" in a_entry


def test_write_blind_audit_json_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"f": [1.0, np.nan]})
    path = tmp_path / "p.json"
    write_blind_audit_json(df, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["n_rows"] == 2
