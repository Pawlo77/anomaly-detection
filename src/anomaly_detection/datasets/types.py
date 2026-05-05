"""Core typed models for dataset subpackage."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

LABEL_COLUMN_NAME = "label"
"""Canonical binary label column name."""

DatasetId = Literal[
    "annthyroid",
    "arrhythmia",
    "breastw",
    "cardio",
    "cover",
    "glass",
    "http",
    "ionosphere",
    "letter",
    "lympho",
    "mammography",
    "mnist",
    "musk",
    "optdigits",
    "pendigits",
    "pima",
    "satellite",
    "satimage-2",
    "shuttle",
    "smtp",
    "speech",
    "thyroid",
    "vertebral",
    "vowels",
    "wbc",
    "wine",
    "wut/smile",
    "wut/circles",
    "wut/isolation",
    "wut/windows",
    "wut/x2",
    "sipu/compound",
    "sipu/jain",
    "sipu/flame",
    "sipu/spiral",
    "fcps/lsun",
    "graves/fuzzyx",
]


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """In-memory dataset representation.

    Attributes:
        dataset_id: Dataset identifier from project catalog.
        view: Materialized view name, e.g. raw/preprocessed/pca_95.
        X: Feature matrix with numeric columns only.
        y: Binary labels where 1 means outlier.
        source_path: Path to persisted canonical table.
    """

    dataset_id: str
    view: str
    X: pd.DataFrame
    y: pd.Series
    source_path: Path
