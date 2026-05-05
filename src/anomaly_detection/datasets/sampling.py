"""Sampling helpers for large and bootstrap workloads."""

import pandas as pd

MIN_CLASS_SAMPLE = 1
"""Minimum samples taken from each class in stratified sampling."""


def stratified_subsample(
    frame: pd.DataFrame,
    label_column: str,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Take deterministic stratified subsample.

    Args:
        frame: Input table with label column.
        label_column: Column containing binary labels.
        n_samples: Output row count.
        seed: Random seed.

    Returns:
        Subsampled frame preserving label proportion.
    """
    if n_samples >= len(frame):
        return frame.copy()
    group_sizes = frame[label_column].value_counts(normalize=True)
    parts: list[pd.DataFrame] = []
    for label, frac in group_sizes.items():
        take = max(MIN_CLASS_SAMPLE, round(n_samples * frac))
        label_frame = frame[frame[label_column] == label]
        part = label_frame.sample(n=min(take, label_frame.shape[0]), random_state=seed)
        parts.append(part)
    combined = pd.concat(parts, axis=0).sample(n=n_samples, random_state=seed, replace=False)
    return combined.reset_index(drop=True)
