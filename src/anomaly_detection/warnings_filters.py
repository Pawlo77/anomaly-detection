"""Suppress sklearn warnings that are expected under benchmark loads but clutter logs."""



import warnings


def apply_known_sklearn_experiment_warnings() -> None:
    """Ignore noisy UserWarnings that do not invalidate comparative OD evaluation.

    Covers: LOF with duplicate coordinates (discrete / subsampled data), Isolation
    Forest when ``max_samples`` exceeds ``n_samples`` (small canonical sets), and
    thresholded metrics when prediction collapses to one class at the natural
    contamination cut.
    """
    # LOF: identical or duplicate nearest-neighbour ties on thin support.
    warnings.filterwarnings(
        "ignore",
        message=r".*Duplicate values are leading to incorrect results.*",
        category=UserWarning,
    )
    # IForest: sklearn clamps max_samples to n_samples and warns.
    warnings.filterwarnings(
        "ignore",
        message=r".*max_samples .* is greater than the total number of samples.*",
        category=UserWarning,
    )
    # Metrics: degenerate confusion matrices when labels/scores yield one predicted class.
    warnings.filterwarnings(
        "ignore",
        message=r".*A single label was found in 'y_true' and 'y_pred'.*",
        category=UserWarning,
    )
