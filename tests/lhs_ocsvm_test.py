"""Latin Hypercube OCSVM sampling tests."""

from anomaly_detection.training.lhs_ocsvm import lhs_dict_rows, lhs_rbf_ocsvm_configs


def test_lhs_draw_count_equals_requested_sample_size() -> None:
    rows = lhs_rbf_ocsvm_configs(n_samples=80, seed=7)
    assert len(rows) == 80
    cells = {(round(r.nu, 6), str(r.gamma)) for r in rows}
    assert len(cells) <= len(rows)


def test_lhs_dict_rows_match_param_grid() -> None:
    rows = lhs_dict_rows(n_samples=10, seed=11)
    assert len(rows) == 10
    for row in rows:
        assert row["nu"] in {0.01, 0.05, 0.10, 0.20, 0.35, 0.50}
        assert row["gamma"] in {"scale", "auto", 0.001, 0.01, 0.1, 1.0, 10.0}
