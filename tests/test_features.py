"""Per-subgraph features for the classical WP4 baseline, on hand-built cache rows."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from qlmaml.features import subgraph_features

T0 = datetime(2022, 9, 1, 8, 0)


def _row(n: int, edges: list[tuple[int, int, float, int, float]]) -> dict:
    """A cache row (see `qlmaml.subgraphs.write_cache`): edges are (u, v, total_usd, n_tx, hours after T0)."""
    return {
        "n_qubits": n,
        "edges": [{"u": u, "v": v, "total_usd": a, "n_tx": c, "first_ts": T0 + timedelta(hours=h)}
                  for u, v, a, c, h in edges],
    }


def test_fan_out_counts_and_degrees():
    f = subgraph_features(_row(4, [(0, 1, 100.0, 1, 0), (0, 2, 100.0, 2, 1), (0, 3, 100.0, 1, 2)]))

    assert f["n_nodes"] == 4
    assert f["n_edges"] == 3
    assert f["n_tx"] == 4
    assert f["out_deg_max"] == 3
    assert f["in_deg_max"] == 1
    assert f["n_sources"] == 1  # account 0 only pays
    assert f["n_sinks"] == 3  # accounts 1, 2, 3 only receive


def test_cycle_vs_no_return_triangle():
    cycle = subgraph_features(_row(3, [(0, 1, 50.0, 1, 0), (1, 2, 50.0, 1, 1), (2, 0, 50.0, 1, 2)]))
    no_return = subgraph_features(_row(3, [(0, 1, 50.0, 1, 0), (1, 2, 50.0, 1, 1), (0, 2, 50.0, 1, 2)]))

    assert cycle["n_3cycles"] == 1
    assert no_return["n_3cycles"] == 0
    assert cycle["n_2cycles"] == no_return["n_2cycles"] == 0


def test_reciprocated_pair():
    f = subgraph_features(_row(3, [(0, 1, 10.0, 1, 0), (1, 0, 10.0, 1, 1), (1, 2, 10.0, 1, 2)]))

    assert f["n_2cycles"] == 1
    assert f["reciprocity"] == pytest.approx(2 / 3)  # 2 of the 3 merged edges have a reverse edge
    assert f["n_3cycles"] == 0


def test_amount_count_and_time_stats():
    # A chain 0 -> 1 -> 2 plus a self-transfer on 2 ("Reinvestment"): no gate, but it is activity.
    f = subgraph_features(_row(3, [(0, 1, 100.0, 2, 0), (1, 2, 1000.0, 1, 3), (2, 2, 50.0, 1, 5)]))

    assert f["n_edges"] == 2
    assert f["n_tx"] == 4
    assert f["n_self_tx"] == 1
    assert f["out_deg_std"] == pytest.approx((2 / 9) ** 0.5)  # out-degrees 1, 1, 0
    assert f["usd_log_total"] == pytest.approx(7.003974)  # log(1 + 1100), self-transfer left out
    assert f["usd_log_max"] == pytest.approx(6.908755)  # log(1 + 1000)
    assert f["usd_log_min"] == pytest.approx(4.615121)  # log(1 + 100)
    assert f["tx_per_edge_max"] == 2
    assert f["tx_per_edge_mean"] == pytest.approx(1.5)
    assert f["time_span_h"] == pytest.approx(5.0)  # first to last first-timestamp, self-transfer included


def test_empty_subgraph_gives_zeros():
    f = subgraph_features(_row(2, []))

    assert f["n_edges"] == 0 and f["time_span_h"] == 0.0 and f["usd_log_total"] == 0.0


def test_feature_matrix_rows_follow_the_cache_frame():
    import polars as pl

    from qlmaml.features import subgraph_feature_matrix

    df = pl.DataFrame([
        _row(3, [(0, 1, 50.0, 1, 0), (1, 2, 50.0, 1, 1), (0, 2, 50.0, 1, 2)]),  # no-return triangle
        _row(3, [(0, 1, 50.0, 1, 0), (1, 2, 50.0, 1, 1), (2, 0, 50.0, 1, 2)]),  # cycle
    ])

    X, names = subgraph_feature_matrix(df, exclude=("time_span_h",))

    assert X.shape == (2, len(names))
    assert "time_span_h" not in names
    assert X[:, names.index("n_3cycles")].tolist() == [0, 1]
