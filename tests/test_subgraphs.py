"""Subgraph dataset (WP1) on synthetic data, through the in-memory FrameGraph source."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from qlmaml.fake_data import HEADER, write_fake_transactions
from qlmaml.subgraphs import FrameGraph, build_dataset, positive_sample, read_cache, write_cache

T0 = datetime(2022, 9, 1, 8, 0)


def _line(minute: int, a: str, b: str, amount: float, label: int, currency: str = "US Dollar") -> str:
    ts = T0 + timedelta(minutes=minute)
    return f"{ts:%Y/%m/%d %H:%M},001,{a},001,{b},{amount:.2f},{currency},{amount:.2f},{currency},ACH,{label}"


def _write(tmp_path: Path, normal: list[str], attempts: list[tuple[str, list[str]]]) -> FrameGraph:
    trans = tmp_path / "T_Trans.csv"
    laund = [ln for _, rows in attempts for ln in rows]
    trans.write_text("\n".join([HEADER, *normal, *laund]) + "\n", encoding="utf-8")
    pat = tmp_path / "T_Patterns.txt"
    txt = []
    for kind, rows in attempts:
        txt += [f"BEGIN LAUNDERING ATTEMPT - {kind}", *rows, "END LAUNDERING ATTEMPT", ""]
    pat.write_text("\n".join(txt), encoding="utf-8")
    return FrameGraph.from_files(trans, pat)


def test_positive_is_cut_in_time_at_max_nodes(tmp_path):
    # Money reaches A, B, C, D, then E. A background (normal) B->C transfer falls inside
    # the window; a later A->B transfer falls after the cut.
    attempt = [
        _line(0, "A", "B", 100, 1),
        _line(10, "A", "C", 100, 1),
        _line(20, "C", "D", 90, 1),
        _line(30, "D", "E", 80, 1),
    ]
    normal = [_line(15, "B", "C", 5, 0), _line(40, "A", "B", 7, 0), _line(15, "X", "Y", 1, 0)]
    src = _write(tmp_path, normal, [("STACK", attempt)])

    s = positive_sample(src, attempt_id=0, max_nodes=4)

    assert s.label == 1 and s.pattern == "STACK" and s.cut
    assert s.graph.nodes["account_id"].to_list() == ["001:A", "001:B", "001:C", "001:D"]  # reach order
    assert s.start_ts == T0
    edges = s.graph.edges
    assert edges["ts"].max() == T0 + timedelta(minutes=20)  # cut before D->E
    assert edges.height == 4  # A->B, A->C, C->D and the background B->C
    assert edges["is_laundering"].sum() == 3


def test_positive_without_cut_keeps_whole_attempt(tmp_path):
    attempt = [_line(0, "A", "B", 100, 1), _line(10, "B", "A", 90, 1)]
    src = _write(tmp_path, [_line(5, "X", "Y", 1, 0)], [("CYCLE", attempt)])

    s = positive_sample(src, attempt_id=0, max_nodes=4)

    assert not s.cut
    assert s.graph.n_qubits == 2
    assert s.graph.edges.height == 2


def test_self_transfers_are_not_gates(tmp_path):
    attempt = [_line(0, "A", "B", 100, 1), _line(10, "B", "C", 90, 1)]
    src = _write(tmp_path, [_line(5, "B", "B", 50, 0)], [("STACK", attempt)])

    s = positive_sample(src, attempt_id=0, max_nodes=4)

    assert s.graph.edges.height == 3  # the B->B reinvestment is in the subgraph...
    assert s.n_edges == 2  # ...but only A->B and B->C become two-qubit gates

    row = read_cache(write_cache([s], tmp_path / "c.parquet", usd_rates={"US Dollar": 1.0})).row(0, named=True)
    assert len(row["edges"]) == 3 and row["n_edges"] == 2


# --------------------------------------------------------------------------- #
# End to end on FAKE data                                                     #
# --------------------------------------------------------------------------- #

MAX_NODES, RATIO, EDGE_TOL = 6, 5, 1


@pytest.fixture(scope="module")
def fake_dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake")
    trans = write_fake_transactions(d / "FAKE_Trans.csv", n_accounts=300, n_normal=6_000, n_motifs=30, n_days=10,
                                    patterns_path=d / "FAKE_Patterns.txt")
    src = FrameGraph.from_files(trans, d / "FAKE_Patterns.txt")
    return src, build_dataset(src, max_nodes=MAX_NODES, ratio=RATIO, edge_tol=EDGE_TOL, seed=0)


def test_separate_pair_attempts_get_pairs_negatives(fake_dataset):
    # A cut BIPARTITE attempt (3 separate pairs: 6 accounts, 3 gates) cannot be
    # matched by a connected ego-net, which has at least 5 gates on 6 accounts.
    _, ds = fake_dataset
    sparse = [p for p in ds.samples if p.label == 1 and p.n_edges + EDGE_TOL < p.graph.n_qubits - 1]
    assert sparse, "fake data should contain separate-pair attempts"
    for p in sparse:
        negs = [n for n in ds.samples if n.matched_to == p.source_id]
        assert len(negs) == RATIO
        assert {n.kind for n in negs} == {"pairs"}


def test_nothing_is_dropped_on_fake_data(fake_dataset):
    _, ds = fake_dataset
    assert ds.dropped == []


def test_unmatchable_positive_is_dropped_with_its_negatives(tmp_path):
    # A full directed triangle (3 accounts, 6 gates): no normal subgraph that small is that dense.
    tri = [_line(k, a, b, 100, 1) for k, (a, b) in enumerate(
        [("A", "B"), ("B", "C"), ("C", "A"), ("B", "A"), ("C", "B"), ("A", "C")])]
    normal = [_line(10 + k, f"N{k}", f"M{k}", 5, 0) for k in range(20)]
    src = _write(tmp_path, normal, [("CYCLE", tri)])

    ds = build_dataset(src, max_nodes=4, ratio=2, max_tries=50, seed=0)

    assert [p.source_id for p in ds.dropped] == [0]
    assert ds.samples == []


def test_every_attempt_gives_one_positive_within_the_qubit_cap(fake_dataset):
    src, ds = fake_dataset
    pos = [s for s in ds.samples if s.label == 1]
    assert sorted(s.source_id for s in pos) == sorted(src.attempts()["attempt_id"])
    assert all(2 <= s.graph.n_qubits <= MAX_NODES for s in ds.samples)
    assert any(s.cut for s in pos)  # fake motifs have up to 9 accounts


def test_negatives_are_matched_on_nodes_and_edges(fake_dataset):
    _, ds = fake_dataset
    pos = {s.source_id: s for s in ds.samples if s.label == 1}
    neg = [s for s in ds.samples if s.label == 0]
    assert len(neg) == RATIO * len(pos)
    for n in neg:
        p = pos[n.matched_to]
        assert n.split == p.split
        assert n.graph.n_qubits == p.graph.n_qubits
        assert abs(n.n_edges - p.n_edges) <= EDGE_TOL
    # Hence the node-count histograms agree exactly, class by class.
    assert sorted(RATIO * [p.graph.n_qubits for p in pos.values()]) == sorted(n.graph.n_qubits for n in neg)


def test_negatives_have_no_laundering_and_no_future_edges(fake_dataset):
    src, ds = fake_dataset
    pos = {s.source_id: s for s in ds.samples if s.label == 1}
    for n in (s for s in ds.samples if s.label == 0):
        e = n.graph.edges
        assert e["is_laundering"].sum() == 0
        if n.kind == "ego":  # nothing after the center transaction, nothing older than the window
            center_ts = src.transfers.filter(pl.col("edge_id") == n.source_id)["ts"].item()
            assert n.graph.center_edge_id == n.source_id
            assert e["ts"].max() <= center_ts
            assert e["ts"].min() >= center_ts - timedelta(days=1)
        else:  # pairs: no longer than its positive (at least 1 h)
            p = pos[n.matched_to]
            span = max(p.graph.edges["ts"].max() - p.start_ts, timedelta(hours=1))
            assert e["ts"].max() - e["ts"].min() <= span


def test_temporal_split_is_monotone(fake_dataset):
    _, ds = fake_dataset
    by_split = {k: [s.start_ts for s in ds.samples if s.split == k] for k in ("train", "val", "test")}
    assert all(by_split.values())
    assert max(by_split["train"]) < min(by_split["val"])
    assert max(by_split["val"]) < min(by_split["test"])
    n_pos = {k: sum(1 for s in ds.samples if s.split == k and s.label == 1) for k in by_split}
    assert n_pos["train"] > n_pos["val"] > 0 and n_pos["test"] > 0


def test_cache_round_trip(fake_dataset, tmp_path):
    src, ds = fake_dataset
    path = write_cache(ds.samples, tmp_path / "subgraphs_FAKE_n6.parquet", usd_rates=src.usd_rates())
    df = read_cache(path)

    assert df.height == len(ds.samples)
    assert {"split", "label", "pattern", "kind", "n_qubits", "n_edges", "nodes", "edges"} <= set(df.columns)
    assert df["label"].sum() == sum(s.label for s in ds.samples)
    row = df.filter(pl.col("label") == 1).row(0, named=True)
    s = next(s for s in ds.samples if s.label == 1 and s.source_id == row["source_id"])
    assert row["nodes"] == s.graph.nodes["account_id"].to_list()
    assert row["n_qubits"] == len(row["nodes"]) == s.graph.n_qubits
    assert sum(e["u"] != e["v"] for e in row["edges"]) == row["n_edges"] == s.n_edges
    e0 = row["edges"][0]
    assert set(e0) == {"u", "v", "first_ts", "total_usd", "n_tx"}
    firsts = [e["first_ts"] for e in row["edges"]]
    assert firsts == sorted(firsts)  # gate order = time order
    assert sum(e["n_tx"] for e in row["edges"]) == s.graph.edges.height


def test_cache_holds_more_than_100_positives_before_the_negatives(tmp_path):
    # Positives come first and have matched_to = None; HI-Small has 370 of them.
    attempt = [_line(0, "A", "B", 100, 1), _line(10, "B", "C", 90, 1)]
    src = _write(tmp_path, [_line(5, "X", "Y", 1, 0)], [("STACK", attempt)])
    p = positive_sample(src, attempt_id=0, max_nodes=4)
    n = replace(p, label=0, pattern=None, matched_to=0)

    df = read_cache(write_cache([p] * 150 + [n], tmp_path / "c.parquet", usd_rates={"US Dollar": 1.0}))

    assert df.height == 151
    assert df["matched_to"].to_list() == [None] * 150 + [0]
