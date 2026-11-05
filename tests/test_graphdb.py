"""Offline tests for the Neo4j export (no database needed)."""
from __future__ import annotations

import polars as pl
import pytest

from qlmaml.data import load_patterns, load_transactions
from qlmaml.fake_data import write_fake_transactions
from qlmaml.graphdb import (
    Subgraph,
    _subgraph_from_records,
    admin_import_args,
    build_import_tables,
    write_import_csvs,
)


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake")
    csv = write_fake_transactions(d / "FAKE_Trans.csv", n_accounts=300, n_normal=3_000, n_motifs=6, n_days=6)
    # Build a *_Patterns.txt from the planted laundering rows: one attempt per contiguous block.
    lines = csv.read_text().splitlines()
    laund = [ln for ln in lines[1:] if ln.endswith(",1")]
    blocks = [laund[i : i + 4] for i in range(0, len(laund), 4)]
    txt = []
    for b in blocks:
        txt += ["BEGIN LAUNDERING ATTEMPT - FAN-OUT", *b, "END LAUNDERING ATTEMPT", ""]
    pat = d / "FAKE_Patterns.txt"
    pat.write_text("\n".join(txt))
    df = load_transactions(csv)
    return df, load_patterns(pat), len(blocks), len(laund)


def test_tables_are_consistent(fake):
    df, patterns, n_blocks, n_laund = fake
    t = build_import_tables(df, patterns)

    assert t.transfers.height == df.height
    assert t.accounts.height == df.select(pl.max_horizontal("from_id", "to_id").max()).item() + 1
    assert t.accounts["account_id"].is_unique().all()
    assert set(t.transfers["start"]) | set(t.transfers["end"]) <= set(t.accounts["account_id"])
    assert set(t.at_bank["end"]) == set(t.banks["bank_id"])
    assert set(t.transfers["split"]) == {"train", "val", "test"}

    # SENDS_TO is a lossless collapse of TRANSFER per (pair, split).
    assert t.sends_to["n_tx"].sum() == df.height
    assert t.sends_to["n_laundering"].sum() == df["is_laundering"].sum()
    assert t.sends_to.select("start", "end", "split").is_unique().all()

    # Every laundering row was listed in the patterns file, so every one matches an attempt.
    matched = t.transfers.filter(pl.col("attempt_id").is_not_null())
    assert matched.height >= n_laund - 1  # duplicates in the fake data may collapse
    assert (matched["is_laundering"] == 1).all()
    assert t.attempts.height == n_blocks
    assert set(t.involves["role"]) <= {"sender", "receiver", "both"}


def test_without_patterns(fake):
    df, *_ = fake
    t = build_import_tables(df)
    assert t.attempts.is_empty() and t.involves.is_empty()
    assert t.transfers["attempt_id"].null_count() == df.height


def test_csv_headers_and_args(fake, tmp_path):
    df, patterns, *_ = fake
    paths = write_import_csvs(build_import_tables(df, patterns), tmp_path)
    assert set(paths) == {"banks", "accounts", "attempts", "at_bank", "transfers", "sends_to", "involves"}

    header = paths["transfers"].read_text().splitlines()[0].split(",")
    assert header[:3] == [":START_ID(Account)", ":END_ID(Account)", "edge_id:long"]
    assert "ts:datetime" in header and "is_laundering:int" in header
    first = paths["transfers"].read_text().splitlines()[1].split(",")
    ts = first[header.index("ts:datetime")]
    assert len(ts) == 19 and ts[10] == "T"  # 2022-09-01T00:20:00

    args = admin_import_args(paths)
    assert "--relationships=TRANSFER=/import/transfers.csv" in args
    assert "--nodes=Account=/import/accounts.csv" in args


def test_subgraph_qubit_mapping():
    accts = [{"account_id": "1:a", "node_id": 5}, {"account_id": "2:b", "node_id": 9}, {"account_id": "1:c", "node_id": 1}]
    edges = [
        {"src": "1:a", "dst": "2:b", "edge_id": 3, "amount_paid": 1.0, "is_laundering": 0},
        {"src": "1:a", "dst": "2:b", "edge_id": 1, "amount_paid": 2.0, "is_laundering": 1},
        {"src": "1:c", "dst": "1:a", "edge_id": 2, "amount_paid": 4.0, "is_laundering": 0},
    ]
    g: Subgraph = _subgraph_from_records(accts, edges, center=1)
    assert g.n_qubits == 3
    assert g.edges["edge_id"].to_list() == [1, 2, 3]
    assert g.undirected_pairs() == [(0, 1), (0, 2)]
    simple = g.simple_edges()
    row = simple.filter((pl.col("u") == 0) & (pl.col("v") == 1)).row(0, named=True)
    assert row["n_tx"] == 2 and row["total_paid"] == 3.0 and row["any_laundering"] == 1
