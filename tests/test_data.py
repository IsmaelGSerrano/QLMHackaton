import numpy as np

from qlmaml.data import describe, load_transactions, temporal_split
from qlmaml.fake_data import write_fake_transactions


def test_load_and_split(tmp_path):
    csv = write_fake_transactions(tmp_path / "FAKE_Trans.csv", n_accounts=300, n_normal=3_000, n_motifs=10, n_days=8)
    df = load_transactions(csv)

    assert df["edge_id"].to_list() == list(range(df.height))
    assert df["timestamp"].is_sorted()
    assert df["from_id"].max() < describe(df)["accounts"]
    assert 0 < df["is_laundering"].sum() < df.height

    split = temporal_split(df)
    tr, va, te = split.masks()
    assert tr.sum() + va.sum() + te.sum() == df.height
    assert not (tr & va).any() and not (va & te).any()
    # Train strictly precedes val precedes test in time.
    ts = df["timestamp"].to_numpy()
    assert ts[tr].max() < ts[va].min() < ts[va].max() < ts[te].min()
    assert abs(tr.mean() - 0.6) < 0.2
