import numpy as np

from qlmaml.data import describe, load_patterns, load_transactions, temporal_split
from qlmaml.fake_data import write_fake_transactions

PATTERNS_SAMPLE = """\
BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1
2022/09/01 04:33,021174,800737690,020,80020C5B0,8630.40,Euro,8630.40,Euro,ACH,1
END LAUNDERING ATTEMPT

BEGIN LAUNDERING ATTEMPT - GATHER-SCATTER
2022/09/02 15:40,021174,800737690,00410,8002CC310,3511.82,Euro,3511.82,Euro,ACH,1
END LAUNDERING ATTEMPT
"""


def test_load_patterns(tmp_path):
    f = tmp_path / "X_Patterns.txt"
    f.write_text(PATTERNS_SAMPLE, encoding="utf-8")
    p = load_patterns(f)
    assert p.height == 3
    assert p["attempt_id"].to_list() == [0, 0, 1]
    assert p["pattern"].to_list() == ["FAN-OUT", "FAN-OUT", "GATHER-SCATTER"]
    assert p["timestamp"].null_count() == 0
    assert p["from_bank"][0] == "021174"  # leading zeros preserved


def test_load_and_split(tmp_path):
    csv = write_fake_transactions(tmp_path / "FAKE_Trans.csv", n_accounts=100, n_normal=500, n_motifs=6, n_days=8)
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
