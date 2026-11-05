"""Motif-style graph features per transaction (edge).

A lightweight, transparent re-implementation of the idea behind IBM's Graph
Feature Preprocessor (Blanuša et al., arXiv:2402.08593): for every transaction
compute statistics of the sender's and receiver's recent activity (fan-in /
fan-out, amounts, unique counterparties) in a time window, plus a few graph
features that capture laundering motifs (reciprocity, short cycles).

Everything is computed from edges that occur *before or at* the transaction's
own timestamp, so the features are leakage-free with respect to time, and the
same code can be applied to train / val / test slices built from the full edge
stream (as Multi-GNN does).

All functions take the clean polars DataFrame produced by `qlmaml.data.load_transactions`.
"""
from __future__ import annotations

import numpy as np
import polars as pl

RAW_FEATURES = [
    "amount_paid_log",
    "amount_received_log",
    "same_currency",
    "same_bank",
    "hour",
    "weekday",
]

CATEGORICAL = ["payment_format", "payment_currency", "receiving_currency"]


def _node_window_stats(df: pl.DataFrame, node_col: str, prefix: str, window: str) -> pl.DataFrame:
    """Rolling (trailing) activity stats of `node_col` over `window`, aligned to each edge."""
    base = df.select("edge_id", "timestamp", node_col, "amount_paid").sort(node_col, "timestamp")
    out = base.rolling(
        index_column="timestamp",
        period=window,
        group_by=node_col,
    ).agg(
        pl.len().alias(f"{prefix}_n_{window}"),
        pl.col("amount_paid").sum().alias(f"{prefix}_sum_{window}"),
        pl.col("amount_paid").mean().alias(f"{prefix}_mean_{window}"),
        pl.col("amount_paid").std().fill_null(0.0).alias(f"{prefix}_std_{window}"),
    )
    # `rolling` yields one row per input row, in the same (sorted) order.
    return pl.concat([base.select("edge_id"), out.drop("timestamp", node_col)], how="horizontal")


def _unique_counterparties(df: pl.DataFrame, node_col: str, other_col: str, prefix: str, window: str) -> pl.DataFrame:
    base = df.select("edge_id", "timestamp", node_col, other_col).sort(node_col, "timestamp")
    out = base.rolling(index_column="timestamp", period=window, group_by=node_col).agg(
        pl.col(other_col).n_unique().alias(f"{prefix}_uniq_{window}")
    )
    return pl.concat([base.select("edge_id"), out.drop("timestamp", node_col)], how="horizontal")


def _reciprocity_and_cycles(df: pl.DataFrame, window: str) -> pl.DataFrame:
    """Flags for 2-cycles (B->A exists shortly before A->B) and 3-cycles via a shared middle node."""
    win = pl.duration(**{_window_unit(window): _window_value(window)})
    e = df.select("edge_id", "timestamp", "from_id", "to_id")
    rev = e.rename({"edge_id": "rev_edge", "timestamp": "rev_ts", "from_id": "to_id", "to_id": "from_id"})
    # A->B at t; was there a B->A in [t - window, t]?
    two = (
        e.join(rev, on=["from_id", "to_id"], how="left")
        .filter((pl.col("rev_ts") <= pl.col("timestamp")) & (pl.col("rev_ts") >= pl.col("timestamp") - win))
        .group_by("edge_id")
        .agg(pl.len().alias("recip_2cycle"))
    )
    # 3-cycle: A->B at t, and some B->C and C->A in the window before t.
    bc = e.rename({"edge_id": "e_bc", "timestamp": "ts_bc", "from_id": "to_id", "to_id": "c_id"})
    ca = e.rename({"edge_id": "e_ca", "timestamp": "ts_ca", "from_id": "c_id", "to_id": "from_id"})
    three = (
        e.join(bc, on="to_id", how="inner")
        .filter((pl.col("ts_bc") <= pl.col("timestamp")) & (pl.col("ts_bc") >= pl.col("timestamp") - win))
        .join(ca, on=["c_id", "from_id"], how="inner")
        .filter((pl.col("ts_ca") <= pl.col("timestamp")) & (pl.col("ts_ca") >= pl.col("timestamp") - win))
        .group_by("edge_id")
        .agg(pl.len().alias("cycle3"))
    )
    return (
        e.select("edge_id")
        .join(two, on="edge_id", how="left")
        .join(three, on="edge_id", how="left")
        .fill_null(0)
    )


def _window_unit(window: str) -> str:
    return {"h": "hours", "d": "days", "m": "minutes"}[window[-1]]


def _window_value(window: str) -> int:
    return int(window[:-1])


def build_features(
    df: pl.DataFrame,
    windows: tuple[str, ...] = ("1d", "7d"),
    cycle_window: str = "7d",
    with_cycles: bool = True,
) -> pl.DataFrame:
    """Return a feature frame with one row per edge, aligned on `edge_id`.

    Set `with_cycles=False` on very large frames; the 3-cycle join is the only
    step whose cost is not linear in the number of edges.
    """
    feats = df.select(
        "edge_id",
        pl.col("amount_paid").log1p().alias("amount_paid_log"),
        pl.col("amount_received").log1p().alias("amount_received_log"),
        (pl.col("payment_currency") == pl.col("receiving_currency")).cast(pl.Int8).alias("same_currency"),
        (pl.col("from_bank") == pl.col("to_bank")).cast(pl.Int8).alias("same_bank"),
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("weekday"),
        *[pl.col(c).cast(pl.Categorical).to_physical().alias(f"{c}_code") for c in CATEGORICAL],
    )

    for w in windows:
        feats = (
            feats.join(_node_window_stats(df, "from_id", "out", w), on="edge_id", how="left")
            .join(_node_window_stats(df, "to_id", "in", w), on="edge_id", how="left")
            .join(_unique_counterparties(df, "from_id", "to_id", "out", w), on="edge_id", how="left")
            .join(_unique_counterparties(df, "to_id", "from_id", "in", w), on="edge_id", how="left")
        )
        # Fan-out / fan-in ratios: how spread the recent activity is.
        feats = feats.with_columns(
            (pl.col(f"out_uniq_{w}") / pl.col(f"out_n_{w}")).alias(f"out_spread_{w}"),
            (pl.col(f"in_uniq_{w}") / pl.col(f"in_n_{w}")).alias(f"in_spread_{w}"),
        )

    if with_cycles:
        feats = feats.join(_reciprocity_and_cycles(df, cycle_window), on="edge_id", how="left")

    return feats.sort("edge_id").fill_null(0)


def feature_matrix(feats: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in feats.columns if c != "edge_id"]
    return feats.select(cols).to_numpy().astype(np.float32), cols
