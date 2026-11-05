"""Loading and splitting the IBM AML transaction data.

Conventions (match IBM's Multi-GNN preprocessing so our numbers are comparable):
- A node is an account identified by (bank, account) pair.
- Edges are transactions, sorted by timestamp, with a stable `edge_id`.
- The split is temporal and day-based: contiguous blocks of days that best
  approximate 60/20/20 of the transactions.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

RAW_COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account_dup",  # the raw file has two columns named "Account"
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]

CLEAN_COLUMNS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_transactions(path: str | Path, n_rows: int | None = None) -> pl.DataFrame:
    """Load a `*_Trans.csv` file into a clean, time-sorted polars DataFrame.

    Adds integer node ids (`from_id`, `to_id`) over (bank, account) pairs and an
    `edge_id` equal to the row position after sorting by time.
    """
    path = Path(path)
    df = pl.read_csv(
        path,
        has_header=True,
        new_columns=CLEAN_COLUMNS,
        n_rows=n_rows,
        schema_overrides={
            "from_bank": pl.Utf8,
            "to_bank": pl.Utf8,
            "from_account": pl.Utf8,
            "to_account": pl.Utf8,
            "amount_received": pl.Float64,
            "amount_paid": pl.Float64,
            "is_laundering": pl.Int8,
        },
    )
    df = df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, "%Y/%m/%d %H:%M"),
        (pl.col("from_bank") + ":" + pl.col("from_account")).alias("from_key"),
        (pl.col("to_bank") + ":" + pl.col("to_account")).alias("to_key"),
    )

    # Stable integer node ids shared by senders and receivers.
    keys = pl.concat([df["from_key"], df["to_key"]]).unique().sort()
    key_to_id = pl.DataFrame({"key": keys, "node_id": np.arange(len(keys), dtype=np.int64)})
    df = (
        df.join(key_to_id.rename({"key": "from_key", "node_id": "from_id"}), on="from_key", how="left")
        .join(key_to_id.rename({"key": "to_key", "node_id": "to_id"}), on="to_key", how="left")
        .drop("from_key", "to_key")
        .sort("timestamp", maintain_order=True)
        .with_row_index("edge_id")
    )
    return df


@dataclass(frozen=True)
class TemporalSplit:
    """Day indices for a contiguous train / val / test split."""

    train_days: np.ndarray
    val_days: np.ndarray
    test_days: np.ndarray
    day_of_edge: np.ndarray  # day index (0-based) for every edge

    def masks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = self.day_of_edge
        return (
            np.isin(d, self.train_days),
            np.isin(d, self.val_days),
            np.isin(d, self.test_days),
        )


def temporal_split(
    df: pl.DataFrame, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
) -> TemporalSplit:
    """Day-based temporal split that best approximates `fractions` of the edges.

    Mirrors Multi-GNN's `data_loading.py`: tries every pair of day cut points and
    keeps the one minimising the largest relative deviation from the target.
    """
    day_of_edge = (
        (df["timestamp"].dt.date() - df["timestamp"].dt.date().min()).dt.total_days().to_numpy()
    ).astype(np.int64)
    n_days = int(day_of_edge.max()) + 1
    daily = np.bincount(day_of_edge, minlength=n_days).astype(float)
    total = daily.sum()

    best, best_score = None, np.inf
    for i, j in itertools.combinations(range(1, n_days), 2):
        parts = [daily[:i].sum(), daily[i:j].sum(), daily[j:].sum()]
        score = max(abs(p / total - f) / f for p, f in zip(parts, fractions))
        if score < best_score:
            best, best_score = (i, j), score
    if best is None:  # fewer than 3 days: fall back to a single block each
        best = (max(1, n_days - 2), max(2, n_days - 1))
    i, j = best
    return TemporalSplit(
        train_days=np.arange(0, i),
        val_days=np.arange(i, j),
        test_days=np.arange(j, n_days),
        day_of_edge=day_of_edge,
    )


_PATTERN_HEADER = re.compile(r"^BEGIN LAUNDERING ATTEMPT\s*-\s*(?P<kind>[A-Z\- ]+?)\s*(?::.*)?$")


def load_patterns(path: str | Path) -> pl.DataFrame:
    """Parse a `*_Patterns.txt` file into one row per (attempt, transaction).

    The file lists laundering attempts as blocks:

        BEGIN LAUNDERING ATTEMPT - FAN-OUT
        <transaction rows in the same CSV format as *_Trans.csv>
        END LAUNDERING ATTEMPT

    Returns columns: attempt_id, pattern, plus the clean transaction columns.
    The parser is tolerant: anything between BEGIN/END that is not a CSV row
    with the expected number of fields is skipped. Verify against the real
    file the first time it is used.
    """
    rows: list[dict] = []
    attempt_id, kind = -1, None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            m = _PATTERN_HEADER.match(line)
            if m:
                attempt_id += 1
                kind = m.group("kind").strip()
                continue
            if line.startswith("END LAUNDERING ATTEMPT"):
                kind = None
                continue
            if kind is None:
                continue
            fields = line.split(",")
            if len(fields) != len(CLEAN_COLUMNS):
                continue
            rows.append({"attempt_id": attempt_id, "pattern": kind, **dict(zip(CLEAN_COLUMNS, fields))})

    if not rows:
        return pl.DataFrame(schema={"attempt_id": pl.Int64, "pattern": pl.Utf8, **{c: pl.Utf8 for c in CLEAN_COLUMNS}})
    df = pl.DataFrame(rows)
    return df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, "%Y/%m/%d %H:%M", strict=False),
        pl.col("amount_received").cast(pl.Float64, strict=False),
        pl.col("amount_paid").cast(pl.Float64, strict=False),
        pl.col("is_laundering").cast(pl.Int8, strict=False),
    )


def describe(df: pl.DataFrame) -> dict:
    """Quick summary used in logs and notebooks."""
    n = df.height
    pos = int(df["is_laundering"].sum())
    return {
        "transactions": n,
        "laundering": pos,
        "illicit_ratio": pos / n if n else float("nan"),
        "accounts": int(max(df["from_id"].max(), df["to_id"].max()) + 1),
        "days": int((df["timestamp"].max() - df["timestamp"].min()).days + 1),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
    }
