"""Neo4j representation of the IBM AML transaction graph.

Graph model (property graph, Neo4j 5):

    (:Bank {bank_id})
    (:Account {account_id: "<bank>:<account>", node_id, bank, account})
        -[:AT_BANK]->(:Bank)

    (:Account)-[:TRANSFER {edge_id, ts, day, split, amount_paid, amount_received,
                           payment_currency, receiving_currency, payment_format,
                           is_laundering, attempt_id?, pattern?}]->(:Account)
        One relationship per transaction row: this keeps the *multigraph*
        (parallel edges with different timestamps / amounts / currencies).

    (:Account)-[:SENDS_TO {split, n_tx, total_paid, first_ts, last_ts,
                           n_laundering}]->(:Account)
        The same edges collapsed to a *simple* weighted digraph, one relationship
        per (sender, receiver, split). This is the representation used when one
        qubit = one account and one CZ = one account pair. Aggregates are per
        temporal split, so a train-time query never sees val/test edges.

    (:LaunderingAttempt {attempt_id (string, as all neo4j-admin :ID values), pattern, n_tx, start_ts, end_ts})
        -[:INVOLVES {role}]->(:Account)          # role: "sender" | "receiver" | "both"
        Ground-truth motif instances from `*_Patterns.txt`. The matching TRANSFER
        relationships also carry `attempt_id` / `pattern`.

Identifiers match `qlmaml.data.load_transactions`: `node_id` is the integer
account id and `edge_id` the time-sorted row index, so anything computed in
Neo4j joins back to the polars pipeline (features, labels, splits).

Loading uses `neo4j-admin database import full` (offline bulk import), which is
the only practical path for millions of edges. `write_import_csvs` produces the
files with typed headers; `scripts/neo4j_load.py` runs the import in Docker.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from qlmaml.data import TemporalSplit, temporal_split

SPLIT_NAMES = np.array(["train", "val", "test"])
_TS_FMT = "%Y-%m-%dT%H:%M:%S"

# Pattern rows are matched to transaction rows on these columns.
_MATCH_COLS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_paid",
    "payment_format",
]


# --------------------------------------------------------------------------- #
# Building the import tables                                                  #
# --------------------------------------------------------------------------- #


def _split_labels(split: TemporalSplit) -> np.ndarray:
    lab = np.full(split.day_of_edge.shape, 2, dtype=np.int8)
    lab[np.isin(split.day_of_edge, split.train_days)] = 0
    lab[np.isin(split.day_of_edge, split.val_days)] = 1
    return SPLIT_NAMES[lab]


def attach_patterns(df: pl.DataFrame, patterns: pl.DataFrame) -> pl.DataFrame:
    """Add `attempt_id` and `pattern` columns to transactions listed in `*_Patterns.txt`.

    A transaction that appears in several attempts keeps the first one. Rows that
    do not match any pattern row get nulls.
    """
    if patterns.is_empty():
        return df.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("attempt_id"),
            pl.lit(None, dtype=pl.Utf8).alias("pattern"),
        )
    p = (
        patterns.select("attempt_id", "pattern", *_MATCH_COLS)
        .with_columns(pl.col("attempt_id").cast(pl.Int64))
        .sort("attempt_id")
        .unique(subset=_MATCH_COLS, keep="first", maintain_order=True)
    )
    return df.join(p, on=_MATCH_COLS, how="left")


@dataclass
class ImportTables:
    """DataFrames ready for `neo4j-admin import`, one per node/relationship file."""

    banks: pl.DataFrame
    accounts: pl.DataFrame
    at_bank: pl.DataFrame
    transfers: pl.DataFrame
    sends_to: pl.DataFrame
    attempts: pl.DataFrame
    involves: pl.DataFrame

    def summary(self) -> dict[str, int]:
        return {k: getattr(self, k).height for k in self.__dataclass_fields__}


def build_import_tables(
    df: pl.DataFrame,
    patterns: pl.DataFrame | None = None,
    split: TemporalSplit | None = None,
) -> ImportTables:
    """Turn the clean transaction DataFrame into node and relationship tables."""
    split = split or temporal_split(df)
    df = df.with_columns(
        (pl.col("from_bank") + ":" + pl.col("from_account")).alias("src"),
        (pl.col("to_bank") + ":" + pl.col("to_account")).alias("dst"),
        pl.Series("day", split.day_of_edge, dtype=pl.Int32),
        pl.Series("split", _split_labels(split)),
    )
    df = attach_patterns(df, patterns if patterns is not None else pl.DataFrame())

    accounts = (
        pl.concat(
            [
                df.select(pl.col("src").alias("account_id"), pl.col("from_id").alias("node_id"),
                          pl.col("from_bank").alias("bank"), pl.col("from_account").alias("account")),
                df.select(pl.col("dst").alias("account_id"), pl.col("to_id").alias("node_id"),
                          pl.col("to_bank").alias("bank"), pl.col("to_account").alias("account")),
            ]
        )
        .unique(subset="account_id")
        .sort("node_id")
    )
    banks = accounts.select(pl.col("bank").alias("bank_id")).unique().sort("bank_id")
    at_bank = accounts.select(pl.col("account_id").alias("start"), pl.col("bank").alias("end"))

    transfers = df.select(
        pl.col("src").alias("start"),
        pl.col("dst").alias("end"),
        "edge_id",
        pl.col("timestamp").alias("ts"),
        "day",
        "split",
        "amount_paid",
        "amount_received",
        "payment_currency",
        "receiving_currency",
        "payment_format",
        "is_laundering",
        "attempt_id",
        "pattern",
    )

    sends_to = (
        df.group_by("src", "dst", "split")
        .agg(
            pl.len().alias("n_tx"),
            pl.col("amount_paid").sum().alias("total_paid"),
            pl.col("timestamp").min().alias("first_ts"),
            pl.col("timestamp").max().alias("last_ts"),
            pl.col("is_laundering").cast(pl.Int64).sum().alias("n_laundering"),
        )
        .rename({"src": "start", "dst": "end"})
        .sort("start", "end", "split")
    )

    lab = df.filter(pl.col("attempt_id").is_not_null())
    attempts = (
        lab.group_by("attempt_id")
        .agg(
            pl.col("pattern").first(),
            pl.len().alias("n_tx"),
            pl.col("timestamp").min().alias("start_ts"),
            pl.col("timestamp").max().alias("end_ts"),
        )
        .sort("attempt_id")
    )
    roles = pl.concat(
        [
            lab.select("attempt_id", pl.col("src").alias("end"), pl.lit("sender").alias("role")),
            lab.select("attempt_id", pl.col("dst").alias("end"), pl.lit("receiver").alias("role")),
        ]
    )
    involves = (
        roles.group_by("attempt_id", "end")
        .agg(pl.col("role").unique())
        .with_columns(
            pl.when(pl.col("role").list.len() > 1).then(pl.lit("both")).otherwise(pl.col("role").list.first()).alias("role")
        )
        .rename({"attempt_id": "start"})
        .sort("start", "end")
    )
    return ImportTables(banks, accounts, at_bank, transfers, sends_to, attempts, involves)


# --------------------------------------------------------------------------- #
# Writing neo4j-admin import CSVs                                             #
# --------------------------------------------------------------------------- #

# Typed headers for neo4j-admin. Keys are column names in ImportTables, values the header token.
_HEADERS: dict[str, dict[str, str]] = {
    "banks": {"bank_id": "bank_id:ID(Bank)"},
    "accounts": {"account_id": "account_id:ID(Account)", "node_id": "node_id:long", "bank": "bank", "account": "account"},
    "at_bank": {"start": ":START_ID(Account)", "end": ":END_ID(Bank)"},
    "transfers": {
        "start": ":START_ID(Account)", "end": ":END_ID(Account)", "edge_id": "edge_id:long", "ts": "ts:datetime",
        "day": "day:int", "split": "split", "amount_paid": "amount_paid:double",
        "amount_received": "amount_received:double", "payment_currency": "payment_currency",
        "receiving_currency": "receiving_currency", "payment_format": "payment_format",
        "is_laundering": "is_laundering:int", "attempt_id": "attempt_id:long", "pattern": "pattern",
    },
    "sends_to": {
        "start": ":START_ID(Account)", "end": ":END_ID(Account)", "split": "split", "n_tx": "n_tx:long",
        "total_paid": "total_paid:double", "first_ts": "first_ts:datetime", "last_ts": "last_ts:datetime",
        "n_laundering": "n_laundering:long",
    },
    "attempts": {
        "attempt_id": "attempt_id:ID(Attempt)", "pattern": "pattern", "n_tx": "n_tx:long",
        "start_ts": "start_ts:datetime", "end_ts": "end_ts:datetime",
    },
    "involves": {"start": ":START_ID(Attempt)", "end": ":END_ID(Account)", "role": "role"},
}

# file name -> (kind, label or relationship type)
IMPORT_FILES: dict[str, tuple[str, str]] = {
    "banks": ("nodes", "Bank"),
    "accounts": ("nodes", "Account"),
    "attempts": ("nodes", "LaunderingAttempt"),
    "at_bank": ("relationships", "AT_BANK"),
    "transfers": ("relationships", "TRANSFER"),
    "sends_to": ("relationships", "SENDS_TO"),
    "involves": ("relationships", "INVOLVES"),
}


def write_import_csvs(tables: ImportTables, out_dir: str | Path) -> dict[str, Path]:
    """Write one CSV per table with neo4j-admin typed headers. Returns {name: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, header in _HEADERS.items():
        t = getattr(tables, name)
        if name in ("attempts", "involves") and t.is_empty():
            continue  # no patterns file given
        path = out_dir / f"{name}.csv"
        t.select(list(header)).rename(header).write_csv(path, datetime_format=_TS_FMT, null_value="")
        paths[name] = path
    return paths


def admin_import_args(paths: dict[str, Path], import_root: str = "/import") -> list[str]:
    """`neo4j-admin database import full` arguments for the files in `paths`."""
    args = []
    for name, path in paths.items():
        kind, label = IMPORT_FILES[name]
        args.append(f"--{kind}={label}={import_root}/{Path(path).name}")
    return args


# Run after the bulk import (neo4j-admin import creates no indexes).
SCHEMA_CYPHER = [
    "CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE",
    "CREATE CONSTRAINT bank_id IF NOT EXISTS FOR (b:Bank) REQUIRE b.bank_id IS UNIQUE",
    "CREATE CONSTRAINT attempt_id IF NOT EXISTS FOR (p:LaunderingAttempt) REQUIRE p.attempt_id IS UNIQUE",
    "CREATE INDEX account_node_id IF NOT EXISTS FOR (a:Account) ON (a.node_id)",
    "CREATE INDEX transfer_edge_id IF NOT EXISTS FOR ()-[t:TRANSFER]-() ON (t.edge_id)",
    "CREATE INDEX transfer_ts IF NOT EXISTS FOR ()-[t:TRANSFER]-() ON (t.ts)",
    "CREATE INDEX transfer_label IF NOT EXISTS FOR ()-[t:TRANSFER]-() ON (t.is_laundering)",
    "CREATE INDEX transfer_attempt IF NOT EXISTS FOR ()-[t:TRANSFER]-() ON (t.attempt_id)",
    "CREATE INDEX sends_to_split IF NOT EXISTS FOR ()-[s:SENDS_TO]-() ON (s.split)",
]


# --------------------------------------------------------------------------- #
# Querying: small, leakage-free subgraphs for quantum encodings               #
# --------------------------------------------------------------------------- #


@dataclass
class Subgraph:
    """A small directed multigraph with nodes relabelled 0..n-1 (qubit indices)."""

    nodes: pl.DataFrame  # qubit, account_id, node_id
    edges: pl.DataFrame  # u, v (qubit indices) + TRANSFER properties
    center_edge_id: int | None = None

    @property
    def n_qubits(self) -> int:
        return self.nodes.height

    def simple_edges(self, usd_rates: Mapping[str, float] | None = None) -> pl.DataFrame:
        """Collapse parallel edges: one row per (u, v) with count, total amount and label.

        With timestamps (`ts`) the rows also get `first_ts` and are sorted in time
        order, ties broken by amount (ADR 0001: the order must not use account
        names). With `usd_rates` ({currency: USD per unit}) they also get
        `total_usd`; a currency missing from `usd_rates` raises KeyError.
        """
        e = self.edges
        aggs = [
            pl.len().alias("n_tx"),
            pl.col("amount_paid").sum().alias("total_paid"),
            pl.col("is_laundering").max().alias("any_laundering"),
        ]
        if usd_rates is not None:
            missing = set(e["payment_currency"].unique().drop_nulls()) - set(usd_rates)
            if missing:
                raise KeyError(f"no USD rate for currencies {sorted(missing)}")
            e = e.with_columns(
                (pl.col("amount_paid") * pl.col("payment_currency").replace_strict(dict(usd_rates), return_dtype=pl.Float64))
                .alias("amount_usd")
            )
            aggs.append(pl.col("amount_usd").sum().alias("total_usd"))
        if "ts" in e.columns:
            aggs.append(pl.col("ts").min().alias("first_ts"))
        simple = e.group_by("u", "v").agg(aggs)
        if "ts" not in e.columns:
            return simple.sort("u", "v")
        amount = "total_usd" if usd_rates is not None else "total_paid"
        return simple.sort("first_ts", amount, "u", "v")

    def undirected_pairs(self) -> list[tuple[int, int]]:
        """Distinct unordered qubit pairs, e.g. for one CZ per account pair."""
        pairs = {tuple(sorted((int(u), int(v)))) for u, v in self.edges.select("u", "v").iter_rows() if u != v}
        return sorted(pairs)


def _subgraph_from_records(accounts: list[dict], edges: list[dict], center: int | None) -> Subgraph:
    nodes = pl.DataFrame(accounts, schema={"account_id": pl.Utf8, "node_id": pl.Int64}).with_row_index("qubit")
    idx = dict(zip(nodes["account_id"], nodes["qubit"]))
    e = pl.DataFrame(edges) if edges else pl.DataFrame(schema={
        "src": pl.Utf8, "dst": pl.Utf8, "edge_id": pl.Int64, "ts": pl.Datetime, "amount_paid": pl.Float64,
        "payment_currency": pl.Utf8, "payment_format": pl.Utf8, "is_laundering": pl.Int64})
    e = e.with_columns(
        pl.col("src").replace_strict(idx, return_dtype=pl.UInt32).alias("u"),
        pl.col("dst").replace_strict(idx, return_dtype=pl.UInt32).alias("v"),
    )
    return Subgraph(nodes=nodes, edges=e.sort("edge_id"), center_edge_id=center)


class GraphDB:
    """Thin wrapper over the official neo4j driver.

    Connection settings come from the environment (never hard-code them):
    NEO4J_URI (default bolt://localhost:7687), NEO4J_USER (default neo4j), NEO4J_PASSWORD.
    """

    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(
            uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(user or os.environ.get("NEO4J_USER", "neo4j"), password or os.environ["NEO4J_PASSWORD"]),
            notifications_min_severity="OFF",  # e.g. "label does not exist" when no patterns were loaded
        )
        self._id_ranges: dict[tuple, tuple] = {}

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> GraphDB:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def run(self, query: str, **params) -> list[dict]:
        records, _, _ = self.driver.execute_query(query, params, database_="neo4j")
        return [r.data() for r in records]

    def create_schema(self) -> None:
        for q in SCHEMA_CYPHER:
            self.run(q)
        self.run("CALL db.awaitIndexes(600)")

    def counts(self) -> dict[str, int]:
        out = {}
        for label in ("Account", "Bank", "LaunderingAttempt"):
            out[label] = self.run(f"MATCH (n:{label}) RETURN count(n) AS c")[0]["c"]
        for rel in ("TRANSFER", "SENDS_TO", "AT_BANK", "INVOLVES"):
            out[rel] = self.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c")[0]["c"]
        return out

    # -- subgraph extraction ------------------------------------------------ #

    def ego_subgraph(
        self,
        edge_id: int,
        hops: int = 1,
        max_nodes: int = 12,
        window: timedelta = timedelta(days=1),
    ) -> Subgraph:
        """Accounts within `hops` of the endpoints of transaction `edge_id`, built
        only from transfers in [t - window, t] where t is that transaction's time.

        Leakage-free by construction (no future edges). Neighbours are added
        breadth-first, most recent counterparties first, until `max_nodes` accounts
        are selected; the result therefore fits a `max_nodes`-qubit register.
        """
        center = self.run(
            "MATCH (a)-[t:TRANSFER {edge_id: $eid}]->(b) "
            "RETURN a.account_id AS a, b.account_id AS b, t.ts AS ts",
            eid=edge_id,
        )
        if not center:
            raise KeyError(f"no TRANSFER with edge_id={edge_id}")
        c = center[0]
        t_end = c["ts"].to_native()  # timezone-aware python datetime
        t_start = t_end - window
        selected = list(dict.fromkeys([c["a"], c["b"]]))
        frontier = list(selected)
        for _ in range(hops):
            if len(selected) >= max_nodes or not frontier:
                break
            nbrs = self.run(
                "UNWIND $frontier AS fid "
                "MATCH (f:Account {account_id: fid})-[t:TRANSFER]-(n:Account) "
                "WHERE t.ts >= $t0 AND t.ts <= $t1 AND NOT n.account_id IN $selected "
                "WITH n, max(t.ts) AS last ORDER BY last DESC, n.account_id "
                "RETURN n.account_id AS id LIMIT $k",
                frontier=frontier, selected=selected, t0=t_start, t1=t_end, k=max_nodes - len(selected),
            )
            frontier = [r["id"] for r in nbrs]
            selected.extend(frontier)
        return self.induced_subgraph(selected, t_start, t_end, center=edge_id)

    def attempt_subgraph(self, attempt_id: int) -> Subgraph:
        """All accounts and transfers of one ground-truth laundering attempt (motif instance)."""
        accts = self.run(
            # Via TRANSFER.attempt_id (a long): neo4j-admin stores the :ID attempt_id as a string.
            "MATCH (a:Account)-[t:TRANSFER {attempt_id: $aid}]-() "
            "RETURN DISTINCT a.account_id AS account_id, a.node_id AS node_id ORDER BY node_id",
            aid=attempt_id,
        )
        edges = self.run(
            "MATCH (a)-[t:TRANSFER {attempt_id: $aid}]->(b) " + _EDGE_RETURN,
            aid=attempt_id,
        )
        return _subgraph_from_records(accts, _py_ts(edges), center=None)

    def induced_subgraph(self, account_ids: list[str], t0: datetime, t1: datetime,
                         center: int | None = None) -> Subgraph:
        """Accounts `account_ids` (in this qubit order) and every transfer between them
        with t0 <= ts <= t1. Naive datetimes are read as UTC, the import time zone."""
        t0, t1 = _utc(t0), _utc(t1)
        accts = self.run(
            "UNWIND range(0, size($ids) - 1) AS i "
            "MATCH (a:Account {account_id: $ids[i]}) RETURN a.account_id AS account_id, a.node_id AS node_id ORDER BY i",
            ids=account_ids,
        )
        edges = self.run(
            "MATCH (a:Account)-[t:TRANSFER]->(b:Account) "
            "WHERE a.account_id IN $ids AND b.account_id IN $ids AND t.ts >= $t0 AND t.ts <= $t1 " + _EDGE_RETURN,
            ids=account_ids, t0=t0, t1=t1,
        )
        return _subgraph_from_records(accts, _py_ts(edges), center=center)

    # -- dataset building (qlmaml.subgraphs.GraphSource) --------------------- #

    def attempts(self) -> pl.DataFrame:
        """One row per ground-truth attempt: attempt_id (int), pattern, start_ts, end_ts."""
        rows = self.run(
            "MATCH (p:LaunderingAttempt) RETURN toInteger(p.attempt_id) AS attempt_id, p.pattern AS pattern, "
            "p.start_ts AS start_ts, p.end_ts AS end_ts ORDER BY attempt_id"
        )
        for r in rows:
            r["start_ts"] = _naive(r["start_ts"])
            r["end_ts"] = _naive(r["end_ts"])
        return pl.DataFrame(rows, schema={"attempt_id": pl.Int64, "pattern": pl.Utf8,
                                          "start_ts": pl.Datetime, "end_ts": pl.Datetime})

    def normal_edge_ids(self, t0: datetime, t1: datetime, n: int, seed: int) -> list[int]:
        """Up to `n` random non-laundering transactions with t0 <= ts < t1.

        `edge_id` is the time-sorted row index, so the window is one contiguous id
        range: draw ids in it, then keep the non-laundering ones.
        """
        key = (t0, t1)
        if key not in self._id_ranges:
            r = self.run(
                "MATCH ()-[t:TRANSFER]->() WHERE t.ts >= $t0 AND t.ts < $t1 "
                "RETURN min(t.edge_id) AS lo, max(t.edge_id) AS hi",
                t0=_utc(t0), t1=_utc(t1),
            )[0]
            self._id_ranges[key] = (r["lo"], r["hi"])
        lo, hi = self._id_ranges[key]
        if lo is None:
            return []
        ids = np.random.default_rng(seed).integers(lo, hi + 1, size=n).tolist()
        rows = self.run(
            "UNWIND $ids AS eid MATCH ()-[t:TRANSFER {edge_id: eid}]->() WHERE t.is_laundering = 0 "
            "RETURN DISTINCT eid",
            ids=ids,
        )
        return [r["eid"] for r in rows]

    def usd_rates(self) -> dict[str, float]:
        """USD per unit of each currency (same rule as `subgraphs.usd_rates_from_transfers`)."""
        rows = self.run(
            "MATCH ()-[t:TRANSFER]->() "
            "WHERE t.payment_currency <> t.receiving_currency "
            "  AND 'US Dollar' IN [t.payment_currency, t.receiving_currency] AND t.amount_paid > 0 "
            "WITH CASE WHEN t.receiving_currency = 'US Dollar' THEN t.payment_currency ELSE t.receiving_currency END AS cur, "
            "     CASE WHEN t.receiving_currency = 'US Dollar' THEN t.amount_received / t.amount_paid "
            "          ELSE t.amount_paid / t.amount_received END AS rate "
            "RETURN cur, percentileCont(rate, 0.5) AS rate"  # interpolated median, as polars
        )
        out = {r["cur"]: r["rate"] for r in rows}
        out["US Dollar"] = 1.0
        return out


_EDGE_RETURN = (
    "RETURN a.account_id AS src, b.account_id AS dst, t.edge_id AS edge_id, t.ts AS ts, "
    "t.amount_paid AS amount_paid, t.payment_currency AS payment_currency, "
    "t.payment_format AS payment_format, t.is_laundering AS is_laundering"
)


def _utc(dt: datetime) -> datetime:
    """Naive datetimes are UTC: the imported `ts:datetime` values carry no zone."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _naive(ts) -> datetime:
    return ts.to_native().replace(tzinfo=None) if hasattr(ts, "to_native") else ts


def _py_ts(rows: list[dict]) -> list[dict]:
    """neo4j DateTime -> python datetime so polars can build a column."""
    for r in rows:
        ts = r.get("ts")
        if ts is not None and not isinstance(ts, datetime):
            r["ts"] = _naive(ts)
    return rows
