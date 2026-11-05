"""Labelled subgraph dataset for the quantum graph classifier (prototype plan, WP1).

Task: graph classification. Input: a small directed transaction subgraph, one
qubit per account. Output: a score for "contains a laundering attempt".

- **Positive:** the accounts of one attempt from `*_Patterns.txt`, as the
  *induced* subgraph over the attempt's time window (background transfers
  between those accounts are included). Attempts larger than `max_nodes` are
  **cut in time**: accounts are kept in the order the money reaches them, and
  the window ends at the last attempt transfer that needs no new account
  beyond the cap. This is subsampling and must be said on the slides.
- **Negative:** an ego-net (`ego_subgraph`) around a normal transaction,
  without any laundering transfer (stricter than "center not laundering": a
  negative must not contain an attempt), matched to a positive of the same temporal
  split on node count (exact) and merged edge count (within `edge_tol`).
  `ratio` negatives per positive (default 1:5).
- **Split:** temporal, by the start time of the subgraph (its first transfer).
  Cut points are the 60 % / 80 % quantiles of the positives' start times;
  negatives use the same cut points. There is no gap between splits, so a
  subgraph that starts just before a cut point can contain transfers from the
  next period (a positive runs to its attempt's end, a negative looks back
  `window`). Say so with the results.
- **Known shortcut risk:** positive windows last as long as the attempt (often
  days); negative windows are `window` long (1 day). A model that sees edge
  timestamps could learn the time span. WP2 uses timestamps only for gate order.

Both classes come from the same graph source and the same induced-subgraph
semantics, so the model cannot separate them by "clean motif vs. noisy ego-net".
Sources: `qlmaml.graphdb.GraphDB` (Neo4j) and `FrameGraph` (in-memory polars,
the same queries over the same import tables; used by tests and `--fake`).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl

from qlmaml.data import load_patterns, load_transactions
from qlmaml.graphdb import ImportTables, Subgraph, _subgraph_from_records, build_import_tables

SPLITS = ("train", "val", "test")
_EDGE_COLS = ["src", "dst", "edge_id", "ts", "amount_paid", "payment_currency", "payment_format", "is_laundering"]


class GraphSource(Protocol):
    """The queries the dataset builder needs. `GraphDB` and `FrameGraph` implement it."""

    def attempts(self) -> pl.DataFrame: ...  # attempt_id, pattern, start_ts, end_ts

    def attempt_subgraph(self, attempt_id: int) -> Subgraph: ...

    def induced_subgraph(self, account_ids: list[str], t0: datetime, t1: datetime,
                         center: int | None = None) -> Subgraph: ...

    def ego_subgraph(self, edge_id: int, hops: int = 1, max_nodes: int = 12,
                     window: timedelta = timedelta(days=1)) -> Subgraph: ...

    def normal_edge_ids(self, t0: datetime, t1: datetime, n: int, seed: int) -> list[int]: ...

    def usd_rates(self) -> dict[str, float]: ...


@dataclass(eq=False)  # identity equality: fields hold DataFrames
class Sample:
    """One labelled subgraph."""

    graph: Subgraph
    label: int  # 1 = contains a laundering attempt
    pattern: str | None  # attempt motif type, None for negatives
    source_id: int  # attempt_id (positives) or center edge_id (negatives)
    start_ts: datetime  # first transfer in the subgraph; decides the split
    cut: bool = False  # True if the attempt was cut at max_nodes
    split: str | None = None
    matched_to: int | None = None  # negatives: attempt_id of the positive they are matched to

    @cached_property
    def n_edges(self) -> int:
        """Merged edges between distinct accounts: one two-qubit gate per account pair and direction.

        Self-transfers (u -> u, common for "Reinvestment") stay in the subgraph
        but are no gate, so they do not count here or in the matching.
        """
        return self.graph.simple_edges().filter(pl.col("u") != pl.col("v")).height


# --------------------------------------------------------------------------- #
# Extraction                                                                  #
# --------------------------------------------------------------------------- #


def positive_sample(src: GraphSource, attempt_id: int, max_nodes: int, pattern: str | None = None) -> Sample:
    """Induced subgraph on the accounts of one attempt, cut in time at `max_nodes` accounts.

    `pattern` saves one `attempts()` lookup per call when the caller has it already.
    """
    if max_nodes < 2:
        raise ValueError("max_nodes must be at least 2")
    att = src.attempt_subgraph(attempt_id)
    if pattern is None:
        pattern = src.attempts().filter(pl.col("attempt_id") == attempt_id)["pattern"].item()
    selected: list[str] = []
    t_end, cut = None, False
    for a, b, ts in att.edges.sort("ts", "amount_paid", "edge_id").select("src", "dst", "ts").iter_rows():
        new = [x for x in dict.fromkeys((a, b)) if x not in selected]
        if len(selected) + len(new) > max_nodes:
            cut = True
            break
        selected += new
        t_end = ts
    graph = src.induced_subgraph(selected, att.edges["ts"].min(), t_end)
    return Sample(graph=graph, label=1, pattern=pattern, source_id=attempt_id,
                  start_ts=graph.edges["ts"].min(), cut=cut)


def negative_sample(src: GraphSource, edge_id: int, max_nodes: int, hops: int = 2,
                    window: timedelta = timedelta(days=1)) -> Sample | None:
    """Ego-net around normal transaction `edge_id`; None if it contains any laundering transfer."""
    g = src.ego_subgraph(edge_id, hops=hops, max_nodes=max_nodes, window=window)
    if g.edges["is_laundering"].sum() > 0:
        return None
    return Sample(graph=g, label=0, pattern=None, source_id=edge_id, start_ts=g.edges["ts"].min())


def split_cuts(start_ts: list[datetime], fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> tuple[datetime, datetime]:
    """Cut points (val start, test start) at the `fractions` quantiles of `start_ts`."""
    s = sorted(start_ts)
    n = len(s)
    i = min(n - 1, round(fractions[0] * n))
    j = min(n - 1, round((fractions[0] + fractions[1]) * n))
    return s[i], s[j]


def assign_split(ts: datetime, cuts: tuple[datetime, datetime]) -> str:
    return "train" if ts < cuts[0] else "val" if ts < cuts[1] else "test"


def build_dataset(
    src: GraphSource,
    max_nodes: int,
    ratio: int = 5,
    edge_tol: int = 1,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    hops: int = 2,
    window: timedelta = timedelta(days=1),
    max_tries: int = 200,
    seed: int = 0,
) -> list[Sample]:
    """Positives for every attempt, `ratio` matched negatives per positive, temporal split.

    Negatives are drawn from normal transactions in the positive's split period.
    A drawn ego-net that does not fit the current positive goes to a pool and can
    match a later one. If `max_tries` candidates do not give `ratio` matches, the
    positive keeps fewer negatives and a warning says so.
    """
    pos = [positive_sample(src, aid, max_nodes, pattern=pat)
           for aid, pat in src.attempts().sort("attempt_id").select("attempt_id", "pattern").iter_rows()]
    if not pos:
        raise ValueError("graph source has no laundering attempts")
    cuts = split_cuts([p.start_ts for p in pos], fractions)
    for p in pos:
        p.split = assign_split(p.start_ts, cuts)
    t_lo = min(p.start_ts for p in pos)
    t_hi = max(p.graph.edges["ts"].max() for p in pos) + timedelta(seconds=1)
    period = {"train": (t_lo, cuts[0]), "val": (cuts[0], cuts[1]), "test": (cuts[1], t_hi)}

    rng = np.random.default_rng(seed)
    pool: dict[str, list[Sample]] = {k: [] for k in SPLITS}
    tried: set[int] = set()
    neg: list[Sample] = []
    for k in rng.permutation(len(pos)):
        p = pos[k]
        n_p, e_p = p.graph.n_qubits, p.n_edges
        found, tries = 0, 0
        while found < ratio:
            fits = [s for s in pool[p.split] if s.graph.n_qubits == n_p and abs(s.n_edges - e_p) <= edge_tol]
            if fits:
                best = min(fits, key=lambda s: (abs(s.n_edges - e_p), s.source_id))
                pool[p.split].remove(best)
                best.matched_to = p.source_id
                neg.append(best)
                found += 1
                continue
            if tries >= max_tries:
                warnings.warn(f"attempt {p.source_id}: only {found}/{ratio} matched negatives "
                              f"(n={n_p}, edges={e_p}, split={p.split})", stacklevel=2)
                break
            batch = 20
            ids = [i for i in src.normal_edge_ids(*period[p.split], n=batch, seed=int(rng.integers(2**31)))
                   if i not in tried]
            tries += batch if not ids else 0  # period exhausted: still count, so the loop ends
            for i in ids:
                tried.add(i)
                tries += 1
                s = negative_sample(src, i, max_nodes=n_p, hops=hops, window=window)
                if s is not None:
                    s.split = assign_split(s.start_ts, cuts)
                    pool[s.split].append(s)
    return pos + neg


# --------------------------------------------------------------------------- #
# Parquet cache                                                               #
# --------------------------------------------------------------------------- #

_CACHE_EDGE_COLS = ["u", "v", "first_ts", "total_usd", "n_tx"]


def write_cache(samples: list[Sample], path: str | Path, usd_rates: dict[str, float]) -> Path:
    """One row per subgraph: split, label, pattern, n_qubits, n_edges, ids, nodes, edges.

    `nodes` lists the account ids in qubit order. `edges` lists the merged edges
    ({u, v, first_ts, total_usd, n_tx}) in gate order (first timestamp, ties by
    amount). Self-transfers (u == v) are kept for node features, but `n_edges`
    does not count them. The per-edge laundering flag is left out on purpose:
    it is the label.
    """
    rows = []
    for s in samples:
        edges = s.graph.simple_edges(usd_rates=usd_rates).select(_CACHE_EDGE_COLS)
        rows.append({
            "split": s.split, "label": s.label, "pattern": s.pattern,
            "n_qubits": s.graph.n_qubits, "n_edges": s.n_edges,
            "source_id": s.source_id, "matched_to": s.matched_to, "start_ts": s.start_ts, "cut": s.cut,
            "nodes": s.graph.nodes["account_id"].to_list(),
            "edges": edges.with_columns(pl.col("u", "v").cast(pl.Int64)).to_dicts(),
        })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Scan all rows for the schema: positives come first with matched_to = None.
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(path)
    return path


def read_cache(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(path)


# --------------------------------------------------------------------------- #
# In-memory graph source                                                      #
# --------------------------------------------------------------------------- #


class FrameGraph:
    """`GraphSource` over the `ImportTables` that `scripts/neo4j_load.py` imports into Neo4j.

    Same semantics as the Cypher queries in `GraphDB`, without a database. Fine for
    FAKE data and for HI-Small (5 M rows fit in memory); Neo4j stays the default.
    """

    def __init__(self, tables: ImportTables):
        self.transfers = tables.transfers.rename({"start": "src", "end": "dst"})
        self.node_id = dict(zip(tables.accounts["account_id"], tables.accounts["node_id"]))
        self._attempts = tables.attempts.select("attempt_id", "pattern", "start_ts", "end_ts")

    @classmethod
    def from_files(cls, trans_csv: str | Path, patterns_txt: str | Path | None = None) -> FrameGraph:
        df = load_transactions(trans_csv)
        patterns = load_patterns(patterns_txt) if patterns_txt else None
        return cls(build_import_tables(df, patterns))

    def attempts(self) -> pl.DataFrame:
        return self._attempts

    def usd_rates(self) -> dict[str, float]:
        return usd_rates_from_transfers(self.transfers)

    def normal_edge_ids(self, t0: datetime, t1: datetime, n: int, seed: int) -> list[int]:
        """Up to `n` random non-laundering transactions with t0 <= ts < t1."""
        ids = self.transfers.filter(
            (pl.col("is_laundering") == 0) & (pl.col("ts") >= t0) & (pl.col("ts") < t1)
        )["edge_id"]
        return ids.sample(min(n, ids.len()), seed=seed).to_list()

    def _subgraph(self, account_ids: list[str], edges: pl.DataFrame, center: int | None) -> Subgraph:
        accts = [{"account_id": a, "node_id": self.node_id[a]} for a in account_ids]
        return _subgraph_from_records(accts, edges.select(_EDGE_COLS).to_dicts(), center=center)

    def attempt_subgraph(self, attempt_id: int) -> Subgraph:
        e = self.transfers.filter(pl.col("attempt_id") == attempt_id)
        ids = sorted(set(e["src"]) | set(e["dst"]), key=self.node_id.__getitem__)
        return self._subgraph(ids, e, center=None)

    def induced_subgraph(self, account_ids: list[str], t0: datetime, t1: datetime,
                         center: int | None = None) -> Subgraph:
        e = self.transfers.filter(
            pl.col("src").is_in(account_ids) & pl.col("dst").is_in(account_ids)
            & (pl.col("ts") >= t0) & (pl.col("ts") <= t1)
        )
        return self._subgraph(list(account_ids), e, center=center)

    def ego_subgraph(self, edge_id: int, hops: int = 1, max_nodes: int = 12,
                     window: timedelta = timedelta(days=1)) -> Subgraph:
        """Same as `GraphDB.ego_subgraph`: breadth-first, most recent counterparties first."""
        c = self.transfers.filter(pl.col("edge_id") == edge_id)
        if c.is_empty():
            raise KeyError(f"no TRANSFER with edge_id={edge_id}")
        row = c.row(0, named=True)
        t_end = row["ts"]
        t_start = t_end - window
        w = self.transfers.filter((pl.col("ts") >= t_start) & (pl.col("ts") <= t_end))
        selected = list(dict.fromkeys([row["src"], row["dst"]]))
        frontier = list(selected)
        for _ in range(hops):
            if len(selected) >= max_nodes or not frontier:
                break
            touching = pl.concat([
                w.filter(pl.col("src").is_in(frontier)).select(pl.col("dst").alias("n"), "ts"),
                w.filter(pl.col("dst").is_in(frontier)).select(pl.col("src").alias("n"), "ts"),
            ])
            frontier = (
                touching.filter(~pl.col("n").is_in(selected))
                .group_by("n").agg(pl.col("ts").max().alias("last"))
                .sort(["last", "n"], descending=[True, False])
                .head(max_nodes - len(selected))["n"].to_list()
            )
            selected.extend(frontier)
        return self.induced_subgraph(selected, t_start, t_end, center=edge_id)


def usd_rates_from_transfers(t: pl.DataFrame) -> dict[str, float]:
    """USD per unit of each currency: median rate over cross-currency transfers to or from USD.

    The IBM generator converts with fixed daily-free rates, so the median recovers them.
    Same rule as `GraphDB.usd_rates`.
    """
    usd = "US Dollar"
    rates = pl.concat([
        t.filter(pl.col("receiving_currency") == usd).select(
            pl.col("payment_currency").alias("currency"),
            (pl.col("amount_received") / pl.col("amount_paid")).alias("rate")),
        t.filter(pl.col("payment_currency") == usd).select(
            pl.col("receiving_currency").alias("currency"),
            (pl.col("amount_paid") / pl.col("amount_received")).alias("rate")),
    ]).filter(pl.col("rate").is_finite() & (pl.col("rate") > 0))
    out = dict(rates.group_by("currency").agg(pl.col("rate").median()).iter_rows())
    out[usd] = 1.0
    return out
