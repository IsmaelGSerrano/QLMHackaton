"""Explore the AML graph in Neo4j and write a short dataset summary with figures.

Usage (Neo4j must already hold the dataset, see scripts/neo4j_load.py):
    uv run python scripts/dataset_summary.py --name HI-Small

Writes reports/<name>/stats.json and reports/<name>/figures/*.png. Every number
comes from a Cypher query against the loaded graph, so the report describes
exactly what the quantum pipeline reads (subgraphs are extracted from the same DB).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import igraph as ig
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from qlmaml.graphdb import GraphDB, Subgraph  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Colours: one neutral for legitimate, one accent for laundering, one per split.
C_LEGIT = "#8a94a6"
C_LAUND = "#d1495b"
C_NODE = "#2e6f95"
C_SPLIT = {"train": "#e8eef5", "val": "#fdf1d6", "test": "#f6e0e3"}
PATTERN_ORDER = ["FAN-OUT", "FAN-IN", "CYCLE", "BIPARTITE", "GATHER-SCATTER", "SCATTER-GATHER", "STACK", "RANDOM"]


def _native(v):
    """neo4j temporal -> ISO string, so stats.json is plain JSON."""
    if hasattr(v, "to_native"):
        v = v.to_native()
    if isinstance(v, datetime):
        return v.isoformat(timespec="minutes")
    return v


def _rows(rows: list[dict]) -> list[dict]:
    return [{k: _native(v) for k, v in r.items()} for r in rows]


# --------------------------------------------------------------------------- #
# Queries                                                                     #
# --------------------------------------------------------------------------- #


def collect_stats(db: GraphDB) -> dict:
    s: dict = {"counts": db.counts()}
    s["schema_properties"] = _rows(db.run(
        "CALL db.schema.relTypeProperties() YIELD relType, propertyName, propertyTypes "
        "RETURN relType, collect(propertyName) AS props ORDER BY relType"
    ))
    s["splits"] = _rows(db.run(
        "MATCH ()-[t:TRANSFER]->() "
        "RETURN t.split AS split, count(*) AS n, sum(t.is_laundering) AS laundering, "
        "min(t.ts) AS first, max(t.ts) AS last, min(t.day) AS day0, max(t.day) AS day1"
    ))
    s["daily"] = _rows(db.run(
        "MATCH ()-[t:TRANSFER]->() RETURN t.day AS day, t.split AS split, count(*) AS n, "
        "sum(t.is_laundering) AS laundering ORDER BY day"
    ))
    s["payment_format"] = _rows(db.run(
        "MATCH ()-[t:TRANSFER]->() RETURN t.payment_format AS format, count(*) AS n, "
        "sum(t.is_laundering) AS laundering ORDER BY n DESC"
    ))
    s["currency"] = _rows(db.run(
        "MATCH ()-[t:TRANSFER]->() RETURN t.payment_currency AS currency, count(*) AS n, "
        "sum(t.is_laundering) AS laundering, "
        "sum(CASE WHEN t.payment_currency <> t.receiving_currency THEN 1 ELSE 0 END) AS cross_currency "
        "ORDER BY n DESC"
    ))
    s["amount_percentiles"] = _rows(db.run(
        "MATCH ()-[t:TRANSFER]->() WITH t.is_laundering AS y, t.amount_paid AS x "
        "RETURN y, count(*) AS n, percentileCont(x, 0.1) AS p10, percentileCont(x, 0.5) AS p50, "
        "percentileCont(x, 0.9) AS p90 ORDER BY y"
    ))
    s["self_loops"] = db.run(
        "MATCH (a:Account)-[t:TRANSFER]->(a) RETURN count(t) AS n, sum(t.is_laundering) AS laundering"
    )[0]
    s["multi_edges"] = db.run(
        "MATCH ()-[s:SENDS_TO]->() RETURN count(s) AS pairs, sum(s.n_tx) AS tx, "
        "sum(CASE WHEN s.n_tx > 1 THEN s.n_tx ELSE 0 END) AS tx_in_repeated_pairs, max(s.n_tx) AS max_tx"
    )[0]
    s["cross_bank"] = db.run(
        "MATCH (a:Account)-[t:TRANSFER]->(b:Account) WHERE a.bank <> b.bank "
        "RETURN count(t) AS n, sum(t.is_laundering) AS laundering"
    )[0]
    s["degree"] = db.run(
        "MATCH (a:Account) WITH COUNT { (a)-[:TRANSFER]->() } AS o, COUNT { (a)<-[:TRANSFER]-() } AS i "
        "RETURN percentileCont(o, 0.5) AS out_p50, percentileCont(o, 0.99) AS out_p99, max(o) AS out_max, "
        "percentileCont(i, 0.5) AS in_p50, percentileCont(i, 0.99) AS in_p99, max(i) AS in_max, "
        "sum(CASE WHEN o + i = 1 THEN 1 ELSE 0 END) AS leaf_accounts"
    )[0]
    s["laundering_accounts"] = db.run(
        "MATCH (a:Account)-[t:TRANSFER {is_laundering: 1}]-() RETURN count(DISTINCT a) AS n"
    )[0]["n"]
    s["banks_top"] = _rows(db.run(
        "MATCH (b:Bank)<-[:AT_BANK]-(a:Account) WITH b, count(a) AS n RETURN b.bank_id AS bank, n "
        "ORDER BY n DESC LIMIT 5"
    ))
    s["patterns"] = _rows(db.run(
        "MATCH (p:LaunderingAttempt) OPTIONAL MATCH (p)-[:INVOLVES]->(a) WITH p, count(a) AS k "
        "RETURN p.pattern AS pattern, count(p) AS attempts, avg(p.n_tx) AS avg_tx, max(p.n_tx) AS max_tx, "
        "avg(k) AS avg_accounts, max(k) AS max_accounts, "
        "avg(duration.inSeconds(p.start_ts, p.end_ts).seconds) / 3600.0 AS avg_span_h ORDER BY attempts DESC"
    ))
    s["pattern_coverage"] = db.run(
        "MATCH ()-[t:TRANSFER {is_laundering: 1}]->() "
        "RETURN count(t) AS laundering, sum(CASE WHEN t.attempt_id IS NULL THEN 0 ELSE 1 END) AS in_patterns"
    )[0]
    return s


def sample_amounts(db: GraphDB, n_legit: int = 60_000) -> dict[str, np.ndarray]:
    """USD-paid amounts only (amounts are in the payment currency, so mixing currencies is meaningless)."""
    q = "MATCH ()-[t:TRANSFER {is_laundering: $y}]->() WHERE t.payment_currency = 'US Dollar' "
    laund = db.run(q + "RETURN t.amount_paid AS x", y=1)
    total = db.run("MATCH ()-[t:TRANSFER]->() WHERE t.payment_currency = 'US Dollar' RETURN count(t) AS n")[0]["n"]
    legit = db.run(q + "AND rand() < $p RETURN t.amount_paid AS x", y=0, p=min(1.0, n_legit / max(total, 1)))
    return {"legit": np.array([r["x"] for r in legit]), "laundering": np.array([r["x"] for r in laund])}


def example_attempts(db: GraphDB, min_tx: int = 4, max_tx: int = 10) -> dict[str, int]:
    """One readable attempt per motif type: the one with n_tx closest to 7 inside [min_tx, max_tx]."""
    rows = db.run(
        "MATCH (p:LaunderingAttempt) WHERE p.n_tx >= $lo AND p.n_tx <= $hi "
        "WITH p ORDER BY abs(p.n_tx - 7), p.attempt_id "
        "WITH p.pattern AS pattern, collect(p)[0] AS p "
        "RETURN pattern, toInteger(p.attempt_id) AS attempt_id",
        lo=min_tx, hi=max_tx,
    )
    found = {r["pattern"]: r["attempt_id"] for r in rows}
    # Fall back to the smallest attempt for motifs that are always large (e.g. big fan-outs).
    for pat in PATTERN_ORDER:
        if pat not in found:
            r = db.run(
                "MATCH (p:LaunderingAttempt {pattern: $pat}) RETURN toInteger(p.attempt_id) AS a "
                "ORDER BY p.n_tx LIMIT 1", pat=pat,
            )
            if r:
                found[pat] = r[0]["a"]
    return found


def example_transactions(db: GraphDB, attempt_id: int) -> list[dict]:
    return _rows(db.run(
        "MATCH (a)-[t:TRANSFER {attempt_id: $aid}]->(b) "
        "RETURN t.ts AS ts, a.account_id AS sender, b.account_id AS receiver, t.amount_paid AS amount, "
        "t.payment_currency AS currency, t.payment_format AS format, t.is_laundering AS laundering "
        "ORDER BY t.ts", aid=attempt_id,
    ))


def laundering_edge_with_context(db: GraphDB, min_nodes: int = 8, max_nodes: int = 12, tries: int = 60) -> int:
    """A pattern laundering transfer (train split, not a self-loop) whose 2-hop / 24-h ego-net has at
    least `min_nodes` accounts, so the example shows a laundering edge inside realistic legitimate context."""
    cands = db.run(
        "MATCH (a)-[t:TRANSFER {is_laundering: 1}]->(b) WHERE t.split = 'train' AND t.day >= 2 "
        "AND t.attempt_id IS NOT NULL AND a <> b RETURN t.edge_id AS eid ORDER BY eid LIMIT $k", k=tries,
    )
    best, best_n = cands[0]["eid"], 0
    for r in cands:
        n = db.ego_subgraph(r["eid"], hops=2, max_nodes=max_nodes).n_qubits
        if n >= min_nodes:
            return r["eid"]
        if n > best_n:
            best, best_n = r["eid"], n
    return best


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #


def _box(ax, xy, text, sub, color):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - 1.15, y - 0.45), 2.3, 0.9, boxstyle="round,pad=0.05,rounding_size=0.25",
                                fc=color, ec="none", zorder=3))
    ax.text(x, y + 0.12, text, ha="center", va="center", color="white", fontsize=12, weight="bold", zorder=4)
    ax.text(x, y - 0.2, sub, ha="center", va="center", color="white", fontsize=8.5, zorder=4)


def _arrow(ax, p, q, label, sub="", rad=0.0, color="#444", lw=1.6, label_xy=None):
    ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=16,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=2))
    lx, ly = label_xy if label_xy else ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    ax.text(lx, ly, label, ha="center", va="bottom", fontsize=10, weight="bold", color=color,
            bbox=dict(fc="white", ec="none", pad=1), zorder=5)
    if sub:
        ax.text(lx, ly - 0.08, sub, ha="center", va="top", fontsize=7.5, color="#333", zorder=5)


def fig_schema(stats: dict, out: Path) -> None:
    c = stats["counts"]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-3.3, 3.2)
    ax.axis("off")
    snd, rcv, bank, att = (-3.4, 0.0), (3.4, 0.0), (0.0, -2.6), (0.0, 2.5)
    _box(ax, snd, "Account", f"{c['Account']:,} nodes  (sender)", C_NODE)
    _box(ax, rcv, "Account", "(receiver)", C_NODE)
    _box(ax, bank, "Bank", f"{c['Bank']:,} nodes", "#5b6c7d")
    _box(ax, att, "LaunderingAttempt", f"{c['LaunderingAttempt']:,} nodes  (ground-truth motifs)", C_LAUND)

    _arrow(ax, (snd[0] + 1.2, 0.18), (rcv[0] - 1.2, 0.18), f"TRANSFER  ({c['TRANSFER']:,})",
           "multigraph: one edge per transaction\nts, day, split, amount_paid/received, currencies,\n"
           "payment_format, is_laundering, attempt_id?, pattern?", rad=-0.12, color=C_LAUND, lw=2.2,
           label_xy=(0, 1.62))
    _arrow(ax, (snd[0] + 1.2, -0.25), (rcv[0] - 1.2, -0.25), f"SENDS_TO  ({c['SENDS_TO']:,})",
           "simple graph: one edge per (sender, receiver, split)\nn_tx, total_paid, first_ts, last_ts, n_laundering",
           rad=0.12, color="#2e6f95", label_xy=(0, -0.78))
    _arrow(ax, (snd[0] + 0.5, -0.5), (bank[0] - 1.0, bank[1] + 0.3), f"AT_BANK ({c['AT_BANK']:,})", rad=0.15,
           color="#5b6c7d", label_xy=(-2.3, -1.85))
    _arrow(ax, (rcv[0] - 0.5, -0.5), (bank[0] + 1.0, bank[1] + 0.3), "AT_BANK", rad=-0.15, color="#5b6c7d",
           label_xy=(2.3, -1.85))
    _arrow(ax, (att[0] - 1.0, att[1] - 0.45), (snd[0] + 0.4, 0.5), "INVOLVES", "role: sender|receiver|both",
           rad=0.1, color=C_LAUND, label_xy=(-2.5, 1.85))
    _arrow(ax, (att[0] + 1.0, att[1] - 0.45), (rcv[0] - 0.4, 0.5), f"INVOLVES ({c['INVOLVES']:,})", rad=-0.1,
           color=C_LAUND, label_xy=(2.5, 1.85))
    ax.set_title("Graph schema in Neo4j (both Account boxes are the same label)", fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_timeline(stats: dict, out: Path) -> None:
    daily = stats["daily"]
    days = np.array([r["day"] for r in daily])
    n = np.array([r["n"] for r in daily])
    lab = np.array([r["laundering"] for r in daily])
    split = [r["split"] for r in daily]
    fig, ax1 = plt.subplots(figsize=(10, 3.8))
    for d, sp in zip(days, split):
        ax1.axvspan(d - 0.5, d + 0.5, color=C_SPLIT[sp], zorder=0)
    ax1.bar(days, n, color=C_LEGIT, width=0.75, zorder=2, label="transactions / day")
    ax1.set_yscale("log")
    ax1.set_ylabel("transactions per day (log)")
    ax1.set_xlabel("day index (day 0 = first day in the file)")
    ax2 = ax1.twinx()
    rate = np.divide(lab, n, out=np.zeros_like(lab, dtype=float), where=n > 0) * 100
    ax2.plot(days, rate, "o-", color=C_LAUND, lw=1.8, ms=4, zorder=3, label="% laundering")
    ax2.set_ylabel("% laundering", color=C_LAUND)
    for sp in ("train", "val", "test"):
        idx = [i for i, s in enumerate(split) if s == sp]
        if idx:
            ax1.text(days[idx].mean(), ax1.get_ylim()[1] * 0.55, sp, ha="center", fontsize=10, weight="bold",
                     color="#333")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", bbox_to_anchor=(0.99, 0.88), frameon=True, fontsize=8)
    ax1.set_title("Daily volume and laundering rate, coloured by temporal split", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_categorical(stats: dict, out: Path) -> None:
    fmt = stats["payment_format"]
    pats = sorted(stats["patterns"], key=lambda r: -r["attempts"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    names = [r["format"] for r in fmt]
    rates = [100 * r["laundering"] / r["n"] for r in fmt]
    ax1.barh(names, rates, color=[C_LAUND if r == max(rates) else C_LEGIT for r in rates])
    for i, r in enumerate(fmt):
        ax1.text(rates[i], i, f"  {rates[i]:.3f}%  (n={r['n']:,})", va="center", fontsize=8)
    ax1.set_xlabel("% of transactions that are laundering")
    ax1.set_xlim(0, max(rates) * 1.9)
    ax1.invert_yaxis()
    ax1.set_title("Laundering rate by payment format", loc="left", fontsize=11)
    ax2.barh([r["pattern"] for r in pats], [r["attempts"] for r in pats], color=C_LAUND)
    for i, r in enumerate(pats):
        ax2.text(r["attempts"], i, f"  {r['attempts']}  (avg {r['avg_tx']:.1f} tx, {r['avg_accounts']:.1f} acc.)",
                 va="center", fontsize=8)
    ax2.set_xlim(0, max(r["attempts"] for r in pats) * 1.7)
    ax2.invert_yaxis()
    ax2.set_xlabel("laundering attempts in *_Patterns.txt")
    ax2.set_title("Ground-truth motif types", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_amounts(amounts: dict[str, np.ndarray], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    logs = {k: np.log10(v[v > 0]) for k, v in amounts.items()}
    lo, hi = np.floor(min(v.min() for v in logs.values())), np.ceil(max(v.max() for v in logs.values()))
    bins = np.arange(lo, hi + 0.25, 0.25)
    for key, col in (("legit", C_LEGIT), ("laundering", C_LAUND)):
        v = logs[key]
        w = np.full(v.size, 100.0 / v.size)
        ax.hist(v, bins=bins, weights=w, histtype="stepfilled" if key == "legit" else "step", color=col,
                alpha=0.6 if key == "legit" else 1, lw=2,
                label=f"{key} (n={v.size:,}{', random sample' if key == 'legit' else ''}), median ${10**np.median(v):,.0f}")
    ax.set_xticks(np.arange(lo, hi + 1, 1))
    ax.set_xticklabels([f"$10^{{{int(t)}}}$" for t in np.arange(lo, hi + 1, 1)])
    ax.set_xlabel("amount paid in US dollars (log scale)")
    ax.set_ylabel("% of transactions per bin")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("USD transaction amounts: legitimate vs laundering", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _draw_subgraph(ax, sg: Subgraph, title: str, center_edge: int | None = None) -> None:
    n = sg.n_qubits
    pairs = sg.undirected_pairs()
    g = ig.Graph(n=n, edges=pairs)
    lay = np.array(g.layout_kamada_kawai().coords) if pairs else np.zeros((n, 2))
    if n > 2 and min(np.ptp(lay[:, 0]), np.ptp(lay[:, 1])) < 0.15 * max(np.ptp(lay[:, 0]), np.ptp(lay[:, 1])):
        lay = np.array(g.layout_circle().coords)  # kamada-kawai puts paths on a line; arrows hide behind nodes
    if n > 1:
        lay = (lay - lay.mean(0)) / (np.abs(lay - lay.mean(0)).max() + 1e-9)
    e = sg.edges.sort("ts")
    seen: dict[tuple[int, int], int] = {}
    for k, row in enumerate(e.iter_rows(named=True)):
        u, v = int(row["u"]), int(row["v"])
        laund = row["is_laundering"] == 1
        col = C_LAUND if laund else C_LEGIT
        lw = 2.4 if row["edge_id"] == center_edge else 1.4
        key = (min(u, v), max(u, v))
        m = seen.get(key, 0)
        seen[key] = m + 1
        if u == v:
            x, y = lay[u]
            ax.add_patch(FancyArrowPatch((x - 0.04, y + 0.05), (x + 0.04, y + 0.05),
                                         connectionstyle="arc3,rad=-2.5", arrowstyle="-|>", mutation_scale=9,
                                         color=col, lw=lw, zorder=1))
            continue
        rad = 0.12 + 0.14 * m if m else 0.08  # parallel edges fan out
        ax.add_patch(FancyArrowPatch(lay[u], lay[v], connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                     mutation_scale=10, color=col, lw=lw, shrinkA=9, shrinkB=9, zorder=1))
    ax.scatter(lay[:, 0], lay[:, 1], s=230, c=C_NODE, zorder=3, edgecolors="white", linewidths=1.5)
    for q in range(n):
        ax.text(lay[q, 0], lay[q, 1], str(q), ha="center", va="center", color="white", fontsize=7, zorder=4,
                weight="bold")
    ax.set_title(title, fontsize=9.5)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_motifs(db: GraphDB, examples: dict[str, int], out: Path) -> list[dict]:
    pats = [p for p in PATTERN_ORDER if p in examples]
    cols = 4
    rows = int(np.ceil(len(pats) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.1 * rows))
    info = []
    for ax, pat in zip(np.ravel(axes), pats):
        sg = db.attempt_subgraph(examples[pat])
        span = (sg.edges["ts"].max() - sg.edges["ts"].min())
        hours = span.total_seconds() / 3600 if span is not None else 0
        _draw_subgraph(ax, sg, f"{pat}\nattempt {examples[pat]}: {sg.n_qubits} accounts, {sg.edges.height} tx, "
                               f"{hours:.0f} h")
        info.append({"pattern": pat, "attempt_id": examples[pat], "accounts": sg.n_qubits, "tx": sg.edges.height,
                     "span_h": round(hours, 1)})
    for ax in np.ravel(axes)[len(pats):]:
        ax.axis("off")
    fig.suptitle("One ground-truth laundering attempt per motif type (node = account = qubit index)", x=0.01,
                 ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return info


def fig_egonet(db: GraphDB, edge_id: int, out: Path, max_nodes: int = 12) -> dict:
    sg = db.ego_subgraph(edge_id, hops=2, max_nodes=max_nodes)
    fig, ax = plt.subplots(figsize=(6, 6))
    n_l = int(sg.edges["is_laundering"].sum())
    _draw_subgraph(ax, sg, f"Ego-net of laundering transaction {edge_id} (thick edge): 2 hops, 24 h before it,\n"
                           f"capped at {max_nodes} accounts: {sg.n_qubits} qubits, {sg.edges.height} transfers "
                           f"({n_l} laundering), {len(sg.undirected_pairs())} account pairs", center_edge=edge_id)
    ax.plot([], [], color=C_LAUND, lw=2, label="laundering transfer")
    ax.plot([], [], color=C_LEGIT, lw=2, label="legitimate transfer")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return {"edge_id": edge_id, "qubits": sg.n_qubits, "transfers": sg.edges.height, "laundering": n_l,
            "pairs": len(sg.undirected_pairs())}


# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="HI-Small", help="dataset name, used for the output folder")
    args = ap.parse_args()
    out = ROOT / "reports" / args.name
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    with GraphDB() as db:
        stats = collect_stats(db)
        print("counts:", stats["counts"])
        fig_schema(stats, figs / "schema.png")
        fig_timeline(stats, figs / "timeline.png")
        fig_categorical(stats, figs / "formats_and_motifs.png")
        fig_amounts(sample_amounts(db), figs / "amounts.png")
        examples = example_attempts(db)
        stats["example_attempts"] = fig_motifs(db, examples, figs / "motif_examples.png")
        cyc = examples.get("CYCLE", next(iter(examples.values())))
        stats["example_transactions"] = {"attempt_id": cyc, "rows": example_transactions(db, cyc)}
        stats["example_egonet"] = fig_egonet(db, laundering_edge_with_context(db), figs / "egonet.png")

    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out / 'stats.json'} and {len(list(figs.glob('*.png')))} figures in {figs}")


if __name__ == "__main__":
    main()
