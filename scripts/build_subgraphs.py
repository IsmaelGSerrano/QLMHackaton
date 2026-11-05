"""Extract the labelled subgraph dataset (prototype plan, WP1) into a parquet cache.

Usage:
    uv run python scripts/build_subgraphs.py                       # HI-Small from Neo4j, n=8 and n=12
    uv run python scripts/build_subgraphs.py --max-nodes 8
    uv run python scripts/build_subgraphs.py --fake                # synthetic data, no Neo4j

Writes data/processed/subgraphs_<name>_n<k>.parquet (git-ignored) and a
summary JSON next to it (and, for real data, in reports/<name>/subgraphs_n<k>.json):
counts per split and label, positive:negative ratio,
how many attempts were cut at the qubit cap, which positives were dropped
for lack of matching negatives (by pattern), and how well negatives match the
positives on edge count. Training and hardware scripts read the parquet and
do not need Neo4j. See `qlmaml.subgraphs` for the sampling rules.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

from qlmaml.subgraphs import SPLITS, Dataset, FrameGraph, build_dataset, write_cache

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"


def summarize(dataset: Dataset, args: argparse.Namespace, max_nodes: int) -> dict:
    ds = dataset.samples
    pos = [s for s in ds if s.label == 1]
    neg = [s for s in ds if s.label == 0]
    by_id = {p.source_id: p for p in pos}
    edge_gap = [abs(n.n_edges - by_id[n.matched_to].n_edges) for n in neg]
    return {
        "max_nodes": max_nodes,
        "ratio": f"1:{args.ratio}",
        "positives": len(pos),
        "negatives": len(neg),
        "negatives_by_kind": dict(sorted(Counter(n.kind for n in neg).items())),
        "dropped_positives": len(dataset.dropped),
        "dropped_by_pattern": dict(sorted(Counter(p.pattern for p in dataset.dropped).items())),
        "dropped_shapes": sorted({(p.graph.n_qubits, p.n_edges) for p in dataset.dropped}),
        "attempts_cut": sum(p.cut for p in pos),
        "per_split": {k: {"pos": sum(1 for s in pos if s.split == k), "neg": sum(1 for s in neg if s.split == k)}
                      for k in SPLITS},
        "split_start": {k: str(min((s.start_ts for s in ds if s.split == k), default=None)) for k in SPLITS},
        "positives_per_pattern": dict(sorted(Counter(p.pattern for p in pos).items())),
        "nodes_hist": {"pos": dict(sorted(Counter(s.graph.n_qubits for s in pos).items())),
                       "neg": dict(sorted(Counter(s.graph.n_qubits for s in neg).items()))},
        "edges_hist": {"pos": dict(sorted(Counter(s.n_edges for s in pos).items())),
                       "neg": dict(sorted(Counter(s.n_edges for s in neg).items()))},
        "edge_match_abs_gap": dict(sorted(Counter(edge_gap).items())),
        "settings": {"edge_tol": args.edge_tol, "hops": args.hops, "window_h": args.window_h,
                     "max_tries": args.max_tries, "seed": args.seed, "source": "fake" if args.fake else "neo4j"},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="HI-Small", help="dataset name, used in the output file name")
    ap.add_argument("--max-nodes", type=int, nargs="+", default=[8, 12], help="qubit cap(s): 8 hardware, 12 simulator")
    ap.add_argument("--ratio", type=int, default=5, help="negatives per positive")
    ap.add_argument("--edge-tol", type=int, default=1, help="max |merged edges(neg) - merged edges(pos)|")
    ap.add_argument("--hops", type=int, default=2, help="ego-net depth for negatives")
    ap.add_argument("--window-h", type=float, default=24, help="ego-net look-back window in hours")
    ap.add_argument("--max-tries", type=int, default=200, help="candidate negatives drawn per positive")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true", help="synthetic data through the in-memory source (no Neo4j)")
    args = ap.parse_args()

    if args.fake:
        from qlmaml.fake_data import write_fake_transactions

        args.name = "FAKE-Small"
        trans = write_fake_transactions(OUT / "FAKE-Small_Trans.csv", n_accounts=300, n_normal=6_000, n_motifs=30,
                                        patterns_path=OUT / "FAKE-Small_Patterns.txt")
        src = FrameGraph.from_files(trans, OUT / "FAKE-Small_Patterns.txt")
        close = lambda: None  # noqa: E731
    else:
        from qlmaml.graphdb import GraphDB

        src = GraphDB()
        close = src.close

    try:
        rates = src.usd_rates()
        print(f"USD rates for {len(rates)} currencies")
        for k in args.max_nodes:
            t0 = time.time()
            ds = build_dataset(src, max_nodes=k, ratio=args.ratio, edge_tol=args.edge_tol, hops=args.hops,
                               window=timedelta(hours=args.window_h), max_tries=args.max_tries, seed=args.seed)
            path = write_cache(ds.samples, OUT / f"subgraphs_{args.name}_n{k}.parquet", usd_rates=rates)
            summary = summarize(ds, args, k)
            path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            if not args.fake:  # counts only, no data: committed so slide numbers are traceable
                report = ROOT / "reports" / args.name / f"subgraphs_n{k}.json"
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"n={k}: {summary['positives']} pos, {summary['negatives']} neg {summary['negatives_by_kind']}, "
                  f"{summary['dropped_positives']} dropped {summary['dropped_by_pattern']}, "
                  f"{summary['attempts_cut']} cut ({time.time() - t0:.0f}s) -> {path}")
    finally:
        close()


if __name__ == "__main__":
    main()
