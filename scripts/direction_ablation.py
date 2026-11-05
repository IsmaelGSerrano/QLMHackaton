"""WP2 slide figure: the circuit output per motif, for the directed ZX gate vs the symmetric RZZ gate.

    uv run python scripts/direction_ablation.py            # 200 random parameter draws, p = 1
    uv run python scripts/direction_ablation.py --draws 50 --p 2

Two motif pairs have the same undirected shape, amounts and timestamps and differ only in direction:
FAN-OUT vs FAN-IN, and CYCLE (the money comes back) vs the no-return triangle. For each random
parameter draw we compute s = mean_i <Z_i> (noiseless, exact) for both motifs of a pair and plot one
against the other. RZZ puts every point on the diagonal: it cannot separate the pair for any
parameters, so no training can. ZX moves the points off the diagonal (ADR 0001; Egressy et al. [2]
for why direction is needed; Skolik et al. [9] for the symmetric RZZ construction).

These are hand-built toy motifs, not HI-Small data, and nothing is trained. So this is not a model
experiment and it does not call record_run. It writes:
    results/figures/direction_ablation.png
    results/figures/direction_ablation.json   (max and median |s_first - s_second| per gate and pair)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from qlmaml.circuit import build_gates, merge_transactions, n_params, pennylane_expvals  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "figures"

# (payer, payee, amount_usd, timestamp). Both motifs of a pair share amounts and timestamps.
FAN_OUT = [(0, 1, 1000.0, 1.0), (0, 2, 2500.0, 2.0), (0, 3, 400.0, 3.0)]
FAN_IN = [(v, u, amount, t) for u, v, amount, t in FAN_OUT]
CYCLE = [(0, 1, 3000.0, 1.0), (1, 2, 2900.0, 2.0), (2, 0, 2800.0, 3.0)]
NO_RETURN = [(0, 1, 3000.0, 1.0), (1, 2, 2900.0, 2.0), (0, 2, 2800.0, 3.0)]
PAIRS = [("FAN-OUT", "FAN-IN", 4, FAN_OUT, FAN_IN), ("CYCLE", "no-return triangle", 3, CYCLE, NO_RETURN)]
GATES = ["zx", "rzz"]


def mean_z(n, transactions, edge_gate, params, p):
    graph = merge_transactions(n, transactions)
    return float(np.mean(pennylane_expvals(build_gates(graph, p=p, edge_gate=edge_gate), n, params)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--p", type=int, default=1, choices=[1, 2])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    draws = rng.uniform(-np.pi, np.pi, (args.draws, n_params(args.p)))
    fig, axes = plt.subplots(len(PAIRS), len(GATES), figsize=(8, 8), sharex=True, sharey=True)
    summary = {"draws": args.draws, "p": args.p, "seed": args.seed, "pairs": {}}
    for row, (name_a, name_b, n, motif_a, motif_b) in enumerate(PAIRS):
        for col, gate in enumerate(GATES):
            s_a = np.array([mean_z(n, motif_a, gate, params, args.p) for params in draws])
            s_b = np.array([mean_z(n, motif_b, gate, params, args.p) for params in draws])
            gap = np.abs(s_a - s_b)
            summary["pairs"].setdefault(f"{name_a} vs {name_b}", {})[gate] = {
                "max_abs_diff": float(gap.max()),
                "median_abs_diff": float(np.median(gap)),
            }
            ax = axes[row, col]
            ax.plot([-1, 1], [-1, 1], color="0.7", lw=1, zorder=0)
            ax.scatter(s_a, s_b, s=10, alpha=0.6)
            ax.set_title(f"{gate.upper()}: median |Δs| = {np.median(gap):.3f}")
            ax.set_xlabel(f"s({name_a})")
            ax.set_ylabel(f"s({name_b})")
            ax.set_aspect("equal")
            print(f"{name_a} vs {name_b} [{gate}]: max |Δs| = {gap.max():.2e}, median = {np.median(gap):.2e}")
    fig.suptitle(f"s = mean ⟨Z⟩ per motif, {args.draws} random parameter draws, p = {args.p} (noiseless)")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "direction_ablation.png", dpi=150)
    (OUT / "direction_ablation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'direction_ablation.png'}")


if __name__ == "__main__":
    main()
