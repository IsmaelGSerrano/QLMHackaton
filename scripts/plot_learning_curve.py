"""WP3/WP4 figure: test PR-AUC versus training fraction, one line per model, one panel per qubit cap.

    uv run python scripts/plot_learning_curve.py                 # latest multi-seed runs of HI-Small
    uv run python scripts/plot_learning_curve.py --name HI-Small --n 8 12

Reads the newest `results/<name>/qgnn-<model>-n<k>_*.json` that has a "seeds" field (written by
`scripts/train_qgnn.py --seeds ...`), and the same for the classical baseline
`xgb-subgraph-n<k>_*.json` (`scripts/train_xgb_subgraph.py --seeds ...`, drawn in black). The line is the mean over seeds; the band is the stratified
bootstrap 95 % CI of the seed-mean test PR-AUC (test-set resampling). The dashed gray line is the
test base rate (the PR-AUC of a random score). Writes results/figures/qgnn_learning_curve.png and a
Markdown table of the numbers, results/figures/qgnn_learning_curve.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CLASSICAL = "xgb-subgraph"  # WP4 baseline: ink, diamonds, so it reads as the reference and not a 4th quantum model
# Fixed categorical order (validated: adjacent CVD dE >= 9); markers and direct labels as second encoding.
STYLE = {"zx-p1": ("#2a78d6", "o"), "zx-p2": ("#eb6834", "s"), "rzz-p1": ("#1baf7a", "^"), CLASSICAL: (INK, "D")}
REFERENCE = "rzz-p1"


def latest_multiseed(name: str, model: str, k: int) -> dict | None:
    prefix = model if model == CLASSICAL else f"qgnn-{model}"
    files = sorted((RESULTS / name).glob(f"{prefix}-n{k}_*.json"))
    for f in reversed(files):
        run = json.loads(f.read_text(encoding="utf-8"))
        # Skip feature ablations (e.g. WP4 --exclude time_span_h): the figure shows the full model.
        if len(run.get("seeds", [])) > 1 and not run.get("config", {}).get("exclude"):
            return run | {"_file": f.relative_to(RESULTS).as_posix()}
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--n", type=int, nargs="+", default=[8, 12])
    args = ap.parse_args()

    runs = {(m, k): latest_multiseed(args.name, m, k) for k in args.n for m in STYLE}
    runs = {key: r for key, r in runs.items() if r}
    if not runs:
        raise SystemExit("no multi-seed qgnn runs found: run scripts/train_qgnn.py --seeds 0 1 2 3 4 first")

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, len(args.n), figsize=(5 * len(args.n), 4.6), sharey=True, squeeze=False)
    table = ["| n | model | fraction | n_train | test PR-AUC mean ± std (seeds) | 95 % CI (bootstrap) | "
             "Δ vs reference [95 % CI] | P(Δ ≤ 0) | per-seed Δ median | seeds better |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for ax, k in zip(axes[0], args.n):
        base_rate, ends = None, []
        for model, (color, marker) in STYLE.items():
            run = runs.get((model, k))
            if not run:
                continue
            curve = run["learning_curve"]
            x = [row["fraction"] for row in curve]
            y = [row["test_pr_auc_mean"] for row in curve]
            lo = [row["test_pr_auc_ci"]["lo"] for row in curve]
            hi = [row["test_pr_auc_ci"]["hi"] for row in curve]
            ax.fill_between(x, lo, hi, color=color, alpha=0.14, linewidth=0)
            ax.plot(x, y, color=color, marker=marker, markersize=6, linewidth=2, label=model)
            # Every seed as a faint marker (slight x offset per model), so outlier seeds stay visible.
            shift = {"zx-p1": 0.97, "zx-p2": 1.0, "rzz-p1": 1.03, CLASSICAL: 0.94}[model]
            for seed_curve in run["per_seed"].values():
                ax.scatter([r["fraction"] * shift for r in seed_curve], [r["test"]["pr_auc"] for r in seed_curve],
                           s=10, color=color, marker=marker, alpha=0.35, linewidths=0)
            ends.append([y[-1], model])
            split = run["data"]["splits"]["test"]
            base_rate = split["pos"] / split["n"]
            diffs = {d["fraction"]: d for d in run.get("paired", {}).get("test_pr_auc_diff", [])}
            ref_name = run.get("paired", {}).get("reference", "")  # e.g. "qgnn-rzz-p1-n8"
            ref_model = ref_name.removeprefix("qgnn-").removesuffix(f"-n{k}") or REFERENCE
            ref = runs.get((ref_model, k))
            for i, row in enumerate(curve):
                ci, d = row["test_pr_auc_ci"], diffs.get(row["fraction"])
                seed_txt = ""
                if d and ref:
                    common = sorted(set(run["per_seed"]) & set(ref["per_seed"]), key=int)
                    per_seed = np.array([run["per_seed"][s][i]["test"]["pr_auc"] - ref["per_seed"][s][i]["test"]["pr_auc"]
                                         for s in common])
                    seed_txt = f"{np.median(per_seed):+.3f} | {int((per_seed > 0).sum())}/{len(per_seed)}"
                dtxt = f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] vs {ref_model}" if d else "(reference)"
                ptxt = f"{d['p_a_not_better']:.2f}" if d else ""
                table.append(f"| {k} | {model} | {row['fraction']:.2f} | {row['n_train']} | "
                             f"{row['test_pr_auc_mean']:.3f} ± {row['test_pr_auc_std']:.3f} | "
                             f"{ci['lo']:.3f}–{ci['hi']:.3f} | {dtxt} | {ptxt} | {seed_txt or ' | '} |")
        # Direct labels at the line ends, pushed apart so they never overlap.
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.035)
        for y_label, model in ends:
            ax.annotate(model, (1.0, y_label), xytext=(9, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK)
        if base_rate is not None:
            ax.axhline(base_rate, color=MUTED, linestyle="--", linewidth=1)
            ax.annotate(f"random ({base_rate:.2f})", (0.1, base_rate), xytext=(0, 4), textcoords="offset points",
                        fontsize=8, color=MUTED)
        seeds = next((r["seeds"] for (m, kk), r in runs.items() if kk == k), [])
        ax.set_title(f"{args.name}, ego-nets ≤ {k} qubits ({len(seeds)} seeds)", fontsize=10, color=INK)
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.25, 0.5, 1.0], ["10 %", "25 %", "50 %", "100 %"])
        ax.minorticks_off()
        ax.set_xlim(0.085, 1.45)
        ax.set_xlabel("fraction of the training split")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel("test PR-AUC (mean over seeds)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9)
    fig.text(0.01, 0.01, "Line: mean over seeds; dots: single seeds. Band: stratified bootstrap 95 % CI of the seed-mean test PR-AUC\n"
             "(1,000 resamples of the test subgraphs). Quantum models: noiseless statevector. Temporal split.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "qgnn_learning_curve.png", dpi=150)

    sources = "\n".join(f"- `{r['_file']}`" for r in runs.values())
    (OUT / "qgnn_learning_curve.md").write_text(
        "# WP3 learning curve (generated by `scripts/plot_learning_curve.py`)\n\n" + "\n".join(table)
        + "\n\nΔ is the paired bootstrap of the seed-mean test PR-AUC difference on the same resampled test "
        "subgraphs; P(Δ ≤ 0) is the fraction of 1,000 draws where the model does not beat its reference (rzz-p1 for the "
        "quantum models, zx-p1 for xgb-subgraph). Both hold the trained "
        "models fixed (test-set uncertainty only). \"per-seed Δ median\" and \"seeds better\" compare seed k with seed k "
        "(same init seed, batch order and training subset): they show the training (seed) variation the bootstrap does not.\n\n"
        f"Sources:\n\n{sources}\n", encoding="utf-8")
    print(OUT / "qgnn_learning_curve.png")
    print("\n".join(table))


if __name__ == "__main__":
    main()
