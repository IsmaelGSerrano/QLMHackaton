"""CRY vs ZX vs RZZ edge gate: simulator learning curves and the IQM Garnet runs, in one figure.

    uv run python scripts/plot_cry_comparison.py

Panels (all HI-Small, n <= 8 subgraph cache, same temporal split):
  A. Test PR-AUC vs training fraction, mean over 5 seeds; band = stratified bootstrap 95 % CI.
  B. Paired test PR-AUC difference (gate - RZZ) on the same resampled test subgraphs, 95 % CI.
  C. Seed-0 model on the 60 hardware test graphs: exact / IQMFakeGarnet / real Garnet, 95 % CI.
  D. Per-graph score on real Garnet vs the exact simulator (distance to the diagonal = hardware error).

Reads the newest 5-seed `qgnn-<model>-n8_*.json` and the newest `garnet_*.json` per model.
Writes results/figures/cry_comparison.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Entity colours match scripts/plot_learning_curve.py (ZX blue, RZZ aqua); CRY takes the next free slot
# (yellow). Validated with the dataviz validator: adjacent CVD dE >= 9; markers + direct labels as relief.
STYLE = {"zx-p1": ("#2a78d6", "o", "ZX"), "cry-p1": ("#eda100", "D", "CRY"), "rzz-p1": ("#1baf7a", "^", "RZZ")}


def latest(pattern: str, keep=lambda r: True) -> dict:
    for f in sorted((RESULTS / "HI-Small").glob(pattern), reverse=True):
        r = json.loads(f.read_text(encoding="utf-8"))
        if keep(r):
            return r
    raise SystemExit(f"no result for {pattern}")


def style_axes(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_facecolor(SURFACE)


def main() -> None:
    runs = {m: latest(f"qgnn-{m}-n8_*.json", lambda r: len(r.get("seeds", [])) > 1) for m in STYLE}
    hw = {m: latest("garnet_*.json", lambda r, m=m: r["config"]["model"] == f"qgnn-{m}-n8") for m in ("zx-p1", "cry-p1")}

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.2), facecolor=SURFACE)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # A. learning curve
    for m, (color, marker, label) in STYLE.items():
        lc = runs[m]["learning_curve"]
        x = np.array([row["n_train"] for row in lc])
        y = np.array([row["test_pr_auc_mean"] for row in lc])
        lo = np.array([row["test_pr_auc_ci"]["lo"] for row in lc])
        hi = np.array([row["test_pr_auc_ci"]["hi"] for row in lc])
        ax_a.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax_a.plot(x, y, color=color, marker=marker, markersize=8, linewidth=2, markeredgecolor=SURFACE,
                  markeredgewidth=1.5, label=label)
    base = runs["zx-p1"]["test"]["positives"] / runs["zx-p1"]["test"]["n"]
    ax_a.axhline(base, color=MUTED, linestyle="--", linewidth=1)
    ax_a.text(130, base + 0.012, f"random score ({base:.2f})", color=MUTED, fontsize=9)
    ax_a.set_xscale("log")
    ax_a.set_xticks([120, 299, 598, 1194], ["120\n(10 %)", "299\n(25 %)", "598\n(50 %)", "1,194\n(100 %)"])
    ax_a.minorticks_off()
    ax_a.set_xlim(100, 1700)
    ax_a.set_ylim(0, 1)
    ax_a.set_xlabel("training subgraphs (share of labels)")
    ax_a.set_ylabel("test PR-AUC (mean of 5 seeds)")
    ax_a.legend(loc="lower right", frameon=False)
    style_axes(ax_a, "A. Simulator: PR-AUC vs number of labels")

    # B. paired difference vs RZZ
    fractions = [row["fraction"] for row in runs["zx-p1"]["paired"]["test_pr_auc_diff"]]
    xs = np.arange(len(fractions))
    for k, m in enumerate(("zx-p1", "cry-p1")):
        color, marker, label = STYLE[m]
        rows = runs[m]["paired"]["test_pr_auc_diff"]
        d = np.array([r["diff"] for r in rows])
        err = np.array([[r["diff"] - r["lo"] for r in rows], [r["hi"] - r["diff"] for r in rows]])
        off = (k - 0.5) * 0.22
        ax_b.errorbar(xs + off, d, yerr=err, fmt=marker, color=color, markersize=8, capsize=4, linewidth=2,
                      markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{label} − RZZ")
        for xi, di, hi_ in zip(xs + off, d, d + err[1]):
            ax_b.annotate(f"{di:+.2f}", (xi, hi_), xytext=(0, 5), textcoords="offset points",
                          ha="center", va="bottom", fontsize=8.5, color=INK)
    ax_b.axhline(0, color=MUTED, linewidth=1)
    ax_b.set_ylim(-0.23, 0.22)
    ax_b.set_xticks(xs, [f"{f:.0%}" for f in fractions])
    ax_b.set_xlim(-0.5, len(fractions) - 0.5)
    ax_b.set_xlabel("share of training labels")
    ax_b.set_ylabel("test PR-AUC difference vs RZZ (95 % CI)")
    ax_b.legend(loc="lower right", frameon=False)
    style_axes(ax_b, "B. Simulator: gain over symmetric RZZ (paired)")

    # C. hardware modes
    modes = [("exact", "exact\nsimulator"), ("fakegarnet", "IQMFakeGarnet\n(noisy sim)"),
             ("hardware", "real Garnet"), ("hardware-mitigated", "real Garnet\n(readout mitig.)")]
    xs = np.arange(len(modes))
    width = 0.36
    for k, m in enumerate(("zx-p1", "cry-p1")):
        color, marker, label = STYLE[m]
        t = hw[m]["table"]
        y = np.array([t[mode]["test"]["pr_auc"] for mode, _ in modes])
        err = np.array([[y[i] - t[mode]["test_pr_auc_ci"]["lo"] for i, (mode, _) in enumerate(modes)],
                        [t[mode]["test_pr_auc_ci"]["hi"] - y[i] for i, (mode, _) in enumerate(modes)]])
        pos = xs + (k - 0.5) * (width + 0.03)
        ax_c.bar(pos, y, width, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        ax_c.errorbar(pos, y, yerr=err, fmt="none", ecolor=MUTED, capsize=3, linewidth=1)
        for xi, yi in zip(pos, y):
            ax_c.text(xi, 0.03, f"{yi:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax_c.set_xticks(xs, [name for _, name in modes])
    ax_c.set_ylim(0, 1.05)
    ax_c.set_ylabel("test PR-AUC, 60 graphs / 10 positive (95 % CI)")
    shots = hw["cry-p1"]["config"]["shots"]
    ax_c.legend(loc="upper right", frameon=False, ncols=2, bbox_to_anchor=(1, 1.02))
    style_axes(ax_c, f"C. Seed-0 model on IQM Garnet ({shots:,} shots per circuit)")

    # D. per-graph exact vs hardware score
    maddict = {}
    for m in ("zx-p1", "cry-p1"):
        color, marker, label = STYLE[m]
        c = [c for c in hw[m]["circuits"] if c["kind"] == "test"]
        ex = np.array([c_["scores"]["exact"] for c_ in c])
        hwv = np.array([c_["scores"]["hardware"] for c_ in c])
        pos = np.array([c_["label"] == 1 for c_ in c])
        mad = np.mean(np.abs(hwv - ex))
        ax_d.scatter(ex[~pos], hwv[~pos], s=40, marker=marker, facecolors="none", edgecolors=color, linewidths=1.4,
                     label=f"{label}, normal")
        maddict[label] = mad
        ax_d.scatter(ex[pos], hwv[pos], s=48, marker=marker, color=color, edgecolors=SURFACE, linewidths=1,
                     label=f"{label}, laundering")
    lims = [0, 1]
    ax_d.plot(lims, lims, color=MUTED, linewidth=1, linestyle="--", label="hardware = exact")
    ax_d.text(0.03, 0.97, "mean |Garnet − exact| per graph\n"
              + "\n".join(f"{k}: {v:.3f}" for k, v in maddict.items()),
              transform=ax_d.transAxes, va="top", fontsize=9, color=INK)
    ax_d.set_xlim(*lims)
    ax_d.set_ylim(*lims)
    ax_d.set_aspect("equal")
    ax_d.set_xlabel("score, exact simulator")
    ax_d.set_ylabel("score, real IQM Garnet")
    ax_d.legend(loc="lower right", frameon=False, fontsize=8.5)
    style_axes(ax_d, "D. Per-graph score: Garnet vs exact")

    fig.suptitle("Edge gate comparison on HI-Small (n ≤ 8 accounts): ZX vs CRY vs RZZ",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.948, "A–B: 5 seeds × 4 label fractions, noiseless simulator, 420 test subgraphs (70 laundering).\n"
             "C–D: seed-0 model, 60 test subgraphs (10 laundering), same graphs and Garnet qubits for both gates, "
             "2 CZ per edge for both.", fontsize=9, color=MUTED, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=2.5, w_pad=2.5)
    out = RESULTS / "figures" / "cry_comparison.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
