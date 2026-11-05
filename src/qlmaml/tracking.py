"""Experiment tracking: every run appends one line to results/runs.jsonl and
writes its full result to results/<dataset>/<run_id>.json; results/README.md
is regenerated as a leaderboard from runs.jsonl.

Usage from a script:

    from qlmaml.tracking import record_run
    record_run(result, dataset="HI-Small", model="xgboost", note="baseline, 35 feats")

`result` must contain "val" and "test" dicts as produced by `qlmaml.metrics.evaluate`.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RUNS_FILE = RESULTS_DIR / "runs.jsonl"
LEADERBOARD = RESULTS_DIR / "README.md"

# Columns shown in the leaderboard, in order: (header, key in the run row).
LEADERBOARD_COLUMNS = [
    ("Date (UTC)", "date"),
    ("Dataset", "dataset"),
    ("Model", "model"),
    ("Test F1", "test_f1"),
    ("Test PR-AUC", "test_pr_auc"),
    ("Test R@0.1%FPR", "test_recall_fpr_0.1pct"),
    ("Val F1", "val_f1"),
    ("Val PR-AUC", "val_pr_auc"),
    ("#feat", "n_features"),
    ("Commit", "commit"),
    ("Note", "note"),
    ("File", "file"),
]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("+" if dirty else "")
    except Exception:  # not a git checkout
        return "unknown"


def record_run(
    result: dict,
    *,
    dataset: str,
    model: str,
    note: str = "",
    when: datetime | None = None,
    commit: str | None = None,
) -> Path:
    """Persist one experiment. Returns the path of the full-result JSON."""
    when = when or datetime.now(timezone.utc)
    run_id = f"{model}_{when:%Y%m%d-%H%M%S}"
    out_dir = RESULTS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{run_id}.json"

    full = {"run_id": run_id, "dataset": dataset, "model": model, "note": note,
            "date": when.isoformat(timespec="seconds"), "commit": commit or git_commit(), **result}
    out_file.write_text(json.dumps(full, indent=2, default=str), encoding="utf-8")

    if dataset.upper().startswith("FAKE"):
        # Synthetic smoke tests never enter the leaderboard (and results/FAKE-* is git-ignored).
        return out_file

    val, test = result["val"], result["test"]
    row = {
        "run_id": run_id,
        "date": f"{when:%Y-%m-%d %H:%M}",
        "dataset": dataset,
        "model": model,
        "note": note,
        "commit": full["commit"],
        "n_features": len(result.get("features", [])) or None,
        "test_f1": test.get("f1_at_threshold"),
        "test_pr_auc": test.get("pr_auc"),
        "test_roc_auc": test.get("roc_auc"),
        "test_recall_fpr_0.1pct": test.get("recall_at_fpr_0.1pct"),
        "test_recall_fpr_1pct": test.get("recall_at_fpr_1pct"),
        "val_f1": val.get("f1_at_threshold"),
        "val_pr_auc": val.get("pr_auc"),
        "file": out_file.relative_to(RESULTS_DIR).as_posix(),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RUNS_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    write_leaderboard()
    return out_file


def load_runs() -> list[dict]:
    if not RUNS_FILE.exists():
        return []
    return [json.loads(line) for line in RUNS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_leaderboard() -> Path:
    runs = load_runs()
    lines = [
        "# Results leaderboard",
        "",
        "One row per run, appended automatically by `qlmaml.tracking.record_run`. Do not edit by hand;",
        "the table is regenerated from `runs.jsonl` on every run (`uv run python -m qlmaml.tracking` rebuilds it).",
        "Test metrics use the threshold chosen on validation. Sorted by dataset, then test F1 (best first).",
        "",
    ]
    for dataset in sorted({r["dataset"] for r in runs}):
        rows = sorted((r for r in runs if r["dataset"] == dataset), key=lambda r: -(r.get("test_f1") or 0))
        lines += [f"## {dataset}", ""]
        lines.append("| " + " | ".join(h for h, _ in LEADERBOARD_COLUMNS) + " |")
        lines.append("|" + "---|" * len(LEADERBOARD_COLUMNS))
        for r in rows:
            cells = []
            for _, key in LEADERBOARD_COLUMNS:
                v = r.get(key)
                cells.append(f"[{v}]({v})" if key == "file" and v else _fmt(v))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    LEADERBOARD.write_text("\n".join(lines), encoding="utf-8")
    return LEADERBOARD


if __name__ == "__main__":
    print(write_leaderboard())
