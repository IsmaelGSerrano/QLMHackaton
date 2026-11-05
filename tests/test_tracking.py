import json
from datetime import datetime, timezone

from qlmaml import tracking


def _metrics(f1, pr):
    return {"f1_at_threshold": f1, "pr_auc": pr, "roc_auc": 0.9, "recall_at_fpr_0.1pct": 0.4, "recall_at_fpr_1pct": 0.7}


def test_record_run_appends_and_ranks(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(tracking, "RUNS_FILE", tmp_path / "runs.jsonl")
    monkeypatch.setattr(tracking, "LEADERBOARD", tmp_path / "README.md")

    base = {"features": ["a", "b"], "val": _metrics(0.3, 0.3)}
    t0 = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)
    f1 = tracking.record_run({**base, "test": _metrics(0.40, 0.45)}, dataset="HI-Small", model="xgb", note="first", when=t0, commit="abc")
    tracking.record_run({**base, "test": _metrics(0.55, 0.50)}, dataset="HI-Small", model="qsvm", note="better", when=t0, commit="def")
    tracking.record_run({**base, "test": _metrics(0.99, 0.99)}, dataset="FAKE-Small", model="xgb", when=t0, commit="zzz")

    assert f1.exists() and json.loads(f1.read_text())["run_id"] == "xgb_20261105-120000"
    rows = tracking.load_runs()
    assert [r["model"] for r in rows] == ["xgb", "qsvm"]  # FAKE run excluded
    assert rows[0]["n_features"] == 2 and rows[0]["test_f1"] == 0.40

    board = (tmp_path / "README.md").read_text()
    assert "## HI-Small" in board and "FAKE" not in board
    assert board.index("qsvm") < board.index("| xgb")  # best test F1 first
