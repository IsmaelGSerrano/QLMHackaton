# Results leaderboard

One row per run, appended automatically by `qlmaml.tracking.record_run`. Do not edit by hand;
the table is regenerated from `runs.jsonl` on every run (`uv run python -m qlmaml.tracking` rebuilds it).
Test metrics use the threshold chosen on validation. Sorted by dataset, then test F1 (best first).

## HI-Small

| Date (UTC) | Dataset | Model | Test F1 | Test PR-AUC | Test R@0.1%FPR | Val F1 | Val PR-AUC | #feat | Commit | Note | File |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-05 13:40 | HI-Small | xgboost | 0.525 | 0.539 | 0.543 | 0.436 | 0.399 | 35 | 8057210 | baseline: 35 activity/motif features (1d,7d windows, 2/3-cycles), 2000 rounds (cap hit), spw=100 | [HI-Small/xgboost_20261105-134024.json](HI-Small/xgboost_20261105-134024.json) |
