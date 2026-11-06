# Results leaderboard

One row per run, appended automatically by `qlmaml.tracking.record_run`. Do not edit by hand;
the table is regenerated from `runs.jsonl` on every run (`uv run python -m qlmaml.tracking` rebuilds it).
Test metrics use the threshold chosen on validation. Sorted by dataset, then test F1 (best first).

## HI-Small

| Date (UTC) | Dataset | Model | Test F1 | Test PR-AUC | Test R@0.1%FPR | Val F1 | Val PR-AUC | #feat | Commit | Note | File |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-06 00:08 | HI-Small | qgnn-zx-p1-n12 | 0.724 | 0.670 | 0.044 | 0.713 | 0.676 |  | babeaa7+ | WP3 first runs: weighted BCE, Adam lr 0.05, early stop on val PR-AUC, learning curve 10/25/50/100% | [HI-Small/qgnn-zx-p1-n12_20261106-000810.json](HI-Small/qgnn-zx-p1-n12_20261106-000810.json) |
| 2026-11-06 00:02 | HI-Small | qgnn-rzz-p1-n8 | 0.677 | 0.671 | 0.014 | 0.661 | 0.735 |  | babeaa7+ | WP3 first runs: weighted BCE, Adam lr 0.05, early stop on val PR-AUC, learning curve 10/25/50/100% | [HI-Small/qgnn-rzz-p1-n8_20261106-000257.json](HI-Small/qgnn-rzz-p1-n8_20261106-000257.json) |
| 2026-11-06 00:00 | HI-Small | qgnn-zx-p1-n8 | 0.656 | 0.681 | 0.071 | 0.702 | 0.710 |  | babeaa7+ | WP3 first runs: weighted BCE, Adam lr 0.05, early stop on val PR-AUC, learning curve 10/25/50/100% | [HI-Small/qgnn-zx-p1-n8_20261106-000005.json](HI-Small/qgnn-zx-p1-n8_20261106-000005.json) |
| 2026-11-06 00:13 | HI-Small | qgnn-rzz-p1-n12 | 0.652 | 0.695 | 0.044 | 0.693 | 0.698 |  | babeaa7+ | WP3 first runs: weighted BCE, Adam lr 0.05, early stop on val PR-AUC, learning curve 10/25/50/100% | [HI-Small/qgnn-rzz-p1-n12_20261106-001349.json](HI-Small/qgnn-rzz-p1-n12_20261106-001349.json) |
| 2026-11-06 00:01 | HI-Small | qgnn-zx-p2-n8 | 0.647 | 0.660 | 0.071 | 0.729 | 0.749 |  | babeaa7+ | WP3 first runs: weighted BCE, Adam lr 0.05, early stop on val PR-AUC, learning curve 10/25/50/100% | [HI-Small/qgnn-zx-p2-n8_20261106-000135.json](HI-Small/qgnn-zx-p2-n8_20261106-000135.json) |
| 2026-11-05 13:40 | HI-Small | xgboost | 0.525 | 0.539 | 0.543 | 0.436 | 0.399 | 35 | 8057210 | baseline: 35 activity/motif features (1d,7d windows, 2/3-cycles), 2000 rounds (cap hit), spw=100 | [HI-Small/xgboost_20261105-134024.json](HI-Small/xgboost_20261105-134024.json) |
| 2026-11-06 00:12 | HI-Small | qgnn-zx-p2-n12 | 0.522 | 0.545 | 0.029 | 0.595 | 0.616 |  | babeaa7+ | WP3 first runs: weighted BCE, Adam lr 0.05, early stop on val PR-AUC, learning curve 10/25/50/100% | [HI-Small/qgnn-zx-p2-n12_20261106-001210.json](HI-Small/qgnn-zx-p2-n12_20261106-001210.json) |
