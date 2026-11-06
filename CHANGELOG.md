# Changelog

## [Unreleased]

### Added
- Add `scripts/train_qgnn.py`: train `qgnn-zx-p1`, `qgnn-zx-p2` and `qgnn-rzz-p1` on the n8 and n12 subgraph caches, one `record_run` per model and size (issue #7)
- Train with weighted binary cross-entropy (positive weight = negatives / positives), Adam on mini-batches, early stopping on validation PR-AUC
- Record a learning curve on 10 / 25 / 50 / 100 % of the training split, with test recall per motif type
- Add `qlmaml.qgnn`: `train`, `predict`, `graph_from_cache_row` and `nested_subsamples` (stratified, nested, seeded subsets that WP4 reuses)
- Add `qlmaml.circuit.statevector_mean_z`: exact numpy statevector with an adjoint gradient, about 10x faster than PennyLane for training, checked against PennyLane value and gradient
- Add `qlmaml.circuit.graph_from_merged` for edges that are already merged (the parquet cache)
- Add `qlmaml.features.subgraph_features` and `subgraph_feature_matrix`: directed graph features per cached subgraph (issue #8)
- Add `scripts/train_xgb_subgraph.py`: XGBoost on the WP1 subgraph cache with the WP3 learning curve, recorded as `xgb-subgraph-n<k>`
- Add WP4 HI-Small runs for n = 8 and n = 12, with and without the time-span feature
- Move `recall_by_pattern` to `qlmaml.metrics` and the `--fake` cache builder to `qlmaml.subgraphs.build_fake_cache`, shared by WP3 and WP4
- Add `--seeds` to `scripts/train_xgb_subgraph.py`: mean and std over seeds, bootstrap 95 % CI of test PR-AUC, paired bootstrap difference against WP3 `qgnn-zx-p1` (seed k against seed k)
- Move the per-seed learning-curve summary to `qlmaml.metrics.seed_curve_summary` and `paired_curves`, shared by WP3 and WP4
- Draw `xgb-subgraph` in `scripts/plot_learning_curve.py` as the classical reference line

## [0.0.1] — 2026-11-06

### Added
- Add `qlmaml.subgraphs`: labelled subgraph dataset for graph classification (issue #5)
- Build positives as induced subgraphs on an attempt's accounts, cut in time at `max_nodes`
- Match 5 laundering-free negatives per positive on account count (exact) and gate count (within 1), per temporal split
- Build "pairs" negatives from unrelated normal transfers for positives sparser than any connected ego-net
- Drop positives that get no 5 matches and list them by motif in the build summary
- Split 60 / 20 / 20 by subgraph start time, with cut points from the positives
- Add `FrameGraph`, an in-memory graph source with the same queries as `GraphDB`, for tests and `--fake`
- Add `scripts/build_subgraphs.py`, which writes `data/processed/subgraphs_<name>_n<k>.parquet` and a summary in `reports/<name>/`
- Add `GraphDB.induced_subgraph` (public), `attempts`, `normal_edge_ids`, `transfers_by_id`, `neighbor_transfer` and `usd_rates`
- Add first timestamp and USD total to `Subgraph.simple_edges`, ordered by time, then amount
- Add a BIPARTITE motif and an optional `*_Patterns.txt` output to the fake data generator
- Add the reveal.js slide deck `docs/slides/index.html` with the dataset section, for GitHub Pages
- Add HI-Small build summaries for n = 8 and n = 12 in `reports/HI-Small/`

### Changed
- Break ties in `GraphDB.ego_subgraph` neighbour order by account id, so extraction is deterministic
