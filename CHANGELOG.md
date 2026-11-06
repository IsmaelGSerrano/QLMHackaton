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
- Add `qlmaml.garnet` (issue #9): `transpile_report` (SWAPs inserted by routing, CZ count, depth, CZ depth, physical layout at optimization level 3 for Garnet), `mean_z_from_counts`, `sampled_mean_z`, `relabel`, `embedding_layouts` (SWAP-free placements on the coupling map), `fit_table` and `equivariance_spread`
- Add `scripts/garnet_fit.py`: the "motif fit on Garnet" table for every test circuit of the n8 and n12 caches, written to `results/hardware/garnet_fit_<name>_n<k>_<UTC>Z.json`
- Add `scripts/eval_noisy.py`: run the trained WP3 model exact, on noiseless Aer and on IQMFakeGarnet with 2,000 shots, one `record_run` per mode (`qgnn-<model>-n<k>-<mode>`), and the equivariance spread over 20 test graphs x 3 relabelled copies on different physical qubits
- Add WP5 HI-Small runs: Garnet fit for n8 and n12, noisy evaluation of `qgnn-zx-p1-n8` (seed 0)
- Add `scripts/run_garnet.py` (issue #10): run the trained `qgnn-zx-p1-n8` on IQM Garnet through `select_backend` (`--target fake|mock|qpu`, `--yes` required for `qpu`), next to exact, noiseless-shot and IQMFakeGarnet runs of the same circuits; 60 test graphs (class ratio kept, 0-SWAP graphs first) plus WP5's 20 x 3 equivariance copies, 2,000 shots, raw counts and job ids stored
- Add readout-error mitigation as separate rows: `qlmaml.garnet.readout_calibration_circuits`, `readout_errors`, `mitigated_mean_z` (independent per-qubit errors, calibration circuits in every job), `measured_qubits` (final qubit per clbit after routing)
- Match qubits to the live chip by name (`qubit_index_map`, `remap_qubits`): live Garnet leaves out qubits that are out of calibration and renumbers the rest; `unsupported_ops` treats CZ as symmetric (live targets list each coupler in one direction)
- Split hardware batches into jobs of at most 100 circuits (the Resonance limit); a declined job now raises instead of failing on an empty result
- Add `qlmaml.garnet.pick_hardware_graphs` and `newest_equivariance_copies`; move `pick_equivariance_graphs` from `scripts/eval_noisy.py` to `qlmaml.garnet`
- Add the first real-hardware run: `results/HI-Small/garnet_20261106-030338.json` (Garnet, 19 qubits, 2 jobs, 122 circuits x 2,000 shots)
- Complete the 5-minute deck `docs/slides/index.html` (issue #11): dataset, challenge (direction carries the label), circuit and its mechanisms, RZZ vs ZX ablation, learning curve, Garnet fit and hardware run, an honest summary, real-world use; detailed data slides moved to backup, with a leakage backup slide and timed speaker notes
- Add `tests/test_slides.py`: every local file the deck links must exist, and the evidence figures must be on a slide
- Restructure the deck `docs/slides/index.html` for clarity: data and findings (schema, payment format, amounts, laundering tail from `reports/HI-Small`), motifs and the option B decision (issue #1), graph-to-gate mapping with the gate table and why ZX instead of RZZ, the 9-stage pipeline, why the cap is 8 qubits (coverage, Garnet lattice, SWAP and CZ counts), and four "what we learnt" slides: gates, hardware shape, calibration, classical vs quantum accuracy and compute time
- Add UNODC as [13] in `literature/README.md` (source of the 2–5 % of global GDP estimate)

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
