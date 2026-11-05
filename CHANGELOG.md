# Changelog

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
