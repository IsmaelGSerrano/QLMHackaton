# Prototype plan: directed equivariant circuit on transaction subgraphs

- **Status:** Draft
- **Date:** 2026-11-05
- **Builds on:** issue #1 (Option B), [ADR 0001](adr/0001-directed-edge-gates.md)
- **Target:** runs in noiseless simulation, in noisy simulation (`IQMFakeGarnet`), and on IQM Garnet through IQM Resonance.

## 1. Goal

Show with a small prototype that the model from ADR 0001 is feasible:

> one qubit per account, one ZX gate per (merged) transaction in timestamp order, shared weights, permutation-invariant readout

The prototype must show three things. Beating XGBoost is not one of them.

1. **Direction matters.** ZX separates motifs that `RZZ` cannot separate (FAN-IN vs FAN-OUT, CYCLE vs no-return triangle).
2. **The model learns from few labels.** It has about 6 trainable parameters, the same number for every graph size, because the weights are shared and equivariant (Skolik et al. [9]).
3. **The model runs on hardware.** The same circuits run on Garnet. We report the SWAP count, the noise, and the equivariance violation.

## 2. Task and data

**Task:** graph classification. The input is a small directed transaction subgraph. The output is a score for "contains a laundering attempt".

**Measured on HI-Small (`HI-Small_Patterns.txt`, 370 attempts):**

| Pattern | n | nodes min / median / max | tx median |
|---|---|---|---|
| BIPARTITE | 49 | 2 / 8 / 30 | 4 |
| CYCLE | 54 | 2 / 4 / 12 | 4 |
| FAN-IN | 40 | 2 / 9 / 17 | 8 |
| FAN-OUT | 48 | 2 / 8 / 17 | 7 |
| GATHER-SCATTER | 51 | 2 / 15 / 29 | 14 |
| RANDOM | 41 | 2 / 4 / 12 | 3 |
| SCATTER-GATHER | 44 | 2 / 9 / 18 | 14 |
| STACK | 43 | 2 / 15 / 45 | 10 |

Attempts with at most *k* accounts: k=4 → 151, k=8 → 196, k=12 → 258, k=16 → 303, k=20 → 324.

**Size decision (issue #1, point 4):**

- **Hardware track: `max_nodes = 8`.** 196 attempts (53 %) fit without cutting. Shallow, and most of them embed in the Garnet lattice.
- **Simulator track: `max_nodes = 12`.** 258 attempts (70 %) fit.
- Larger attempts are **cut in time**: keep the first accounts that the money reaches, in timestamp order, until the cap. We say this as subsampling on the slides.

**Positives and negatives (issue #1, point 2).** Both classes must come from the same extraction, or the model learns "clean motif vs. noisy ego-net" and not "laundering vs. normal":

- **Positive:** the accounts of an attempt. We take the **induced** subgraph on those accounts over the attempt's time window, so normal background transfers between them are included.
- **Negative:** a normal ego-net (`GraphDB.ego_subgraph`) whose center transaction is not laundering, from the same time period. It is **matched** on node count and edge count (the same histogram as the positives).
- **Ratio:** 1 : 5 positives to negatives. This keeps the imbalance real but trainable. Report the ratio on the slides.
- **Split:** temporal, by the start time of the subgraph: 60 / 20 / 20. Negatives follow the same time cut.

**Cache:** extract once from Neo4j into `data/processed/subgraphs_HI-Small_n{8,12}.parquet` (git-ignored). Training and hardware scripts then do not need Neo4j.

## 3. Circuit (as in ADR 0001)

Per subgraph with *n* accounts (qubits) and merged edges *E*, sorted by first timestamp (ties: by amount):

```
for layer in 1..p (p = 1 or 2):
    for each account i:          RY(a · f_i + b)          # node feature, shared a, b
    for (u -> v) in E, in time:  exp(-i θ_uv Z_u X_v)     # θ_uv = w · log(1 + amount) + v · log(count)
    for each account i:          RX(c)                    # shared mixer
readout:  s = mean_i <Z_i>   ->  logistic(α · s + β0)     # permutation-invariant
```

- Trainable parameters: `a, b, w, v, c` per layer, plus `α, β0`. That is 7 parameters for p = 1 and 12 for p = 2, the same for every graph size.
- **Pitfall:** a ZX gate on a payee in `|+>` (an eigenstate of X) only puts a phase on the payer. So we start in `|0>` and apply the RY node rotation first. A unit test checks that the gate entangles.
- Node feature `f_i`: start with `log(1 + total flow)`, which has no direction. In/out-degree features would give direction to `RZZ` too and spoil the ablation. Add them only after the ablation.
- Amounts are converted to USD before the log (ADR 0001 detail).
- **Garnet compile:** `ZX = H_v · RZZ · H_v` and `RZZ` = 2 `CZ`, so the cost is 2 `CZ` per merged edge.

**One circuit description, two backends.** We describe the circuit once, as a list of gates, in `src/qlmaml/circuit.py`. From this list we build:

- a **PennyLane** QNode for training (`default.qubit`, backprop, exact expectations),
- a **Qiskit** `QuantumCircuit` for Aer, `IQMFakeGarnet` and Garnet.

This is more robust than an OpenQASM export. A test checks that both backends give the same `<Z_i>`.

## 4. Work packages

Each package ends in code plus tests, and every experiment calls `record_run`.

### WP0. Environment (½ h)

- Done: `iqm-client[qiskit]` (brings Qiskit and Aer) and `requests` are dependencies. The organisers' picker is vendored as `src/qlmaml/qpu_pick.py`. `src/qlmaml/hardware.py` adds `select_backend(circuits, target="fake" | "mock" | "qpu")`, pinned to Garnet.
- Put `IQM_TOKEN` in `.env` (git-ignored), never in code. Check access read-only and for free: `uv run python -m qlmaml.qpu_pick --qubits 8`.
- Run one 2-qubit Bell circuit through `select_backend(..., target="fake")`, then `target="mock"` (free `garnet:mock`). This tests token, submission and reading counts back without spending credits. A real-Garnet job is only needed later, in WP6.
- Why pinned: Resonance lists several machines, and the organisers' picker takes the shortest queue. Our noise model (`IQMFakeGarnet`) and SWAP table assume Garnet, so hardware results must come from Garnet. If Garnet is not ready, we wait.

### WP1. Subgraph dataset (`src/qlmaml/subgraphs.py`, `scripts/build_subgraphs.py`)

- Add a public `GraphDB.induced_subgraph(account_ids, t0, t1)`. Today this is the private `_induced`.
- Positive extraction with the time cut at `max_nodes`. Negative extraction matched on (nodes, edges).
- Merge parallel edges with `Subgraph.simple_edges()`, extended with first timestamp, total USD amount and count.
- Write the parquet cache with columns: split, label, pattern, n_qubits, a node list, an edge list.
- Tests on `FAKE-Small` data: matched histograms, no future edges, temporal split is monotone.

### WP2. Circuit and the direction ablation (`src/qlmaml/circuit.py`, `tests/test_circuit.py`)

- Gate list builder, PennyLane backend and Qiskit backend. The edge gate is a switch: `"zx"` or `"rzz"`.
- Tests:
  - **Equivariance:** permute the account labels and check that the noiseless output is the same (|Δ| < 1e-10).
  - **Direction:** `"rzz"` gives the same output for FAN-IN and FAN-OUT, `"zx"` gives different outputs. The same holds for CYCLE vs the no-return triangle.
  - **Backend parity:** PennyLane and Qiskit statevector agree.
- This WP gives the first slide figure: the output distribution per motif, for ZX vs RZZ.

### WP3. Training (`scripts/train_qgnn.py`)

- Loss: weighted binary cross-entropy (positive weight = the negative/positive ratio). Adam, early stopping on validation PR-AUC.
- Runs to record: `qgnn-zx-p1`, `qgnn-zx-p2`, `qgnn-rzz-p1` (ablation), on n8 and n12.
- **Learning curve:** train on 10 %, 25 %, 50 % and 100 % of the training set. This tests the sample-complexity claim against XGBoost (WP4).
- Metrics from `qlmaml.metrics.evaluate`: minority F1, PR-AUC, recall at a fixed FPR. Never accuracy.
- Cost estimate: about 1,000 circuits of at most 12 qubits per epoch in statevector simulation. That is minutes on a laptop.

### WP4. Classical baseline on the same subgraphs (extend `src/qlmaml/features.py`)

- Add `subgraph_features(sg)`: node and edge counts, in/out-degree stats, reciprocity, 2- and 3-cycles, amount and count stats, time span.
- XGBoost on the same split, recorded as `xgb-subgraph`, with the same learning curve.
- Cite Egressy et al. [2] for why undirected message passing misses these motifs. A GNN is optional; Multi-GNN is the reference.

### WP5. Garnet fit and noisy simulation (`scripts/garnet_fit.py`, `scripts/eval_noisy.py`)

- Transpile every test circuit for `IQMFakeGarnet` (`optimization_level=3`, VF2 layout first). Per motif type, report: % that embed with 0 SWAPs, median SWAPs, median `CZ` count, depth. This is the "motif fit on Garnet" table from issue #1.
- Run the trained model on Aer in three modes: noiseless with shots, `IQMFakeGarnet` noise with shots, and exact. Use the same shot count (2,000) as hardware.
- **Equivariance violation:** for 20 test graphs, run 3 random relabellings each (these map to different physical qubits). Report the spread of the score: noiseless, noisy, and hardware (issue #1, point 3).

### WP6. Hardware run on Garnet (`scripts/run_garnet.py`)

Train in simulation and run only inference on hardware (CLAUDE.md).

- **Selection:** about 60 test subgraphs with at most 8 qubits, the true positive/negative ratio kept, preferring the ones with 0 SWAPs. Plus 20 × 3 relabelled copies for the equivariance check. That is about 120 circuits.
- **One batched job,** 2,000 shots per circuit, with the trained parameters bound in advance. Keep the job id and the raw counts in `results/HI-Small/garnet_<timestamp>.json`.
- `scripts/run_garnet.py --target fake|mock|qpu` calls `select_backend`. Everything else stays the same. Run it end to end with `fake`, then `mock`, then `qpu`. Store the returned `info` (machine, queue, price per second) with the results. Credits are charged per second, which is one more reason to batch.
- Report one table: the same circuits and shots for noiseless, `IQMFakeGarnet`, and Garnet. Show the metrics and the equivariance spread.
- Optional: readout-error mitigation, reported as a separate row and never mixed with the raw results.

### WP7. Slides (5 min)

Map to the challenge steps:

| Step | Content | Evidence |
|---|---|---|
| 1. Dataset | IBM AML HI-Small: a directed temporal multigraph, about 0.1 % laundering | dataset summary figures |
| 2. Challenge | Laundering = directed motifs; extreme imbalance; few labeled attempts (370) | motif table above |
| 3. Quantum argument | Circuit = the subgraph (Skolik [9]); ZX gate = direction; timestamp order = time; shared weights → equivariance → few parameters | WP2 ablation, WP3 learning curve |
| 4. Prototype | Simulator results + Garnet run | WP3–WP6 tables |
| 5. Real world | AML (UNODC estimate: 2–5 % of global GDP laundered per year), fraud rings, mule networks, any directed flow network (supply chains, payments) | short slide |

## 5. Order and scope

1. **Must have:** WP0, WP1 (n8), WP2, WP3 (zx vs rzz, n8), WP4, WP5 fit table.
2. **Should have:** learning curve, noisy simulation, WP6 hardware run.
3. **Nice to have:** n12 track, p = 2, readout mitigation, ego-net edge classification on the full imbalanced test split.

Do WP0's Bell test on `garnet:mock` first. If hardware access fails, we find out early, and WP5 alone still gives a credible hardware-aware story.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Model learns size, not motif | Negatives matched on node and edge counts. Report a "size-only" logistic baseline. |
| Fan-in/fan-out hubs with more than 4 neighbours need SWAPs | Cap at 8 nodes on hardware. Report the fit table. Pick 0-SWAP circuits for the hardware run and say so. |
| Noise on Garnet hides the signal | p = 1, merged edges, at most 8 qubits. Compare against `IQMFakeGarnet` beforehand. |
| Too few positives after the temporal split (about 40 in test) | Report confidence intervals (bootstrap), and do not over-claim. |
| Resonance quota or queue | One batched job. Keep the circuits ready in advance. |
