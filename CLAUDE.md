# CLAUDE.md

Hackathon repository: quantum machine learning on the **IBM Transactions for Anti-Money Laundering (AML)** dataset,
for the [ExtremeData4QML Hackathon](https://github.com/Ilmosal/ExtremeData4QMLHackathon/blob/main/instructions.md).
See `README.md` for the dataset description and `literature/README.md` for the reading list.

## What the project must deliver

- A 5-minute slide presentation. Judged on: **theoretical grounding** > novelty > presentation quality.
- The argument must start from a **property of the data** (here: a directed temporal multigraph, extreme class imbalance, and laundering as subgraph motifs) and explain *mechanistically* why a quantum model helps with it (inductive bias, symmetry/equivariance, kernel alignment, sample complexity, qubit/memory efficiency).
- A small prototype on a **classical simulator** of quantum circuits. Beating classical baselines is NOT required. Optional extra: run the trained model on IQM Garnet (see "Quantum hardware" below).

## Rules for working in this repo

- **Never commit data.** Raw Kaggle files go to `data/raw/` and derived files to `data/processed/`. Both are git-ignored. The CSVs are GBs in size.
- **Never commit PDFs.** `literature/pdfs/` is git-ignored. Add papers as links in `literature/README.md`.
- **Default to HI-Small** for every experiment. Do not load Medium/Large without asking. Use chunked reads or `polars`/`pyarrow` for large CSVs.
- **Quantum prototypes must stay small** (about 4–16 qubits): sample subgraphs (ego-nets, k-hop neighborhoods around a transaction, or pattern instances from `*_Patterns.txt`) or compress features before encoding. Say clearly what was subsampled and why.
- **Metrics:** never report accuracy alone. Use minority-class F1, PR-AUC, and recall at a fixed FPR. Use a **temporal split** (train on earlier, test on later transactions), as in the Altman et al. and Multi-GNN papers, to avoid leakage.
- **Baselines:** compare against a cheap classical baseline (for example, gradient boosting on engineered graph features) so the quantum claim has context. Multi-GNN (`github.com/IBM/Multi-GNN`) is the reference GNN.
- **Data gotchas:** `*_Trans.csv` has two columns named `Account` (sender, receiver); `qlmaml.data.load_transactions` renames them. Bank ids have leading zeros, keep them as strings. On Windows set `PYTHONIOENCODING=utf-8` before printing polars tables, or the console (cp1252) raises `UnicodeEncodeError`.
- **HI-Small facts (verified 2026-11-05):** 370 laundering attempts in `HI-Small_Patterns.txt` covering 3,209 transactions over 8 motif types (FAN-OUT, FAN-IN, CYCLE, BIPARTITE, GATHER-SCATTER, SCATTER-GATHER, STACK, RANDOM); `qlmaml.data.load_patterns` parses it. Laundering transactions flagged in `*_Trans.csv` are more than those in the patterns file (patterns list only the structured attempts).

## Tooling (default choices; change them here if the team decides otherwise)

- **Python dependencies are managed with `uv` and `pyproject.toml` only.** Never use `pip install`, `python -m venv`, or a `requirements.txt`.
  - Add a library: `uv add <package>` (dev-only tools: `uv add --group dev <package>`).
  - Create/refresh the environment: `uv sync`. Commit `uv.lock`.
  - Run anything: `uv run python ...`, `uv run pytest`, `uv run jupyter lab`.
- Python 3.12 (pinned in `.python-version`; uv would otherwise pick the newest interpreter), polars for data, XGBoost for the classical baseline, PennyLane for circuits. Add PyTorch Geometric only if we actually train a GNN.
- IBM's `snapml` (Graph Feature Preprocessor) has no Windows/py3.12 wheel, so motif features live in `src/qlmaml/features.py`. Extend that file rather than adding a new feature module.
- Reusable code goes in `src/qlmaml/`, runnable experiments in `scripts/`, exploration in `notebooks/` (runnable top to bottom). Each experiment writes a JSON to `results/<dataset>/`.
- `uv run pytest` must pass before committing. `uv run python scripts/baseline.py --fake` is the end-to-end smoke test (synthetic data, never report its numbers).

## Quantum hardware: IQM Garnet on IQM Resonance

The team has access to real hardware through [IQM Resonance](https://resonance.iqm.tech/). The only device is **IQM Garnet**.

- **Hard limit: 20 qubits.** Garnet is a 20-qubit superconducting chip with a square-lattice (nearest-neighbor) coupling map. Native gates: `PRX` (single-qubit) and `CZ` (two-qubit, neighbors only). Two-qubit gates between non-neighbors need SWAPs, and SWAPs add noise.
- **Design for hardware, not just for the simulator.** Keep circuits at about 4–16 qubits and shallow (few `CZ` layers). Choose entangling patterns that fit the lattice. For graph encodings (one qubit per account in an ego-net, `CZ` on transaction edges), check that the subgraph embeds in the coupling map, or transpile and report the SWAP count.
- **Train on the simulator, evaluate on hardware.** Do the optimization loop on a classical simulator, preferably the noisy Garnet model (`IQMFakeGarnet` in `iqm-client[qiskit]`). Use Resonance for a small number of final runs: inference on the test split, a few kernel-matrix entries, or a noise comparison. Hardware runs cost quota and queue time. Batch many circuits into one job.
- **SDK:** build and train circuits in PennyLane (the project default). To run on Garnet, export the trained circuit (for example as OpenQASM) and submit it with Qiskit through `iqm-client[qiskit]` (`uv add "iqm-client[qiskit]"`; the old `qiskit-iqm` package is deprecated). Read the server URL from the Resonance dashboard.
- **Credentials:** the Resonance API token goes in the `IQM_TOKEN` environment variable. **Never commit the token** or write it into notebooks.
- Always report results from hardware next to results from noiseless and noisy simulation of the same circuit, with the shot count.

## Writing

- Every claim about quantum advantage in code comments, notebooks, or slides must name its mechanism and cite a source from `literature/`. Avoid vague "exponential speedup" statements. The judges penalize them.
