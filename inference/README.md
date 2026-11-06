# Inference image: XGBoost vs the quantum circuit, on one subgraph at a time

A small Docker image for the demo. Pick a transaction subgraph from the HI-Small **test split**
(later in time than all training data), score it with:

| engine | model | runs on |
|---|---|---|
| `xgboost` | XGBoost on 21 hand-made graph features (WP4) | this CPU |
| `exact` | trained circuit `qgnn-zx-p1-n8` (7 parameters, WP3), noiseless statevector | this CPU |
| `fake` | same circuit, `IQMFakeGarnet` noise model, shots | this CPU |
| `mock` | same circuit on `garnet:mock` (IQM Resonance) | IQM cloud, free, **random bits** |
| `qpu` | same circuit on **IQM Garnet** (20-qubit superconducting chip) | IQM cloud, costs credits |

and see the score, the decision at the validation threshold, and the response time split into
steps (`build_circuit`, `connect`, `transpile`, `submit`, `wait_result` = network + queue +
execution, `postprocess`). Hardware runs also show the Resonance job id, the chip qubit each
account was placed on (`account_to_qubit`), SWAPs, CZ count and the raw counts.

What is subsampled, and why (CLAUDE.md): only subgraphs with at most 8 accounts (the hardware
cache, so circuits stay shallow on Garnet), and only 24 of them (16 laundering attempts, 2 per
pattern type, plus 8 normal ego-nets; those that embed in Garnet with 0 SWAPs first). This is a
latency and connection demo, not an evaluation: the metrics are in `results/`.

## Build

From the repo root, with the main project environment:

```bash
uv run python inference/export_bundle.py
```

```bash
docker build -f inference/Dockerfile -t qlmaml-inference .
```

`export_bundle.py` needs `data/processed/subgraphs_HI-Small_n8.parquet` (`scripts/build_subgraphs.py`)
and a WP3 run in `results/`. It retrains XGBoost with the WP4 settings (seed 0, checks the test
PR-AUC against the recorded run), takes the circuit parameters from the newest WP3 run (no
retraining), and writes `inference/bundle/` (git-ignored: it holds dataset subgraphs).

## Run

Web page on http://localhost:8000 (local engines only, no token):

```bash
docker run --rm -p 8000:8000 qlmaml-inference
```

With the free `garnet:mock` (the token comes from the main checkout's git-ignored `.env`):

```bash
docker run --rm -p 8000:8000 --env-file .env qlmaml-inference
```

With the real Garnet (each click submits one job of 1 circuit; costs credits):

```bash
docker run --rm -p 8000:8000 --env-file .env -e ALLOW_QPU=1 qlmaml-inference
```

Terminal table instead of the page (each engine runs twice: cold, then warm):

```bash
docker run --rm --env-file .env qlmaml-inference compare ex12 --engines xgboost exact fake mock
```

```bash
docker run --rm qlmaml-inference list
```

API: `GET /api/summary`, `GET /api/examples/<id>`, `POST /api/infer {"example": "ex12", "engine": "fake", "shots": 1000}`.

## Reading the numbers

- Measured 2026-11-06 in the container on a laptop, `ex12` (SCATTER-GATHER), warm: XGBoost
  about 1 ms, exact circuit about 1 ms, IQMFakeGarnet about 0.5–0.7 s (1,000 shots),
  `garnet:mock` about 1.9 s (500 shots, mostly network). The first call of each engine also pays
  library imports (shown apart as `cold_start_import_ms`); the server warms up at start.
- `mock` returns random bitstrings, so its score is meaningless: it proves the submit path and
  measures the network round trip. Use `qpu` for real hardware scores; WP6 measured about 92 s
  wall time for a 2-job batch on Garnet (`results/HI-Small/garnet_20261106-030338.json`).
- Hardware scores are raw (no readout mitigation): noise pulls them towards logistic(beta0)
  (`results/hardware/wp6_garnet_summary.md`). Compare `exact`, `fake` and `qpu` on the same
  subgraph, as CLAUDE.md asks.
- XGBoost is much faster and, on this cache, more accurate (test PR-AUC 0.975 vs 0.664 for the
  exact circuit). The quantum claim of the project is about inductive bias and parameter count
  (7 shared parameters, permutation-equivariant; Skolik et al. [9] in `literature/README.md`),
  not speed.

## Files

- `app.py`: engines, timing, HTTP server (stdlib) and the `compare` / `list` CLI.
- `static/index.html`: the page.
- `export_bundle.py`: builds `bundle/` from the cache and `results/`.
- `pyproject.toml`, `uv.lock`: the image's own small dependency set (no PennyLane, polars or
  scikit-learn; `xgboost-cpu` without CUDA libraries). Versions match the main `uv.lock`.
- `Dockerfile`, `Dockerfile.dockerignore`: build context is the repo root, but only these files and
  6 modules of `src/qlmaml/` go into the image (about 1 GB, mostly Qiskit and the IQM client).
