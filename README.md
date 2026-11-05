# QLMHackaton — Quantum ML for Anti-Money Laundering

Our team's repository for the **Extreme Data for Quantum Machine Learning Hackathon**
([challenge repo](https://github.com/Ilmosal/ExtremeData4QMLHackathon),
[instructions](https://github.com/Ilmosal/ExtremeData4QMLHackathon/blob/main/instructions.md)).
All code, notebooks, and deliverables we produce live here.

## The challenge, in short

The hackathon asks you to **start from a dataset with a hard property** and argue for a
quantum model that exploits it. Do not apply a generic variational circuit to a convenient dataset.

1. **Choose a dataset** with a hard property.
2. **Describe the challenge**: what makes it hard to learn from (graph structure, extreme imbalance, few samples, symmetries, ...).
3. **Argue for a quantum model or technique** that targets that property. A strong argument counts for more than novelty.
4. **Build a small prototype** with classical simulation of quantum circuits. A full benchmark is optional, and purely theoretical work is accepted.
5. **Connect to the real world**: where does data like this appear, and why does it matter?

You do **not** need to beat classical baselines. Mechanistic arguments win: sample complexity,
generalization bounds, kernel alignment, expressivity, qubit/memory efficiency, symmetry-based inductive bias.

**Deliverable:** slides and a **5-minute presentation**, followed by a short Q&A.

**Judging:** (1) theoretical grounding, (2) novelty and creativity, (3) quality of presentation.

## Our dataset: IBM Transactions for Anti-Money Laundering (AML)

- Kaggle: <https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml>
- Paper: Altman et al., *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*, NeurIPS 2023 Datasets & Benchmarks ([arXiv:2306.16424](https://arxiv.org/abs/2306.16424))

IBM's agent-based simulator (AMLworld) generated this synthetic data. Because the data is
simulated, **every laundering transaction has a ground-truth label**, which real bank data never has.
The simulator models the full laundering cycle (placement, layering, integration).

**Why it fits this hackathon:** the dataset is listed as *graph-structured with extreme class imbalance*.

| Property | What it means here |
|---|---|
| Directed temporal multigraph | Accounts are nodes and transactions are edges. Two accounts can have many parallel edges with different timestamps, amounts, and currencies. |
| Extreme class imbalance | Roughly 0.05–0.1% of transactions are laundering (about 1 in 1,000 for HI, about 1 in 2,000 for LI). Accuracy is meaningless, so use minority-class F1, PR-AUC, and recall at a fixed false-positive rate. |
| Laundering is a subgraph pattern | Laundering appears as motifs: fan-out, fan-in, cycle, scatter-gather, gather-scatter, bipartite, stack, random. Single transactions often look normal. Context in the graph is the signal. |
| Large scale | From about 5M edges (Small) to about 180M edges (Large). Quantum prototypes must work on sampled subgraphs or compressed features. |

### Dataset variants

Six datasets: `{HI, LI} × {Small, Medium, Large}`.

- **HI** = higher illicit ratio, **LI** = lower illicit ratio.
- Start with **HI-Small** (about 515K accounts, about 5M transactions).

### Files per variant

- `*_Trans.csv`: one row per transaction.
  Columns: `Timestamp, From Bank, Account, To Bank, Account, Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering`
- `*_Patterns.txt`: the laundering attempts, grouped by pattern type (useful for motif-level labels).
- `*_accounts.csv` (some variants): account to bank and entity mapping.

Note: the two `Account` columns have the same header. Rename them (for example `from_account`, `to_account`) on load.

### Getting the data

The data is **not** committed to git. Download it to `data/raw/`:

```bash
uv run kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -p data/raw --unzip
```

You need a Kaggle API token in `~/.kaggle/kaggle.json`. You can also download the files by hand from the Kaggle page.

## Repository layout

```
.
├── README.md            # This file
├── CLAUDE.md            # Guidance for Claude Code / AI assistants working in this repo
├── pyproject.toml       # Project metadata and dependencies (managed with uv)
├── uv.lock              # Locked dependency versions (commit it)
├── src/qlmaml/          # Reusable code
│   ├── data.py          #   load *_Trans.csv, node ids, temporal split, parse *_Patterns.txt
│   ├── features.py      #   per-transaction motif/activity features (fan-in/out, cycles, ...)
│   ├── metrics.py       #   PR-AUC, minority F1, recall at fixed FPR
│   └── fake_data.py     #   tiny synthetic file in the same format, for smoke tests only
├── scripts/baseline.py  # Classical baseline: features + XGBoost on a temporal split
├── tests/               # pytest
├── results/             # Experiment tracking: runs.jsonl (one row per run), README.md leaderboard,
│   └── <dataset>/       #   and the full JSON (+ log) of every run. Committed; FAKE-* runs are ignored.
├── literature/          # Key papers on the dataset, AML graph ML, and quantum ML (links + notes)
└── data/                # Local only, git-ignored (raw/ and processed/)
```

## Setup and running the baseline

We use [uv](https://docs.astral.sh/uv/) for everything Python. Install it once, then:

```bash
uv sync
```

That creates `.venv` from `pyproject.toml` + `uv.lock`. Add a dependency with `uv add <package>`;
never edit a requirements file or `pip install` by hand.

```bash
uv run pytest
```

Smoke test on synthetic data (no download needed; the numbers mean nothing):

```bash
uv run python scripts/baseline.py --fake
```

Real baseline on HI-Small (needs the Kaggle data in `data/raw/`):

```bash
uv run python scripts/baseline.py data/raw/HI-Small_Trans.csv
```

The script does a day-based 60/20/20 temporal split (same recipe as IBM's Multi-GNN), builds
per-transaction features from each account's recent activity (fan-in, fan-out, unique counterparties,
amounts, 2- and 3-cycles), trains XGBoost with early stopping on validation, picks the threshold on
validation, and reports PR-AUC, minority-class F1 and recall at 0.1% / 1% FPR on test.
Every run is appended to `results/runs.jsonl` and to the leaderboard in
[`results/README.md`](results/README.md); the full result goes to `results/<dataset>/<model>_<timestamp>.json`.
Pass `--note "what changed"` so the leaderboard row is meaningful. Any new experiment script should
call `qlmaml.tracking.record_run(...)` the same way.

### Baseline results so far

HI-Small, temporal split 64/19/17 by days (train 3.25M / val 966K / test 864K transactions),
35 features, XGBoost 2000 rounds (hit the cap; 35 min on CPU). Threshold chosen on validation.

| Metric (test) | Value |
|---|---|
| Minority-class F1 at val threshold | **0.525** |
| PR-AUC | 0.539 |
| ROC-AUC | 0.981 |
| Recall at 0.1% FPR | 0.543 |
| Recall at 1% FPR | 0.734 |

For reference, published minority-F1 on HI-Small: ~0.3 for a vanilla GIN, ~0.5–0.65 for Multi-GNN
with ports/ego-IDs/reverse MP ([2] in the literature list), ~0.6+ for Graph Feature Preprocessor + boosting ([3]).
Top features: payment format, same-currency flag, 7-day fan-out (unique receivers), same-bank flag,
1-day fan-out spread, 3-cycle membership. Note the illicit rate drifts upward over time
(train 0.08%, val 0.11%, test 0.19%), so test numbers are not directly comparable to validation.

## Graph database (Neo4j)

The transactions also load into a local Neo4j (Docker) as a property graph, for exploring
motifs visually and for extracting small subgraphs for quantum encodings. The model is in
`src/qlmaml/graphdb.py`:

- `(:Account)-[:TRANSFER]->(:Account)`: one relationship per transaction (keeps the multigraph), with label, split and pattern id.
- `(:Account)-[:SENDS_TO]->(:Account)`: parallel edges collapsed per account pair and temporal split (the simple graph, one qubit per account).
- `(:Account)-[:AT_BANK]->(:Bank)`, and `(:LaunderingAttempt {pattern})-[:INVOLVES]->(:Account)` from `*_Patterns.txt`.

```bash
export NEO4J_PASSWORD=<any local password>
uv run python scripts/neo4j_load.py data/raw/HI-Small_Trans.csv --run
```

Then open <http://localhost:7474> and try `cypher/aml_queries.cypher`. From Python,
`GraphDB().ego_subgraph(edge_id, hops=1, max_nodes=12)` returns a leakage-free ego-net
(only transfers in the 24 h before the transaction) with nodes relabelled to qubit indices.

## Literature

See [`literature/README.md`](literature/README.md) for an annotated reading list:
the dataset paper, classical state of the art on this dataset (Multi-GNN, Graph Feature Preprocessor),
and quantum ML work on graphs, fraud, and imbalanced data.
`literature/download_papers.sh` fetches the open-access PDFs to a local folder that git ignores.
