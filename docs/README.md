# docs

Pages and design documents for the project. GitHub Pages serves this folder at
<https://ismaelgserrano.github.io/QLMHackaton/>.

## Slides

| Page | Source | Published |
|---|---|---|
| **Laundering as directed motifs**: the 5-minute hackathon deck (reveal.js, no build step). Press `S` for speaker notes; add `?print-pdf` to the URL to export a PDF. | [`slides/index.html`](slides/index.html) | [GitHub Pages](https://ismaelgserrano.github.io/QLMHackaton/slides/) |

## HTML pages

| Page | Source | Published |
|---|---|---|
| **Which way does the money flow?**: explainer for ADR 0001. Why a symmetric edge gate (RZZ) cannot tell FAN-IN from FAN-OUT, and why the model uses a directed ZX gate. | [`edge-direction.html`](edge-direction.html) | [GitHub Pages](https://ismaelgserrano.github.io/QLMHackaton/edge-direction.html) |
| **AML Subgraph Circuit Pipeline**: visual walkthrough of the code, from the HI-Small CSV to subgraphs, circuits, training and the IQM Garnet run. Includes a live ZX vs RZZ circuit builder, gate and calibration explainers, the 8-qubit cap, and compute times. | [`pipeline.html`](pipeline.html) | [GitHub Pages](https://ismaelgserrano.github.io/QLMHackaton/pipeline.html) · [Claude artifact](https://claude.ai/artifact/5gc8uMNmfMXjS7GvwQM95T) (private, share it before sending) |

## Data description

| Document | What it covers |
|---|---|
| [`../reports/HI-Small/README.md`](../reports/HI-Small/README.md) | **HI-Small: dataset summary (from Neo4j).** Size and imbalance (5,078,345 transfers, 0.102 % laundering), graph schema, time and the temporal split, what distinguishes laundering, motif examples, what a quantum model sees (an ego-net), and what this means for the prototype. Raw numbers in [`stats.json`](../reports/HI-Small/stats.json), figures in [`figures/`](../reports/HI-Small/figures). Rebuild with `uv run python scripts/dataset_summary.py --name HI-Small`. |
| [`../reports/HI-Small/subgraphs_n8.json`](../reports/HI-Small/subgraphs_n8.json), [`subgraphs_n12.json`](../reports/HI-Small/subgraphs_n12.json) | The sampled subgraph datasets: positives, negatives, dropped attempts, cuts and the split per class. |

These files live outside `docs/`, so the links work on GitHub but not on the GitHub Pages site.

## Design documents

| Document | What it covers |
|---|---|
| [`prototype-plan.md`](prototype-plan.md) | Prototype plan: task, data, circuit, and work packages WP0–WP7. |
| [`adr/0001-directed-edge-gates.md`](adr/0001-directed-edge-gates.md) | ADR 0001: ZX edge gate, timestamp gate order, angle from amount. |

`.nojekyll` makes GitHub Pages serve the files as they are, without a Jekyll build.
