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

## Design documents

| Document | What it covers |
|---|---|
| [`prototype-plan.md`](prototype-plan.md) | Prototype plan: task, data, circuit, and work packages WP0–WP7. |
| [`adr/0001-directed-edge-gates.md`](adr/0001-directed-edge-gates.md) | ADR 0001: ZX edge gate, timestamp gate order, angle from amount. |

`.nojekyll` makes GitHub Pages serve the files as they are, without a Jekyll build.
