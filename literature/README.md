# Literature: IBM AML dataset × Quantum Machine Learning

An annotated reading list. Run `bash literature/download_papers.sh` to fetch the open-access
arXiv PDFs into `literature/pdfs/`. Git ignores that folder, so the PDFs stay out of the repository.

**Start here:** read [1] (the dataset), [2] (why graph structure matters), then [8]–[10] (quantum on graphs).

---

## Data links

| Resource | Link |
|---|---|
| Kaggle dataset (all 6 variants) | <https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml> |
| IBM Multi-GNN code (reference GNN baselines + preprocessing script `format_kaggle_files.py`) | <https://github.com/IBM/Multi-GNN> |
| Hackathon instructions | <https://github.com/Ilmosal/ExtremeData4QMLHackathon/blob/main/instructions.md> |
| Elliptic Bitcoin dataset (related real-world AML graph, for comparison) | <https://www.kaggle.com/datasets/ellipticco/elliptic-data-set> |

---

## A. The dataset

**[1] Altman, Blanuša, von Niederhäusern, Egressy, Anghel, Atasu — *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*.**
NeurIPS 2023 (Datasets & Benchmarks). [arXiv:2306.16424](https://arxiv.org/abs/2306.16424)
- The source paper for the dataset. Describes the AMLworld agent-based generator, how it was calibrated against real data, the HI/LI × Small/Medium/Large variants, and the laundering patterns (fan-in, fan-out, cycle, scatter-gather, ...).
- Gives GNN and gradient-boosting baselines with the **temporal train/val/test split** that we should reuse.

## B. Classical state of the art on this dataset

**[2] Egressy, von Niederhäusern, Blanuša, Altman, Wattenhofer, Atasu — *Provably Powerful Graph Neural Networks for Directed Multigraphs*.**
AAAI 2024. [arXiv:2306.11586](https://arxiv.org/abs/2306.11586) · [code](https://github.com/IBM/Multi-GNN)
- Adds port numbering, ego IDs, and reverse message passing to message-passing GNNs, and proves that the result can detect **any directed subgraph pattern**. Gives up to +30% minority-class F1 on the AML data.
- **Key for our argument:** it shows that the problem *is* subgraph/motif detection and that standard GNNs lack the needed expressivity. This is the gap a quantum model could aim at.

**[3] Blanuša et al. — *Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection*.**
ICAIF 2024. [arXiv:2402.08593](https://arxiv.org/abs/2402.08593)
- Mines laundering-like subgraph patterns (cycles, fan-in/out, scatter-gather) as transaction features, then uses gradient boosting. Beats GNNs on minority-class F1 and throughput.
- **Strong, cheap baseline:** hand-crafted motif features + XGBoost. It is also a natural way to compress features before a quantum encoding.

**[4] Weber et al. — *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics*.**
KDD 2019 Workshop. [arXiv:1908.02591](https://arxiv.org/abs/1908.02591)
- Introduces the Elliptic dataset (real Bitcoin graph). The classic reference for graph ML in AML. Useful for the "real-world relevance" part of the talk.

## C. Quantum ML for fraud / AML

**[5] Innan, Sawaika, Dhor, Dutta, Thota, Gokal, Patel, Khan, Theodonis, Bennai — *Financial Fraud Detection using Quantum Graph Neural Networks*.**
Quantum Machine Intelligence (2024). [arXiv:2309.01127](https://arxiv.org/abs/2309.01127)
- A QGNN with variational quantum circuits for fraud detection on graph-structured transaction data. The closest prior work to our setting.

**[6] Grossi, Ibrahim, Khamesi, ... (IBM) — *Mixed Quantum–Classical Method for Fraud Detection with Quantum Feature Selection*.**
IEEE Transactions on Quantum Engineering 3 (2022). [arXiv:2208.07963](https://arxiv.org/abs/2208.07963)
- QSVM on IBM Safer Payments card data, plus a quantum–classical ensemble. Reports fraud KPIs (recall, FPR), not accuracy. A good model for how to report results.

**[7] Innan, Khan, Bennai — *Financial Fraud Detection: A Comparative Study of Quantum Machine Learning Models*.**
[arXiv:2308.05237](https://arxiv.org/abs/2308.05237)
- Compares QSVM, VQC, QNN, and other models on a fraud dataset. A quick survey of what has been tried.

**[7b] D'Amico et al. — *Blockchain Network Analysis using Quantum Inspired Graph Neural Networks & Ensemble Models*.**
[arXiv:2508.09237](https://arxiv.org/abs/2508.09237) (2025)
- Quantum-*inspired* GNN (tensor/CP decomposition) + QBoost for AML node classification. Recent related work, but not a gate-based QML model.

## D. Quantum ML foundations relevant to our data properties

**[8] Verdon, McCourt, Luzhnica, Singh, Leichenauer, Hidary — *Quantum Graph Neural Networks*.**
[arXiv:1909.12264](https://arxiv.org/abs/1909.12264) (2019)
- The original QGNN / quantum graph convolution framework: Hamiltonians built from the graph's edges.

**[9] Skolik, Cattelan, Yarkoni, Bäck, Dunjko — *Equivariant Quantum Circuits for Learning on Weighted Graphs*.**
npj Quantum Information 9, 47 (2023). [arXiv:2205.06109](https://arxiv.org/abs/2205.06109)
- Permutation-equivariant circuits on weighted graphs. The main reference for a **symmetry-based inductive bias** argument on the transaction graph.

**[10] Huang, Broughton, Mohseni, Babbush, Boixo, Neven, McClean — *Power of Data in Quantum Machine Learning*.**
Nature Communications 12, 2631 (2021). [arXiv:2011.01938](https://arxiv.org/abs/2011.01938)
- Geometric difference between quantum and classical kernels: tells you *when* a quantum kernel can beat classical ones. Use it to make our advantage claim rigorous, or honestly limited.

**[11] Havlíček, Córcoles, Temme, Harrow, Kandala, Chow, Gambetta — *Supervised Learning with Quantum-Enhanced Feature Spaces*.**
Nature 567, 209–212 (2019). [arXiv:1804.11326](https://arxiv.org/abs/1804.11326)
- Quantum kernel / QSVM foundations.

**[12] Schuld, Killoran — *Quantum Machine Learning in Feature Hilbert Spaces*.**
Physical Review Letters 122, 040504 (2019). [arXiv:1803.07128](https://arxiv.org/abs/1803.07128)
- Frames quantum circuits as kernel methods. Background for kernel-alignment arguments.

## E. Real-world context

**[13] United Nations Office on Drugs and Crime (UNODC) — *Money laundering: overview*.**
<https://www.unodc.org/unodc/en/money-laundering/overview.html>
- Source of the estimate that 2–5 % of global GDP is laundered each year. Used on the "real world" slide.

---

## Possible angles for our argument (to discuss)

- **Motif detection as the signal ([2], [3]) → equivariant quantum circuits on small ego-subgraphs ([9], [8]).** Inductive bias matches the permutation symmetry of the subgraph.
- **Extreme imbalance → quantum one-class / anomaly-detection kernels ([6], [10]).** Train only on the normal class and argue with kernel geometry.
- **Directed multigraph structure.** Most QGNNs assume simple undirected graphs. Encoding direction, multiplicity, and time (in the spirit of [2]) in a circuit could be our novelty.

Add new papers in the same format: full citation, link, and 1–2 bullets on why it matters *for us*.
