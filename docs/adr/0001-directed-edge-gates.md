# ADR 0001: Directed edge gates for the equivariant subgraph circuit

- **Status:** Accepted
- **Date:** 2026-11-05
- **Issue:** #1, point 1 ("Edge direction")
- **Explainer:** [`docs/edge-direction.html`](../edge-direction.html)

## Context

We build the quantum model as an equivariant circuit on small transaction subgraphs (Option B in issue #1, after Skolik et al. [9]).
Each account in the subgraph is one qubit. Each transaction is one two-qubit gate between the qubits of the payer and the payee.

The transaction graph is **directed**, and the direction carries the label. FAN-OUT (one account pays many) and FAN-IN (many accounts pay one) have the same undirected shape.
A CYCLE (A → B → C → A, the money comes back) and a triangle where the money does not come back (A → B → C and A → C) also have the same undirected shape.
Egressy et al. [2] show that a model must use edge direction to detect these patterns.

The standard edge gates in the literature are symmetric: `ZZ` couplings in Verdon et al. [8], `exp(-iγ w_ij Z_i Z_j)` in Skolik et al. [9], and `CZ` in graph states.
A symmetric gate is the same gate when payer and payee are swapped. So the circuit receives an undirected graph, and no training can separate the pairs above. The explainer page shows this with a state-vector simulation.

We had to choose three things:

1. The gate for one transaction.
2. The order in which the gates run.
3. How the gate angle is set, and how to handle several transactions between the same two accounts (the graph is a multigraph).

## Decision

1. **Gate: ZX interaction.** Each transaction applies `exp(-iθ Z_payer X_payee)`.
2. **Order: timestamp order.** Gates run in the order of the transaction timestamps.
3. **Angle: `θ = w · log(amount)`.** Repeated transactions between the same payer and payee are merged into one gate. `amount` is the total of the merged transactions. The **count** of merged transactions is kept as an extra feature. The trainable weight `w` is shared by all edges.

## Why

### ZX gate

- **It is directed.** The payer end gets `Z` and the payee end gets `X`. Swapping the two ends gives a different gate. In simulation, ZX gives different outputs for FAN-OUT vs FAN-IN and for CYCLE vs the no-return triangle, while `RZZ` gives identical outputs for both pairs at every angle.
- **It costs the same as `RZZ` on IQM Garnet.** `ZX = H_payee · ZZ · H_payee`, so it compiles to 2 `CZ` plus single-qubit gates, the same as the `RZZ` used by Skolik et al. Garnet's native gates are `PRX` and `CZ`.
- **It stays close to Skolik et al. [9].** It is the same construction with a basis change on the payee, so the equivariance argument and the weighted-angle idea carry over with few changes.
- **It is new for this problem.** We found no graph QML paper that uses an asymmetric edge gate for direction. This is our novelty point: direction, multiplicity and time inside the circuit (see "Possible angles" in `literature/README.md`).

### Timestamp order

- Symmetric `ZZ` gates commute, so Skolik et al. need no order. Directed gates do **not** commute when an account both receives and pays. That is the intermediate account in a chain or cycle, which is the typical laundering mule. In simulation, swapping the order of A → B and B → C changes the output.
- So we need a fixed order. The order must not depend on account names, or renaming accounts would change the answer and we would lose permutation equivariance.
- The timestamp is a property of the data, not of the labelling. Renaming accounts does not change it, so equivariance under relabelling is kept.
- It also puts **time** into the model: the circuit applies the transactions in the order the money moved.

### Angle from amount, merge repeats, count as a feature

- One shared weight `w` for every edge is required for equivariance: every edge must be treated by the same rule.
- `log(amount)` compresses amounts that span several orders of magnitude into a useful angle range.
- Merging repeats keeps the circuit shallow: one gate (2 `CZ`) per account pair instead of 2 `CZ` per transaction. Every extra `CZ` adds noise on hardware (CLAUDE.md: keep circuits shallow).
- The number of repeated payments is itself a laundering signal (for example structuring into many small payments), so we keep it as a feature instead of dropping it.

## Alternatives considered

| Option | Why not chosen |
|---|---|
| `CRY(θ)` (payer = control, payee = target) | Also directed and also about 2 `CZ`, and easy to explain. It moves further from Skolik's diagonal-coupling construction. Its effect is weak when the control qubit is mostly in `|0>`. |
| Keep `RZZ`, add in-degree and out-degree as single-qubit features | Cheap and keeps all gates commuting. But direction is then only a count per account, not a property of each edge, so it is a weaker argument. |
| Chiral term `exp(-iθ (XY − YX)/2)` (from chiral quantum walks) | Directed, with a clean physics citation (it breaks time-reversal symmetry). It needs more `CZ` on Garnet than ZX. It stays as a possible later comparison. |
| Order by account name or id | Breaks permutation equivariance. |
| One gate per transaction | Keeps every amount and timestamp, but `CZ` depth grows with the number of transactions and adds hardware noise. |

## Consequences

- The circuit layer is: single-qubit feature rotations, then one ZX gate per merged edge in timestamp order, then readout.
- **Equivariance is now "under relabelling of accounts, given the timestamps".** We must test it in noiseless simulation by permuting account labels and checking the output (issue #1, point 3).
- **Hardware fit:** ZX gates between non-neighbouring Garnet qubits still need SWAPs. The per-motif SWAP count table from issue #1 is still needed.
- **Details to settle in the implementation:**
  - Use `log(1 + amount)` so that small amounts do not give negative or very large angles.
  - Convert amounts to one currency before the log. `*_Trans.csv` has payments in several currencies.
  - A merged edge needs one timestamp for ordering. Use the time of the first transaction in the group.
  - Ties (same timestamp) need a rule that does not use account names, for example order by amount. Gates with a tie that share no account commute, so their order does not matter.
  - Decide how the count enters: a second shared weight in the same angle (`θ = w · log(1 + amount) + v · log(count)`) or a single-qubit feature on the payee. Start with the angle term.
- The classical baseline (XGBoost on `features.py` features, issue #1 point 5) must use the same subgraphs, so the comparison stays fair.

## References

Numbers refer to `literature/README.md`.

- [2] Egressy et al., *Provably Powerful Graph Neural Networks for Directed Multigraphs*, AAAI 2024. Direction is needed to detect these patterns.
- [8] Verdon et al., *Quantum Graph Neural Networks*, 2019. Symmetric `ZZ` edge Hamiltonians.
- [9] Skolik et al., *Equivariant Quantum Circuits for Learning on Weighted Graphs*, npj QI 2023. Weighted `ZZ` edge gates with shared parameters.
- Chiral quantum walks (Zimborás et al. 2013; Lu, Biamonte et al. 2016). Considered alternative, not yet in the reading list.
