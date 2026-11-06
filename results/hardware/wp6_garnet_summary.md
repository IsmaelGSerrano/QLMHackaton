# WP6: first run on IQM Garnet (2026-11-06)

Issue #10. Full result: [`results/HI-Small/garnet_20261106-030338.json`](../HI-Small/garnet_20261106-030338.json)
(job ids, Resonance `info`, raw counts of every circuit, per-circuit layouts and scores).
Script: `scripts/run_garnet.py`.

## What ran

- **Model:** `qgnn-zx-p1-n8`, seed 0, trained in WP3 (`HI-Small/qgnn-zx-p1-n8_20261106-003450.json`). Inference only; 7 parameters bound before transpiling.
- **Circuits:** 60 HI-Small test subgraphs (≤ 8 accounts, 10 positive / 50 negative = the test split's 1:5 ratio, all embed in Garnet with 0 SWAPs) plus WP5's 20 graphs × 3 relabelled copies (same permutations and pinned layouts as `qgnn-zx-p1-n8-fakegarnet_20261106-020125.json`), plus 2 readout-calibration circuits per job.
- **Hardware:** IQM Garnet (`target=qpu`), 2,000 shots per circuit, 2 jobs (98 + 22 circuits, each with its own calibration), 92 s wall time, queue 0, 0.5 credits/s.
- **Same circuits in every column:** each circuit was transpiled once (optimization level 3, VF2 layout first) and the same physical circuit ran on IQMFakeGarnet and on Garnet, matched by qubit name.

## Result (60 test graphs, 2,000 shots, threshold 0.606 from the exact model's validation F1)

| mode | test PR-AUC (95% CI) | test F1 | R@1%FPR | mean \|score − exact\| | equiv. mean range | equiv. max range |
|---|---|---|---|---|---|---|
| exact (statevector) | 0.732 (0.481–0.952) | 0.667 | 0.500 | 0 | 0 | 0 |
| aer (noiseless, shots) | 0.733 (0.482–0.952) | 0.636 | 0.600 | 0.0039 | 0.0071 | 0.0169 |
| IQMFakeGarnet | 0.734 (0.485–0.952) | 0.667 | 0.500 | 0.0142 | 0.0066 | 0.0153 |
| **Garnet** | **0.713 (0.450–0.919)** | 0.667 | 0.500 | 0.0220 | 0.0242 | 0.0552 |
| IQMFakeGarnet, readout-mitigated | 0.734 (0.485–0.952) | 0.636 | 0.500 | 0.0079 | 0.0073 | 0.0166 |
| Garnet, readout-mitigated | 0.713 (0.450–0.919) | 0.700 | 0.500 | 0.0229 | 0.0246 | 0.0592 |

## Reading it

- **The model survives the hardware.** PR-AUC on Garnet is 0.713 against 0.732 exact on the same 60 graphs; the confidence intervals overlap almost completely (60 graphs, 10 positives: the CI is wide, so this is "no detectable loss", not "equal").
- **Equivariance violation is measurable on the device.** In exact simulation the 3 relabelled copies give identical scores (shared weights + timestamp gate order: Skolik et al. [9], ADR 0001). With shots alone (aer) the copies differ by 0.007 on average; that is the shot-noise floor. IQMFakeGarnet adds nothing visible beyond it (0.0066), but **Garnet spreads the copies 3.4× more (0.024 mean, 0.055 max)**: the same graph on other physical qubits scores differently. This is the device-noise part of the equivariance violation (issue #1, point 3).
- **IQMFakeGarnet underestimates the real device.** Mean |score − exact| is 0.014 on the fake and 0.022 on Garnet, and the fake model shows no layout dependence at all while Garnet does. Today's chip also differed from the fake's snapshot: QB14 was out of calibration.
- **Readout mitigation helps the simulator, not the chip.** On IQMFakeGarnet it halves the distance to exact (0.014 → 0.008), so about half of the noise model's error is readout. On Garnet it changes nothing (0.022 → 0.023), so the real error is dominated by gate noise and/or qubit-to-qubit variation that a per-qubit readout model does not capture. Kept as separate rows, never mixed with the raw numbers.
- F1 and R@1%FPR are not informative here: noise pulls every score toward logistic(β₀), the threshold is the exact model's, and with 50 negatives "R@1%FPR" means "recall with no false positive". Use PR-AUC and the equivariance spread.
- The subset favours small, simple graphs (0-SWAP graphs first), so compare the rows with each other, not with the full-test-split runs in the leaderboard.

## What broke on the way (and is now fixed in code)

1. **The live chip is not the fake chip.** On 2026-11-06 Garnet listed 19 qubits (QB14 out) and **renumbers** the rest, so a circuit transpiled for IQMFakeGarnet would have landed on the wrong qubits for every index ≥ 13. The script now maps qubits by name (`qlmaml.garnet.qubit_index_map`, `remap_qubits`); 35 of the 120 circuits used QB14 or a coupler that the live target lacks, so they were transpiled for the live chip and then mapped back, so that IQMFakeGarnet ran the same circuit. None used QB14.
2. **CZ direction.** Live targets list each coupler once, e.g. (3, 8) and not (8, 3). `unsupported_ops` now treats CZ as symmetric (before the fix, 88 of 96 mock circuits were needlessly re-transpiled).
3. **100-circuit limit.** Resonance declined the first submission (`Job circuit count exceeds the maximum allowed value: 100`); nothing ran. Batches are now split (`--max-circuits 100`), with calibration circuits in every job, and a declined job raises a clear error.
4. The HI-Small n8 cache was not on this machine; it was rebuilt from Neo4j (`scripts/build_subgraphs.py --max-nodes 8`, 607 s). Its summary is identical to `reports/HI-Small/subgraphs_n8.json`, and the script checks that the exact test scores match WP3's stored scores (it would abort otherwise).

## Not done

- One run, one seed, one calibration day. A repeat on another day would show how much of the layout spread is stable.
- No `garnet` row for `zx-p2` or `rzz-p1`; the script takes `--model`.
- The existing WP5 test `test_fake_garnet_noise_pulls_the_readout_off_the_exact_value` failed once in the full suite while the machine was under heavy load; it passed in every later run (alone and in the full suite).
