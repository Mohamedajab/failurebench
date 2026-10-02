# CV and interview preparation

This is a research project, not a published paper. Use it under **Research Projects** until a manuscript actually exists, and use **Preprints** only after a real preprint has been posted. An AI-assisted experiment is not evidence that you can independently explain its theory and code; understanding and extending it is the next step.

## Accurate CV wording after reproducing and understanding the work

**FailureBench — uncertainty under input-dependent feature loss (2026).** Developed an AI-assisted reproducible conformal-prediction audit across five public regression tasks and two synthetic tasks, with two predictor families, ten prespecified splits, nine calibration baselines, and disjoint training/scale/calibration/test partitions. Investigated the gap between fixed-mask and simultaneous coverage and measured the interval-width cost of robustness.

Adapt “developed” to your actual involvement. Do not claim independent authorship of all code or a novel robust-conformal theorem. The repository and reproducible findings can support an interview discussion once you understand them. Do not list the 90,720 dependent metric rows as 90,720 independent experiments.

## What to be able to explain without notes

1. Why a 90% coverage claim averages over calibration and future examples, and is not a 90% guarantee at every input.
2. Why fixed-mask guarantees do not survive arbitrary per-input selection. Recreate the two-disjoint-failure-region counterexample.
3. Why applying split conformal to one maximum score per original row gives simultaneous finite-family coverage.
4. Why a scale must be fitted independently of calibration labels for the stated proof, and why a positive floor is needed.
5. Why artificially masking each calibration row many times does not create many independent calibration units.
6. Why Bonferroni may give infinite intervals with small calibration sets, and how the implementation handles that honestly.
7. Which policies use true labels (the oracle diagnostic), which use complete input values during loss generation, and what is available at interval inference time.
8. Why narrower intervals matter only alongside the matching coverage endpoint. A narrower method with lower simultaneous coverage is not automatically better.
9. Why the experiment has no guarantee for unseen pair deletions or source-distribution shift.
10. What the most relevant existing methods already do, and why this project's novelty is provisional.

## Work that would strengthen your personal contribution

First reproduce the delivered experiment and review one raw NPZ file against its CSV metrics. Then implement a new feature-group loss setting or a real telemetry task under a new frozen protocol. Compare a learned input-dependent scale against the existing mask-only scale without using calibration outcomes to choose it. Reproduce the strongest relevant missingness method before claiming a methods contribution.

Keep a short log of decisions, bugs you found, negative results, and what you changed. That gives you concrete evidence of research judgment. A supervisor or collaborator can assess whether the resulting extension warrants a paper.
