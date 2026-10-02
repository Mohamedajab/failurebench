# FailureBench

**Research question:** when feature loss is selected per input, how do fixed-mask and simultaneous prediction-interval coverage differ, and what width cost buys protection?

This project contains working Python code, a frozen experiment protocol, public-source dataset provenance, tests, raw predictions, result tables, and figures. The manuscript has deliberately not been written.

Start with [experiment results](results/full/RESULTS.md), then [the protocol](protocol.md), [literature audit](literature.md), and [mathematical notes](theory_notes.md). The max-score construction uses established conformal theory. The proposed contribution is an evaluation artifact; novelty and publication suitability remain unverified.

[walkthrough.ipynb](walkthrough.ipynb) contains a short executed Python walkthrough of the actual run and one saved experiment cell. Open it in Jupyter or a notebook-capable editor; the command-line experiment does not need Jupyter.

## Reproduce

Python 3.11–3.13 recommended; the delivered run used Python 3.12.14. CPU execution is sufficient. No GPU, paid API, credentials, or cloud service is required.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m failurebench.counterexample
.\.venv\Scripts\python.exe -m failurebench.run --config config.json --output results\reproduction
.\.venv\Scripts\python.exe -m failurebench.analyze --results results\reproduction
.\.venv\Scripts\python.exe -m failurebench.audit_saved --results results\reproduction
```

The data loader downloads small original UCI archives on its first run, then uses the local cache. It never substitutes another dataset silently. Each run preserves source/config/data hashes and its installed package versions. Use a new output directory for each run; existing manifests are never overwritten by the runner.

Small end-to-end check:

```powershell
.\.venv\Scripts\python.exe -m failurebench.run --smoke --output results\my_smoke
```

The original workspace environment is outside this deliverable, at ../../work/.venv. It is deliberately excluded from the portable research package. No pretrained weights or GPU runtime are needed.

## What was run

* Five public tasks plus two controlled synthetic tasks.
* Ten prespecified split seeds, ridge and histogram gradient boosting, clean and mask-augmented training.
* Nine interval methods, three finite mask-family sizes, two target error rates, six policy/endpoints.
* Four disjoint data roles: predictor fit, scale fit, conformal calibration, evaluation.
* Exhaustive single-feature mask enumeration, a label-free policy audit, and explicitly uncertified two-feature transfer diagnostics.

The number of metric rows is not a count of independent experiments. Policies, methods, and masks reuse the same fitted models and original rows. Every raw prediction and split index is retained for audit.

## Files

| Path | Contents |
| --- | --- |
| failurebench/conformal.py | Corrected quantiles, nine calibration methods, label-free selection |
| failurebench/data.py | Public-source data acquisition, deduplication, deterministic simulations |
| failurebench/models.py | Disjoint splits and training-only predictor/preprocessing fits |
| failurebench/run.py | Paired factorial benchmark and provenance |
| failurebench/analyze.py | Full result checks, paired seed-bootstrap summaries, figures |
| failurebench/audit_saved.py | Independent sorted-rank recomputation from every saved predictor cell |
| failurebench/counterexample.py | Exact fixed-mask/adaptive-selection counterexample |
| tests/test_research_integrity.py | Quantile, leakage, ranking, counterexample, and rank-check tests |
| config.json | Frozen full-run configuration |
| results/full/manifest.json | Actual runtime, versions, dataset/source/result hashes |
| results/full/metrics.csv | Every dataset/model/arm/seed/family/alpha/method/policy metric |
| results/full/summary.csv | All split-averaged cells and descriptive split-bootstrap intervals |
| results/full/paired_comparisons.csv | Within-split differences and width savings |
| results/full/per_mask.csv | Full-family 90% fixed-mask diagnostics |
| results/full/predictions/ | Compressed normalized predictions, labels, preprocessing, and split IDs |
| results/full/validation.json | Whole-run completeness and numerical consistency checks |
| results/full/independent_audit.json | Independent recomputation of 3,360 robust endpoint cells and prediction hashes |
| results/full/RESULTS.md | Measured results and failure boundaries in research-note form |
| DATASETS.md | Credits, licenses, and dataset-specific limitations |
| interview_notes.md | Honest CV wording and points to understand before interviews |

Raw prediction files contain counterfactual complete-input/masked predictions for each fitted model, with explicit train/tune/cal/test IDs. Outcomes are standardized using training rows only. To recover original units, multiply predictions by response_std and add response_mean. Clean is column zero; singleton feature j is column j+1. pairs contains feature indices for the separately cached pair-loss predictions. Raw predictor objects are not serialized; the deterministic configuration and pinned environment support retraining.

## Interpretation

Simultaneous coverage is an original-row event: all masks must cover the same true outcome. It implies selected-policy coverage for every in-family selector under the stated exchangeability assumptions. It does not imply coverage conditional on a feature value or selected mask. The finite mask family is essential. Outside-family pair loss and source-distribution shift have no certificate in this experiment.

Repeated split seeds overlap within fixed benchmarks. Bootstrap intervals quantify split stability, not independent-population inference. Ordinal wine targets retain ties. Exact predictor duplicates are deterministically reduced before splitting, altering original dataset weights. Full-scale reproduction of recent weighted/ARC missingness methods and robust verification methods remains necessary before any state-of-the-art claim.

Code is MIT licensed. Dataset/paper/dependency rights remain with their sources. See LICENSE and DATASETS.md.
