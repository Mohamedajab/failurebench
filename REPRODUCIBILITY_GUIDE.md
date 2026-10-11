# Reproduction and evidence audit guide

FailureBench is a small CPU-based research benchmark, not a deployed missing-data
prediction service. This guide separates **reproducing a new run** from
**checking the frozen results already committed**. For the prespecified design,
read [the protocol](protocol.md); for empirical findings, read
[the results](results/full/RESULTS.md).

## 1. Environment

The delivered experiment used Python 3.12.14; the README recommends Python
3.11–3.13. The dependency versions in `requirements.txt` are pinned. From the
repository root:

```bash
python -m venv .venv
# Activate the environment for your shell, or call its Python executable directly.
python -m pip install -r requirements.txt
python -m pytest -q
python -m failurebench.counterexample
```

Make sure `python` points at the environment just created. On Windows
PowerShell, the environment interpreter is `.venv\Scripts\python.exe`; on
Linux/macOS it is `.venv/bin/python`. The commands after activation are the same.

All models run on CPU. The first acquisition of the original UCI sources may
need internet access; later runs use a local dataset cache. Failed downloads
are not silently replaced with substitute data.

## 2. Smoke run, then full run

Pick a fresh output directory for every attempt. The runner rejects an output
that already contains `manifest.json`, preserving earlier evidence.

```bash
python -m failurebench.run --smoke --output results/my_smoke
python -m failurebench.run --config config.json --output results/my_full_run
python -m failurebench.analyze --results results/my_full_run
python -m failurebench.audit_saved --results results/my_full_run
```

The smoke option uses two tasks, one seed and a smaller synthetic data size; it
is a plumbing check, **not** a replacement for the frozen factorial study.
Use `--config config.json` for the full design.

`analyze` produces summary/paired-comparison artifacts and figures from the
run. `audit_saved` performs an independent sorted-rank recomputation of the
row-max endpoints and writes `independent_audit.json` into the selected
results directory. It does not substitute for the full test suite.

Do **not** use `results/full` as the output for a new run. That path contains
the delivered benchmark and its provenance; a reproduction belongs in its
own folder.

## 3. Read a saved run

| Artifact | What to check |
| --- | --- |
| `manifest.json` | Completion status, configuration, versions, source and result hashes, original dataset provenance |
| `metrics.csv` | Each method/policy evaluation row, including test successes and widths |
| `per_mask.csv` | Fixed-mask coverage and interval width for the all-feature, 90% setting |
| `summary.csv` | Split-averaged result cells |
| `paired_comparisons.csv` | Within-split width differences and descriptive bootstrap intervals |
| `runtime.csv` | Predictor-fit and prediction timings |
| `cells/` | Per-fit metric tables and metadata |
| `predictions/*.npz` | Original-row split IDs, calibration/test targets and counterfactual predictions |
| `validation.json` | Completeness and numerical consistency checks produced by analysis |
| `independent_audit.json` | Recomputed robust endpoints and prediction-file hashes |

The full run is **7 tasks, 10 seeds, 2 predictors, 2 training arms**:
280 model fits. The 90,720 metric rows are *not* independent experiment
replications: many methods and policies reuse the same rows and fits.
See [executed verification](verification.md) for the recorded local checks.

## 4. What the audit actually establishes

The independent checker in `failurebench/audit_saved.py`:

- reads each compressed prediction artifact without allowing pickle;
- verifies that train, scale-fit (`tune`), calibration and test partitions are
  disjoint and cover the original row IDs;
- independently computes sorted conformal order statistics for `rowmax` and
  `scaled_rowmax`, rather than calling the runner's calibration function;
- compares saved success counts and mean interval widths to recomputed values;
- checks the number of prediction files against the manifest.

The delivered verification record reports **280 prediction files** and
**3,360 robust endpoint cells** audited. This checks internal consistency,
not the truth of every research assumption or a claim of scientific novelty.

A fresh run may differ in wall-clock time and machine metadata. Compare design,
package versions, source/data provenance and substantive result tables before
ascribing numerical differences to the method. The reporting/audit scripts
were refined after the original raw run; `verification.md` explains this
source-hash distinction.

## 5. Interpreting the primary endpoint

- **Fixed-mask coverage** checks a prespecified missing-feature pattern.
- **Worst fixed-mask coverage** is the minimum *empirical* coverage across
  masks; it is not the same event as simultaneous coverage.
- **Simultaneous coverage** requires that, for a single original test row,
  **all** allowed masks produce an interval covering its true outcome.
- **Row-max calibration** calibrates the maximum residual across the finite
  mask family on each calibration row.
- **Scaled row-max** uses positive mask scales learned on a separate scale-fit
  subset before the row maximum is calibrated.

The original analysis describes a descriptive gain in simultaneous coverage
from **69.2% for mask-matched CP to 90.6% for row-max CP** in its primary
histogram-boosting/clean-training comparison. Width scaling helped in **5 of
7** primary tasks, but not all. These are results for the frozen benchmark,
not conditional coverage guarantees or claims about real sensor failures.

## 6. Boundaries a reproduction must preserve

- Use original-row splits; masked copies must not cross data roles.
- Do not fit preprocessing, feature ranking or scales using calibration/test
  outcomes in ways excluded by `protocol.md`.
- Keep finite mask-family definitions explicit. Single-feature certificates
  do **not** certify unseen pair losses.
- Treat the outcome-aware oracle as a diagnostic only, never a deployable
  policy; other stress-test selectors may use full inputs *before* deletion.
- Bootstrap intervals over overlapping split seeds reflect split stability
  within these benchmarks, not independent population sampling.
- Preserve failed comparisons and infinite-width cases rather than quietly
  filtering them out.

This repository does not establish a new conformal theorem, state-of-the-art
performance, clinical utility, or a publishable result. Any expanded benchmark
should use a newly specified protocol and output directory; keep the frozen
artifact intact.

