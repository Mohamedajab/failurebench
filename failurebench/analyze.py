"""Summarize all runs, paired split-bootstrap intervals, and scientific figures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .run import ROOT, write_csv


METHOD_LABELS = {
    "clean_cp": "Clean CP", "random_mask_cp": "Random-mask CP",
    "pooled_augmentation": "Pooled augmentation", "mask_matched": "Mask-matched CP",
    "mask_envelope": "Mask envelope", "bonferroni": "Bonferroni CP",
    "rowmax": "Row-max CP", "scaled_rowmax": "Scaled row-max CP",
    "policy_matched_shift": "Policy-matched CP",
}


def read_metrics(path):
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    strings = {"dataset", "model", "training_arm", "family", "method", "policy"}
    for row in records:
        for key in row:
            if key not in strings:
                row[key] = float(row[key])
    return records


def validate_results(records, manifest):
    expected = manifest["completed_predictor_fits"] * len(manifest["config"]["family_sizes"]) * len(manifest["config"]["alphas"]) * 9 * 6
    assert len(records) == expected, (len(records), expected)
    index = {}
    for r in records:
        key = tuple(r[k] for k in ("dataset", "model", "training_arm", "seed", "family", "alpha", "method", "policy"))
        assert key not in index, f"duplicate metric cell {key}"
        index[key] = r
        assert 0 <= r["coverage"] <= 1
        assert abs(r["coverage"] - r["successes"] / r["n_test"]) < 1e-12
        assert r["coverage_wilson_lo"] - 1e-12 <= r["coverage"] <= r["coverage_wilson_hi"] + 1e-12
        assert r["mean_width"] >= 0
        assert 0 <= r["infinite_width_fraction"] <= 1
        if r["policy"] == "simultaneous":
            assert r["coverage"] <= r["worst_fixed_mask_coverage"] + 1e-12
    for key, r in index.items():
        if r["policy"] == "simultaneous":
            oracle = index[key[:-1] + ("oracle_worst",)]
            assert r["coverage"] == oracle["coverage"]
    return {"validated_metric_rows": len(records), "checks": [
        "expected complete factorial row count", "unique cell keys", "coverage bounds",
        "original-row successes/counts", "Wilson interval consistency", "nonnegative widths",
        "simultaneous no larger than worst fixed mask", "oracle equals simultaneous coverage"]}


def bootstrap_interval(values, indices):
    if np.any(np.isnan(values)):
        return float("nan"), float("nan")
    if np.all(np.isposinf(values)):
        return float("inf"), float("inf")
    if np.all(np.isneginf(values)):
        return float("-inf"), float("-inf")
    if np.any(~np.isfinite(values)):
        return float("nan"), float("nan")
    draws = values[indices].mean(axis=1)
    return tuple(float(v) for v in np.quantile(draws, [.025, .975]))


def aggregate(records, config):
    groups = defaultdict(list)
    keys = ["dataset", "model", "training_arm", "family", "alpha", "method", "policy"]
    for r in records:
        groups[tuple(r[k] for k in keys)].append(r)
    rng = np.random.default_rng(26851)
    n = len(config["seeds"])
    boot = rng.integers(n, size=(config["bootstrap_replicates"], n))
    output = []
    for key, group in groups.items():
        group.sort(key=lambda r: r["seed"])
        assert len(group) == n
        coverage = np.array([r["coverage"] for r in group])
        width = np.array([r["mean_width"] for r in group])
        clo, chi = bootstrap_interval(coverage, boot)
        wlo, whi = bootstrap_interval(width, boot)
        output.append(dict(zip(keys, key), seeds=n, mean_coverage=float(coverage.mean()),
            coverage_seed_sd=float(coverage.std(ddof=1)) if n > 1 else 0.,
            coverage_split_bootstrap_lo=clo, coverage_split_bootstrap_hi=chi,
            mean_width=float(width.mean()), width_split_bootstrap_lo=wlo, width_split_bootstrap_hi=whi,
            mean_worst_fixed_mask_coverage=float(np.mean([r["worst_fixed_mask_coverage"] for r in group])),
            mean_clean_coverage=float(np.mean([r["clean_mask_coverage"] for r in group])),
            infinite_width_fraction=float(np.mean([r["infinite_width_fraction"] for r in group])),
            mean_rmse=float(np.mean([r["rmse"] for r in group]))))
    # Paired differences: preserve dataset, split, model, and training-arm pairing.
    cells = defaultdict(dict)
    for r in records:
        if r["family"] == "all" and r["alpha"] == .1 and r["policy"] == "simultaneous":
            cells[(r["dataset"], r["model"], r["training_arm"], r["seed"])][r["method"]] = r
    comparisons = defaultdict(list)
    pairs = [("scaled_rowmax", "rowmax"), ("rowmax", "mask_matched"),
             ("rowmax", "mask_envelope"), ("rowmax", "random_mask_cp"),
             ("rowmax", "bonferroni"), ("policy_matched_shift", "rowmax")]
    for (dataset, model, arm, seed), methods in cells.items():
        for target, reference in pairs:
            a, b = methods[target], methods[reference]
            width_diff = a["mean_width"] - b["mean_width"]
            saving = ((b["mean_width"] - a["mean_width"]) / b["mean_width"]
                      if np.isfinite(a["mean_width"]) and np.isfinite(b["mean_width"]) and b["mean_width"] > 0 else np.nan)
            comparisons[(dataset, model, arm, target, reference)].append(
                (seed, a["coverage"] - b["coverage"], width_diff, saving))
    paired = []
    for key, values in comparisons.items():
        values.sort()
        a = np.array(values)
        clo, chi = bootstrap_interval(a[:, 1], boot)
        wlo, whi = bootstrap_interval(a[:, 2], boot)
        slo, shi = bootstrap_interval(a[:, 3], boot)
        paired.append(dict(zip(("dataset", "model", "training_arm", "target", "reference"), key),
            seeds=n, coverage_difference=float(a[:, 1].mean()), coverage_difference_lo=clo, coverage_difference_hi=chi,
            width_difference=float(a[:, 2].mean()), width_difference_lo=wlo, width_difference_hi=whi,
            relative_width_saving=float(a[:, 3].mean()), relative_width_saving_lo=slo, relative_width_saving_hi=shi))
    return output, paired


def figures(summary, config, out):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    datasets = config["datasets"]
    lookup = {(r["dataset"], r["model"], r["training_arm"], r["family"], r["alpha"], r["method"], r["policy"]): r for r in summary}
    get = lambda ds, method, policy="simultaneous", arm="clean", family="all": lookup[(ds, "histgb", arm, family, .1, method, policy)]
    labels = [s.replace("synthetic_", "syn. ").replace("_", " ") for s in datasets]
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(11, 5))
    fixed = np.array([get(ds, "mask_matched")["mean_worst_fixed_mask_coverage"] for ds in datasets])
    sim = np.array([get(ds, "mask_matched")["mean_coverage"] for ds in datasets])
    robust = np.array([get(ds, "rowmax")["mean_coverage"] for ds in datasets])
    ax.bar(x - .24, fixed, .24, label="Mask-matched: worst fixed mask", color="#7a93b7")
    ax.bar(x, sim, .24, label="Mask-matched: all masks simultaneously", color="#e19059")
    ax.bar(x + .24, robust, .24, label="Row-max: all masks simultaneously", color="#238b79")
    ax.axhline(.9, linestyle="--", color="#333333", label="90% target")
    ax.set(ylim=(0, 1.04), ylabel="Coverage (mean over 10 splits)", title="Fixed-mask coverage and simultaneous coverage are different targets")
    ax.set_xticks(x, labels)
    ax.legend(loc="lower left", fontsize=9)
    fig.text(.01, .005, "Histogram gradient boosting; clean training; all singleton feature losses. Benchmark-level descriptive means.", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(out / "endpoint_gap.png", dpi=180)
    fig.savefig(out / "endpoint_gap.svg")
    plt.close(fig)
    methods = ["mask_matched", "bonferroni", "rowmax", "scaled_rowmax", "policy_matched_shift"]
    colors = ["#e19059", "#9b73b0", "#238b79", "#396fba", "#777777"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
    for ax, ds in zip(axes.flat, datasets):
        for method, color in zip(methods, colors):
            r = get(ds, method)
            if np.isfinite(r["mean_width"]):
                ax.scatter(r["mean_width"], r["mean_coverage"], label=METHOD_LABELS[method], color=color, s=55)
            else:
                ax.text(.03, .05, "Bonferroni: infinite width", transform=ax.transAxes, color="#9b73b0", fontsize=8)
        ax.axhline(.9, color="#333333", linestyle="--", linewidth=1)
        ax.set_title(ds.replace("_", " "))
        ax.set_ylim(.3, 1.04)
        ax.set_xlabel("Mean width / training response SD")
    for ax in axes[:, 0]:
        ax.set_ylabel("Simultaneous coverage")
    for ax in axes.flat[len(datasets):]:
        ax.axis("off")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    # Include methods absent because their first-dataset width is infinite.
    if "Bonferroni CP" not in legend_labels:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], marker="o", linestyle="", color="#9b73b0"))
        legend_labels.append("Bonferroni CP")
    fig.legend(handles, legend_labels, loc="lower center", ncol=5)
    fig.suptitle("Simultaneous reliability has an interval-width cost", fontsize=15)
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(out / "coverage_width.png", dpi=180)
    fig.savefig(out / "coverage_width.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method, color in (("rowmax", "#238b79"), ("scaled_rowmax", "#396fba"), ("policy_matched_shift", "#777777")):
        certified = [get(ds, method)["mean_coverage"] for ds in datasets]
        unseen = [get(ds, method, "unseen_pair_shift")["mean_coverage"] for ds in datasets]
        axes[0].plot(x, certified, marker="o", label=METHOD_LABELS[method], color=color)
        axes[1].plot(x, unseen, marker="o", label=METHOD_LABELS[method], color=color)
    for ax in axes:
        ax.axhline(.9, color="#333333", linestyle="--", linewidth=1)
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.set_ylim(0, 1.04)
        ax.set_ylabel("Coverage (mean over splits)")
    axes[0].set_title("In-family simultaneous coverage")
    axes[1].set_title("Unseen two-feature policy: no certificate")
    axes[0].legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "failure_boundary.png", dpi=180)
    fig.savefig(out / "failure_boundary.svg")
    plt.close(fig)


def fmt_width(v):
    return "infinite" if not np.isfinite(v) else f"{v:.3f}"


def report(summary, paired, manifest, out):
    primary = [r for r in summary if r["model"] == "histgb" and r["training_arm"] == "clean" and r["family"] == "all" and r["alpha"] == .1 and r["policy"] == "simultaneous"]
    lookup = {(r["dataset"], r["method"]): r for r in primary}
    datasets = manifest["config"]["datasets"]
    nfits, nrows = manifest["completed_predictor_fits"], manifest["metric_rows"]
    lines = ["# Experiment results — research notes, not a paper", "",
        f"Completed {nfits} predictor fits and {nrows:,} metric cells across {len(datasets)} datasets, "
        f"{len(manifest['config']['seeds'])} prespecified splits, two model families, two training arms, three mask families, and two error levels.", "",
        "These are measured local results. Primary tables below use histogram gradient boosting with clean training, "
        "all single-feature losses plus the clean mask, and 90% target coverage. The complete factorial results are in summary.csv. "
        "No model or seed was selected using test performance.", "",
        "## Primary endpoint comparison", "",
        "Each entry averages the ten splits within a dataset. 'Worst fixed' is the smallest coverage among counterfactual fixed masks; "
        "'simultaneous' requires every mask to cover the same row. Widths are divided by the training response standard deviation.", "",
        "| Dataset | Mask-matched worst fixed | Mask-matched simultaneous | Row-max simultaneous | Scaled row-max simultaneous | Row-max width | Scaled width |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for ds in datasets:
        a, b, c = [lookup[(ds, m)] for m in ("mask_matched", "rowmax", "scaled_rowmax")]
        lines.append(f"| {ds} | {100*a['mean_worst_fixed_mask_coverage']:.1f}% | {100*a['mean_coverage']:.1f}% | {100*b['mean_coverage']:.1f}% | {100*c['mean_coverage']:.1f}% | {b['mean_width']:.3f} | {c['mean_width']:.3f} |")
    gaps = [lookup[(ds, "mask_matched")]["mean_worst_fixed_mask_coverage"] - lookup[(ds, "mask_matched")]["mean_coverage"] for ds in datasets]
    rowcov = [lookup[(ds, "rowmax")]["mean_coverage"] for ds in datasets]
    scalecov = [lookup[(ds, "scaled_rowmax")]["mean_coverage"] for ds in datasets]
    lines += ["", f"Across these seven primary cells, the descriptive mean fixed-versus-simultaneous gap for mask-matched CP is {100*np.mean(gaps):.1f} percentage points. "
        f"Row-max simultaneous coverage averages {100*np.mean(rowcov):.1f}%; scaled row-max averages {100*np.mean(scalecov):.1f}%. "
        "These equal-weight cross-dataset averages are descriptive, not population estimates or statistical significance claims.", "",
        "The reported worst-fixed value is a minimum over finite empirical mask coverages. Selecting that minimum introduces downward selection bias, "
        "so a value below 90% is not by itself evidence against the per-fixed-mask theorem. Simultaneous coverage measures a different event.", "",
        "![Endpoint gap](endpoint_gap.png)", "", "## Efficiency and paired uncertainty", "",
        "Positive saving means scaled row-max is narrower than unscaled row-max. Intervals resample the ten paired split seeds; "
        "they describe split stability conditional on these fixed datasets. Repeated splits overlap, so this is not an independent-population confidence statement.", "",
        "| Dataset | Relative width saving | Paired split-bootstrap 95% interval | Simultaneous coverage change |", "| --- | ---: | ---: | ---: |"]
    chosen = [r for r in paired if r["model"] == "histgb" and r["training_arm"] == "clean" and r["target"] == "scaled_rowmax" and r["reference"] == "rowmax"]
    for r in chosen:
        lines.append(f"| {r['dataset']} | {100*r['relative_width_saving']:.1f}% | [{100*r['relative_width_saving_lo']:.1f}%, {100*r['relative_width_saving_hi']:.1f}%] | {100*r['coverage_difference']:+.1f} pp |")
    positive = sum(r["relative_width_saving"] > 0 for r in chosen)
    lines += ["", f"Scaling improves mean width in {positive}/{len(chosen)} primary dataset cells. Its benefit is empirical and not universal; all losses remain reported.", "",
        "![Coverage and width](coverage_width.png)", "", "## Model and training ablations", "",
        "Each row averages datasets equally, after averaging the ten splits in each dataset. These are descriptive summaries; raw paired cells and uncertainty are in the CSV files.", "",
        "| Predictor | Training | Mask-matched simultaneous | Row-max simultaneous | Scaled simultaneous | Row-max width | Scaled width |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for model in manifest["config"]["models"]:
        for arm in manifest["config"]["training_arms"]:
            rows = [r for r in summary if r["model"] == model and r["training_arm"] == arm and r["family"] == "all" and r["alpha"] == .1 and r["policy"] == "simultaneous"]
            covs = [np.mean([r["mean_coverage"] for r in rows if r["method"] == m]) for m in ("mask_matched", "rowmax", "scaled_rowmax")]
            widths = [np.mean([r["mean_width"] for r in rows if r["method"] == m]) for m in ("rowmax", "scaled_rowmax")]
            lines.append(f"| {model} | {arm} | {100*covs[0]:.1f}% | {100*covs[1]:.1f}% | {100*covs[2]:.1f}% | {widths[0]:.3f} | {widths[1]:.3f} |")
    lines += ["", "Mask augmentation lowers robust interval width in these descriptive aggregates; it does not remove the difference between fixed-mask and simultaneous coverage.", "",
        "## Policy transfer and failure boundaries", "",
        "| Dataset | Policy-matched: matching shift policy | Policy-matched: saturation | Policy-matched: simultaneous | Row-max: unseen pair policy |", "| --- | ---: | ---: | ---: | ---: |"]
    filt = {(r["dataset"], r["method"], r["policy"]): r for r in summary if r["model"] == "histgb" and r["training_arm"] == "clean" and r["family"] == "all" and r["alpha"] == .1}
    for ds in datasets:
        values = [filt[(ds, "policy_matched_shift", p)]["mean_coverage"] for p in ("prediction_shift", "saturation", "simultaneous")]
        pair = filt[(ds, "rowmax", "unseen_pair_shift")]["mean_coverage"]
        lines.append(f"| {ds} | {100*values[0]:.1f}% | {100*values[1]:.1f}% | {100*values[2]:.1f}% | {100*pair:.1f}% |")
    lines += ["", "The pair extension is explicitly heuristic: singleton certificates do not certify pair deletion. The oracle uses outcomes only as an exhaustive stress-test endpoint. "
        "Prediction-shift and saturation selectors use full input values at the loss-generation stage; they are not detectors after the values disappear.", "",
        "![Failure boundary](failure_boundary.png)", "", "## What this supports", "",
        "An initial reproducible research project showing why fixed-mask, selected-policy, and simultaneous coverage require separate evaluation. "
        "The implementation applies established max-score conformal theory and evaluates an independent mask-scale ablation. It does not establish a novel algorithm or state-of-the-art performance. "
        "The modern weighted/ARC missingness methods and robust verification algorithms in literature.md have not been reproduced. Real failures and dataset-level shift are not measured.", "",
        "A publishable manuscript would require expert novelty review, stronger related-method reproductions, and a richer task setting. "
        "A sensible extension is dependent sensor-group deletion with a user-specified budget and a real telemetry task, then a learned input-dependent positive scale trained on an independent split. "
        "The frozen results here should remain as the initial benchmark; extensions need a new protocol and results directory.", "",
        "## Provenance and verification", "",
        f"Run elapsed time: {manifest['elapsed_seconds']:.1f} seconds. Machine/platform and complete pinned environment are recorded in manifest.json. "
        "Original archives, dataset hashes, duplicate removal, source hashes, raw predictions, split IDs, per-mask metrics, and runtime are saved. "
        "The test suite checks order-statistic calibration, infinity handling, split disjointness, frozen preprocessing, oracle equivalence, the exact counterexample, and a Monte Carlo rank check. "
        "validation.json records whole-run result consistency checks.", "",
        "Artifact paths: results/full/metrics.csv, summary.csv, paired_comparisons.csv, per_mask.csv, runtime.csv, manifest.json, "
        "validation.json, and predictions/*.npz."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "full")
    args = parser.parse_args()
    out = args.results.resolve()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    for name, sha in manifest["result_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == sha, f"Hash mismatch: {name}"
    records = read_metrics(out / "metrics.csv")
    validation = validate_results(records, manifest)
    summary, paired = aggregate(records, manifest["config"])
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired_comparisons.csv", paired)
    figures(summary, manifest["config"], out)
    report(summary, paired, manifest, out)
    validation.update(summary_rows=len(summary), paired_comparisons=len(paired), raw_result_hashes_verified=True)
    validation["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
