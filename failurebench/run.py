"""Execute frozen benchmark: python -m failurebench.run --config config.json."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from .conformal import calibrate, independent_scales, prediction_shift_indices, wilson_interval
from .data import load_dataset
from .models import feature_ranking, fit_predictor, split_indices


ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, records):
    if not records:
        raise ValueError("cannot write empty result table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def source_hashes():
    files = sorted((ROOT / "failurebench").glob("*.py")) + [ROOT / "protocol.md", ROOT / "requirements.txt"]
    return {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def metric(base, policy, covered, selected_radius, selected_pred, y, response_std,
           worst_mask, clean_coverage, calibration_seconds):
    successes = int(np.count_nonzero(covered))
    lo, hi = wilson_interval(successes, len(covered))
    return dict(base, policy=policy, n_test=len(covered), successes=successes,
                coverage=float(covered.mean()), coverage_wilson_lo=lo, coverage_wilson_hi=hi,
                mean_width=float(2 * np.mean(selected_radius)),
                mean_width_original_units=float(2 * np.mean(selected_radius) * response_std),
                infinite_width_fraction=float(np.isinf(selected_radius).mean()),
                rmse=float(np.sqrt(np.mean((selected_pred - y) ** 2))),
                worst_fixed_mask_coverage=worst_mask, clean_mask_coverage=clean_coverage,
                calibration_batch_seconds=calibration_seconds)


def run_cell(x, y, features, dataset, model_name, arm, seed, config, out):
    start = time.perf_counter()
    train, tune, cal, test = split_indices(len(y), seed, config["split_fractions"])
    ranking = feature_ranking(x[train], y[train])
    predictor = fit_predictor(x[train], y[train], model_name, arm, seed)
    fit_seconds = time.perf_counter() - start
    d = x.shape[1]
    all_masks = np.concatenate((np.zeros((1, d), dtype=bool), np.eye(d, dtype=bool)))
    prediction_start = time.perf_counter()
    tune_pred = predictor.predict_masks(x[tune], all_masks)
    cal_pred = predictor.predict_masks(x[cal], all_masks)
    test_pred = predictor.predict_masks(x[test], all_masks)
    yn_tune, yn_cal, yn_test = [predictor.normalized_response(y[ids]) for ids in (tune, cal, test)]
    pairs = np.array(list(itertools.combinations(ranking[:min(4, d)], 2)), dtype=int)
    pair_masks = np.zeros((len(pairs), d), dtype=bool)
    for j, pair in enumerate(pairs):
        pair_masks[j, pair] = True
    pair_pred = predictor.predict_masks(x[test], pair_masks)
    pair_shift = np.abs(pair_pred - test_pred[:, :1]).argmax(axis=1)
    prediction_seconds = time.perf_counter() - prediction_start
    all_tune_res = np.abs(tune_pred - yn_tune[:, None])
    all_cal_res = np.abs(cal_pred - yn_cal[:, None])
    cell_name = f"{dataset}__{model_name}__{arm}__{seed}"
    np.savez_compressed(out / "predictions" / f"{cell_name}.npz",
        train_ids=train, tune_ids=tune, cal_ids=cal, test_ids=test, feature_ranking=ranking,
        tune_y=yn_tune, cal_y=yn_cal, test_y=yn_test, tune_pred=tune_pred,
        cal_pred=cal_pred, test_pred=test_pred, pair_pred=pair_pred, pairs=pairs,
        response_mean=predictor.response_mean, response_std=predictor.response_std,
        train_median=predictor.median, train_mean=predictor.mean, train_std=predictor.std)
    records, per_mask = [], []
    rows = np.arange(len(test))
    for family_setting in config["family_sizes"]:
        dropped = ranking if family_setting == "all" else ranking[:min(int(family_setting), d)]
        cols = np.concatenate(([0], dropped + 1))
        k = len(cols)
        pp = test_pred[:, cols]
        residuals = np.abs(pp - yn_test[:, None])
        cal_res = all_cal_res[:, cols]
        scales = independent_scales(all_tune_res[:, cols], config["scale_quantile"], config["scale_floor"])
        random_cal = np.random.default_rng(np.random.SeedSequence([seed, 129, k])).integers(k, size=len(cal))
        random_test = np.random.default_rng(np.random.SeedSequence([seed, 239, k])).integers(k, size=len(test))
        shift_cal = prediction_shift_indices(cal_pred[:, cols])
        shift_test = prediction_shift_indices(pp)
        saturation = np.abs(predictor.standardized(x[test])[:, dropped]).argmax(axis=1) + 1
        for alpha in config["alphas"]:
            calibration_start = time.perf_counter()
            radii = calibrate(cal_res, scales, alpha, random_cal, shift_cal)
            calibration_seconds = time.perf_counter() - calibration_start
            for method, radius in radii.items():
                coverage = residuals <= radius[None, :]
                fixed_cov = coverage.mean(axis=0)
                worst_mask, clean_cov = float(fixed_cov.min()), float(fixed_cov[0])
                base = dict(dataset=dataset, model=model_name, training_arm=arm, seed=seed,
                            family=str(family_setting), n_masks=k, alpha=alpha, method=method)
                for policy, choices in (("uniform", random_test), ("saturation", saturation),
                                        ("prediction_shift", shift_test)):
                    records.append(metric(base, policy, coverage[rows, choices], radius[choices],
                        pp[rows, choices], yn_test, predictor.response_std, worst_mask, clean_cov,
                        calibration_seconds))
                oracle_choices = (residuals / np.maximum(radius, 1e-12)[None, :]).argmax(axis=1)
                records.append(metric(base, "oracle_worst", coverage.all(axis=1), radius[oracle_choices],
                    pp[rows, oracle_choices], yn_test, predictor.response_std, worst_mask, clean_cov,
                    calibration_seconds))
                # RMSE for the simultaneous row is the uniform average squared error over all masks.
                records.append(metric(base, "simultaneous", coverage.all(axis=1),
                    np.broadcast_to(radius, pp.shape), pp.ravel(), np.repeat(yn_test, k),
                    predictor.response_std, worst_mask, clean_cov, calibration_seconds))
                records[-1]["rmse"] = float(np.sqrt(np.mean(residuals ** 2)))
                # Unseen pair extension is heuristic. Map missing features to in-family columns.
                local_index = {int(feature): j + 1 for j, feature in enumerate(dropped)}
                pair_radius = []
                for pair in pairs:
                    singleton_ids = [local_index[int(f)] for f in pair if int(f) in local_index]
                    pair_radius.append(max(radius[singleton_ids]) if len(singleton_ids) == 2 else max(radius))
                pair_radius = np.array(pair_radius)
                selected_pair_pred = pair_pred[rows, pair_shift]
                selected_pair_radius = pair_radius[pair_shift]
                records.append(metric(base, "unseen_pair_shift", np.abs(selected_pair_pred - yn_test) <= selected_pair_radius,
                    selected_pair_radius, selected_pair_pred, yn_test, predictor.response_std, worst_mask,
                    clean_cov, calibration_seconds))
                if family_setting == "all" and alpha == .1:
                    for j, feature in enumerate([-1] + dropped.tolist()):
                        per_mask.append(dict(base, feature_index=feature,
                            feature="clean" if feature == -1 else features[feature],
                            coverage=float(fixed_cov[j]), width=float(2 * radius[j]), n_test=len(test)))
    write_csv(out / "cells" / f"{cell_name}.csv", records)
    write_csv(out / "cells" / f"{cell_name}__masks.csv", per_mask)
    cell_metadata = dict(dataset=dataset, model=model_name, training_arm=arm, seed=seed,
                         n_train=len(train), n_tune=len(tune), n_cal=len(cal), n_test=len(test),
                         fit_seconds=fit_seconds, prediction_seconds=prediction_seconds,
                         total_seconds=time.perf_counter() - start,
                         dropped_feature_order=[features[j] for j in ranking])
    (out / "cells" / f"{cell_name}.json").write_text(json.dumps(cell_metadata, indent=2), encoding="utf-8")
    return records, per_mask, cell_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "full")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config.update(datasets=["diabetes", "synthetic_linear"], seeds=[101], synthetic_n=1000)
    out = args.output.resolve()
    if (out / "manifest.json").exists():
        raise SystemExit("Output already has a manifest. Use a new --output directory to preserve earlier runs.")
    for sub in ("cells", "predictions"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    manifest = dict(started_utc=started, status="running", config=config,
                    config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                    source_sha256=source_hashes(), python=sys.version, platform=platform.platform(),
                    numpy=np.__version__, sklearn=sklearn.__version__, smoke=args.smoke,
                    environment=subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    records, masks, metadata = [], [], []
    sources = []
    total = len(config["datasets"]) * len(config["models"]) * len(config["training_arms"]) * len(config["seeds"])
    completed = 0
    with threadpool_limits(limits=config["threads"]):
        for dataset in config["datasets"]:
            x, y, features, source = load_dataset(dataset, ROOT / "data", config["synthetic_n"])
            sources.append(source)
            print(f"DATA {dataset}: {len(y)} rows, {len(features)} features, removed {source['removed_duplicate_predictor_rows']} duplicate predictor rows", flush=True)
            for model, arm, seed in itertools.product(config["models"], config["training_arms"], config["seeds"]):
                cell, cell_masks, cell_meta = run_cell(x, y, features, dataset, model, arm, seed, config, out)
                records.extend(cell)
                masks.extend(cell_masks)
                metadata.append(cell_meta)
                completed += 1
                print(f"[{completed}/{total}] {dataset} {model} {arm} seed={seed}: {cell_meta['total_seconds']:.2f}s", flush=True)
    write_csv(out / "metrics.csv", records)
    write_csv(out / "per_mask.csv", masks)
    write_csv(out / "runtime.csv", metadata)
    manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter() - wall_start, completed_predictor_fits=completed,
                    metric_rows=len(records), datasets=sources)
    manifest["result_sha256"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in (out / "metrics.csv", out / "per_mask.csv", out / "runtime.csv")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"COMPLETE {completed} fits, {len(records)} metric rows, {manifest['elapsed_seconds']:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
