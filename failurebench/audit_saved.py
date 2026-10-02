"""Independently recompute robust endpoints from all saved predictions.

Uses sorted lists for ranks and does not call the benchmark's calibration code.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def independent_rank(scores, alpha):
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return float("inf") if rank > len(scores) else sorted(float(x) for x in scores)[rank - 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/full"))
    args = parser.parse_args()
    out = args.results.resolve()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    with (out / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    desired = {(r["dataset"], r["model"], r["training_arm"], int(r["seed"]), r["family"], float(r["alpha"]), r["method"]): r
               for r in metrics if r["policy"] == "simultaneous" and r["method"] in ("rowmax", "scaled_rowmax")}
    checked, files, hashes = 0, 0, {}
    for path in sorted((out / "predictions").glob("*.npz")):
        ds, model, arm, seed = path.stem.split("__")
        seed = int(seed)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as raw:
            splits = [raw[f"{role}_ids"] for role in ("train", "tune", "cal", "test")]
            flattened = np.concatenate(splits)
            assert len(np.unique(flattened)) == len(flattened), path
            assert np.array_equal(np.sort(flattened), np.arange(len(flattened))), path
            ranking = raw["feature_ranking"]
            for family in manifest["config"]["family_sizes"]:
                chosen = ranking if family == "all" else ranking[:min(int(family), len(ranking))]
                cols = [0] + (chosen + 1).tolist()
                tuning_errors = np.abs(raw["tune_y"][:, None] - raw["tune_pred"][:, cols])
                cal_errors = np.abs(raw["cal_y"][:, None] - raw["cal_pred"][:, cols])
                test_errors = np.abs(raw["test_y"][:, None] - raw["test_pred"][:, cols])
                scales = np.maximum(np.quantile(tuning_errors, manifest["config"]["scale_quantile"], axis=0), manifest["config"]["scale_floor"])
                for alpha in manifest["config"]["alphas"]:
                    for method in ("rowmax", "scaled_rowmax"):
                        scale = np.ones(len(cols)) if method == "rowmax" else scales
                        score = np.max(cal_errors / scale, axis=1)
                        radius = independent_rank(score, alpha) * scale
                        successes = int(np.count_nonzero(np.all(test_errors <= radius, axis=1)))
                        width = float(2 * radius.mean())
                        recorded = desired[(ds, model, arm, seed, str(family), alpha, method)]
                        assert int(recorded["successes"]) == successes, (path, family, alpha, method)
                        assert int(recorded["n_test"]) == len(splits[-1])
                        assert np.isclose(float(recorded["mean_width"]), width, atol=1e-12, rtol=1e-12)
                        checked += 1
        files += 1
    assert files == manifest["completed_predictor_fits"]
    assert checked == len(desired)
    report = {"prediction_files_audited": files, "robust_metric_cells_recomputed": checked,
              "original_row_partitions_disjoint_and_complete": True,
              "independent_sorted_rank_matches_saved_results": True,
              "prediction_sha256": hashes,
              "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "independent_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "prediction_sha256"}, indent=2))


if __name__ == "__main__":
    main()
