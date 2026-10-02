"""Conformal primitives. Calibration units are original rows, not mask copies."""
from __future__ import annotations

import math
import numpy as np


METHODS = (
    "clean_cp", "random_mask_cp", "pooled_augmentation", "mask_matched",
    "mask_envelope", "bonferroni", "rowmax", "scaled_rowmax", "policy_matched_shift",
)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Corrected order statistic, including infinity for insufficient n."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or np.any(~np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("scores must be a finite, nonnegative 1D array")
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


def independent_scales(residuals: np.ndarray, quantile: float = .75,
                       floor: float = .05) -> np.ndarray:
    if residuals.ndim != 2 or residuals.shape[0] == 0:
        raise ValueError("need a nonempty row-by-mask array")
    if floor <= 0 or not 0 < quantile < 1 or np.any(~np.isfinite(residuals)):
        raise ValueError("scales must be finite and positive")
    return np.maximum(np.quantile(residuals, quantile, axis=0), floor)


def calibrate(residuals: np.ndarray, scales: np.ndarray, alpha: float,
              random_indices: np.ndarray, shift_indices: np.ndarray) -> dict[str, np.ndarray]:
    """Return one radius per mask. No test outcomes are accepted here."""
    n, k = residuals.shape
    if scales.shape != (k,) or np.any(scales <= 0):
        raise ValueError("one strictly positive scale per mask is required")
    rows = np.arange(n)
    matched = np.array([conformal_quantile(residuals[:, j], alpha) for j in range(k)])
    constant = lambda q: np.full(k, q, dtype=float)
    return {
        "clean_cp": constant(conformal_quantile(residuals[:, 0], alpha)),
        "random_mask_cp": constant(conformal_quantile(residuals[rows, random_indices], alpha)),
        "pooled_augmentation": constant(conformal_quantile(residuals.ravel(), alpha)),
        "mask_matched": matched,
        "mask_envelope": constant(float(matched.max())),
        "bonferroni": np.array([conformal_quantile(residuals[:, j], alpha / k) for j in range(k)]),
        "rowmax": constant(conformal_quantile(residuals.max(axis=1), alpha)),
        "scaled_rowmax": scales * conformal_quantile((residuals / scales).max(axis=1), alpha),
        "policy_matched_shift": constant(conformal_quantile(residuals[rows, shift_indices], alpha)),
    }


def prediction_shift_indices(predictions: np.ndarray) -> np.ndarray:
    """Label-free, fixed tie-breaking by earliest column; clean is column zero."""
    return np.abs(predictions - predictions[:, :1]).argmax(axis=1)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half
