"""Predictor fit uses only model-training rows; preprocessing is frozen there."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


def split_indices(n: int, seed: int, fractions=(.5, .2, .15, .15)):
    if not np.isclose(sum(fractions), 1) or any(f <= 0 for f in fractions):
        raise ValueError("positive fractions must sum to one")
    ids = np.random.default_rng(seed).permutation(n)
    boundaries = np.floor(np.cumsum(fractions[:-1]) * n).astype(int)
    return tuple(np.split(ids, boundaries))


def feature_ranking(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=0)
    yc = y - y.mean()
    denom = np.sqrt((centered ** 2).sum(axis=0) * (yc ** 2).sum())
    corr = np.divide(np.abs(centered.T @ yc), denom, out=np.zeros(x.shape[1]), where=denom > 0)
    return np.lexsort((np.arange(x.shape[1]), -corr))


@dataclass
class FrozenPredictor:
    model: object
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    response_mean: float
    response_std: float

    def standardized(self, x):
        return (x - self.mean) / self.std

    def encode(self, x, masks):
        z = self.standardized(x).copy()
        # Impute raw values by the training median, then apply frozen scaling.
        z = np.where(masks, (self.median - self.mean) / self.std, z)
        return np.concatenate((z, masks.astype(float)), axis=1)

    def predict_masks(self, x, masks):
        """masks: K-by-d Boolean array; returns n-by-K predictions."""
        n, d = x.shape
        k = len(masks)
        repeated = np.tile(x, (k, 1))
        tiled_masks = np.repeat(masks, n, axis=0)
        return self.model.predict(self.encode(repeated, tiled_masks)).reshape(k, n).T

    def normalized_response(self, y):
        return (y - self.response_mean) / self.response_std


def fit_predictor(x, y, model_name, arm, seed):
    std = x.std(axis=0)
    std[std < 1e-8] = 1
    ystd = max(float(y.std()), 1e-8)
    model = (Ridge(alpha=1.0) if model_name == "ridge" else
             HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15,
                learning_rate=.08, l2_regularization=1.0, min_samples_leaf=20,
                early_stopping=False, random_state=seed))
    frozen = FrozenPredictor(model, np.median(x, axis=0), x.mean(axis=0), std,
                             float(y.mean()), ystd)
    clean_masks = np.zeros_like(x, dtype=bool)
    encoded = frozen.encode(x, clean_masks)
    yn = frozen.normalized_response(y)
    if arm == "mask_augmented":
        # One extra view per original training row. It never crosses split boundaries.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 67]))
        choice = rng.integers(0, x.shape[1] + 1, size=len(x))
        masks = np.zeros_like(x, dtype=bool)
        affected = np.flatnonzero(choice > 0)
        masks[affected, choice[affected] - 1] = True
        encoded = np.concatenate((encoded, frozen.encode(x, masks)), axis=0)
        yn = np.concatenate((yn, yn))
    elif arm != "clean":
        raise ValueError(arm)
    model.fit(encoded, yn)
    return frozen
