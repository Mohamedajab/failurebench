import math
import numpy as np
import pytest

from failurebench.conformal import (conformal_quantile, calibrate, independent_scales,
                                    prediction_shift_indices, wilson_interval)
from failurebench.models import feature_ranking, fit_predictor, split_indices
from failurebench.data import synthetic


def test_rank_not_interpolated():
    assert conformal_quantile(np.arange(1, 10), .2) == 8
    assert conformal_quantile(np.arange(1, 10), .1) == 9


def test_infinite_quantile_for_small_sample():
    assert math.isinf(conformal_quantile(np.arange(9), .05))
    assert math.isinf(conformal_quantile(np.array([]), .1))


@pytest.mark.parametrize("alpha", [0, 1, -.1, 1.1])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        conformal_quantile(np.arange(10), alpha)


def test_nonfinite_and_negative_scores_rejected():
    for scores in ([np.nan], [np.inf], [-1]):
        with pytest.raises(ValueError):
            conformal_quantile(np.array(scores), .1)


def test_ties_are_retained():
    assert conformal_quantile(np.zeros(19), .1) == 0


def test_original_row_split_disjoint_complete_and_deterministic():
    ids = split_indices(1001, 101)
    assert len(set(np.concatenate(ids))) == 1001
    for a in range(4):
        for b in range(a):
            assert len(np.intersect1d(ids[a], ids[b])) == 0
    assert all(np.array_equal(a, b) for a, b in zip(ids, split_indices(1001, 101)))


def test_scale_floor():
    assert np.all(independent_scales(np.zeros((20, 4))) == .05)


def test_clean_predictor_preprocessing_is_training_only():
    rng = np.random.default_rng(57)
    x = rng.normal(size=(100, 5))
    y = x[:, 0] + rng.normal(size=100)
    p = fit_predictor(x, y, "ridge", "clean", 101)
    before = p.mean.copy()
    p.predict_masks(np.ones((10, 5)) * 1e9, np.eye(5, dtype=bool))
    assert np.array_equal(p.mean, before)
    assert np.allclose(p.mean, x.mean(axis=0))


def test_normalization_and_imputation_consistent():
    x = np.array([[0., 1.], [1., 2.], [3., 9.], [10., 20.]])
    p = fit_predictor(x, np.array([0., 1., 3., 5.]), "ridge", "clean", 101)
    z = p.encode(x, np.ones_like(x, dtype=bool))
    assert np.allclose(z[:, :2], (np.median(x, axis=0) - x.mean(axis=0)) / x.std(axis=0))
    assert np.all(z[:, 2:] == 1)


def test_feature_order_ties_are_stable():
    x = np.tile(np.arange(10)[:, None], (1, 3))
    assert np.array_equal(feature_ranking(x, np.arange(10)), [0, 1, 2])


def test_unit_scales_recover_rowmax_exactly():
    rng = np.random.default_rng(193)
    residuals = np.abs(rng.normal(size=(101, 5)))
    radii = calibrate(residuals, np.ones(5), .1, np.zeros(101, dtype=int), np.zeros(101, dtype=int))
    assert np.array_equal(radii["rowmax"], radii["scaled_rowmax"])


def test_simultaneous_coverage_matches_oracle():
    rng = np.random.default_rng(31)
    residuals = rng.uniform(size=(100, 4))
    radii = np.array([.5, .1, .4, .3])
    worst = (residuals / radii).argmax(axis=1)
    assert np.array_equal((residuals <= radii).all(axis=1), residuals[np.arange(100), worst] <= radii[worst])


def test_fixed_mask_counterexample_adaptive_selection():
    # Exact finite population: two disjoint 10% failure regions. At radius zero,
    # each fixed mask covers 90%, but adaptive selection covers only 80%.
    u = (np.arange(1000) + .5) / 1000
    residuals = np.stack((u < .1, (u >= .1) & (u < .2)), axis=1).astype(float)
    assert np.allclose((residuals <= 0).mean(axis=0), .9)
    assert np.isclose((residuals <= 0).all(axis=1).mean(), .8)


def test_label_free_policy_interface():
    assert np.array_equal(prediction_shift_indices(np.array([[0, 1, 3], [2, -2, 3]])), [2, 1])


def test_rank_guarantee_monte_carlo():
    # Exchangeable original rows. Multiple scores per row are strongly dependent.
    rng = np.random.default_rng(131)
    hits = 0
    trials = 4000
    for _ in range(trials):
        shared = np.abs(rng.normal(size=(100, 1)))
        scores = shared + rng.uniform(0, .5, size=(100, 6))
        q = conformal_quantile(scores[:99].max(axis=1), .1)
        hits += scores[99].max() <= q
    assert abs(hits / trials - .9) < .02


def test_wilson_extremes():
    lo, hi = wilson_interval(0, 100)
    assert lo == pytest.approx(0, abs=1e-12) and 0 < hi < .1
    lo, hi = wilson_interval(100, 100)
    assert .9 < lo < 1 and hi == pytest.approx(1)


def test_synthetic_reproducibility_and_shapes():
    a, y, _ = synthetic("synthetic_linear", 500)
    b, y2, _ = synthetic("synthetic_linear", 500)
    assert a.shape == (500, 10) and y.shape == (500,)
    assert np.array_equal(a, b) and np.array_equal(y, y2)
