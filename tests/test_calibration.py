"""Platt calibration, calibration metrics, prior adjustment and false-alarm thresholds."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from sentinel import calibration as cal


def _sample(n=200_000, a=1.7, b=-2.1, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.normal(0, 2, n)
    return s, rng.random(n) < expit(a * s + b)


def test_platt_recovers_the_generating_curve():
    s, y = _sample()
    c = cal.PlattCalibrator().fit(s, y)
    assert c.a_ == pytest.approx(1.7, abs=0.03) and c.b_ == pytest.approx(-2.1, abs=0.03)
    assert c.prior_ == pytest.approx(y.mean())
    np.testing.assert_allclose(c.predict(np.array([-3.0, 0.0, 3.0])), expit(1.7 * np.array([-3.0, 0.0, 3.0]) - 2.1), atol=0.01)


def test_platt_matches_sklearn_logistic_regression():
    s, y = _sample(n=20_000, seed=3)
    c = cal.PlattCalibrator().fit(s, y)
    lr = LogisticRegression(C=1e12, tol=1e-10, max_iter=1000).fit(s[:, None], y)
    assert c.a_ == pytest.approx(float(lr.coef_[0, 0]), abs=1e-4) and c.b_ == pytest.approx(float(lr.intercept_[0]), abs=1e-4)


def test_platt_is_monotone_and_validates_input():
    s, y = _sample(n=5000)
    p = cal.PlattCalibrator().fit(s, y).predict(np.linspace(-6, 6, 50))
    assert np.all(np.diff(p) > 0) and p.min() > 0 and p.max() < 1
    with pytest.raises(ValueError, match="both classes"):
        cal.PlattCalibrator().fit(np.arange(5.0), np.zeros(5, bool))
    with pytest.raises(ValueError, match="1-D"):
        cal.PlattCalibrator().fit(np.zeros((3, 2)), np.array([0, 1, 0]))


def test_platt_handles_well_separated_scores_without_blowing_up():
    s = np.r_[np.linspace(-5, -1, 200), np.linspace(1, 5, 200)]
    y = s > 0
    c = cal.PlattCalibrator().fit(s, y)
    assert np.isfinite(c.a_) and np.isfinite(c.b_) and c.a_ > 0
    assert c.predict(np.array([-5.0]))[0] < 0.01 and c.predict(np.array([5.0]))[0] > 0.99


def test_adjust_prior_is_the_bayes_odds_shift():
    # class-conditional likelihood ratio LR(s) is known, so p at any prior is known in closed form
    lr = np.exp(np.linspace(-4, 4, 41))
    p_old = 0.09 * lr / (0.09 * lr + 0.91)
    p_new = 0.002 * lr / (0.002 * lr + 0.998)
    np.testing.assert_allclose(cal.adjust_prior(p_old, 0.09, 0.002), p_new, rtol=1e-9)
    np.testing.assert_allclose(cal.adjust_prior(p_old, 0.09, 0.09), p_old, rtol=1e-9)
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            cal.adjust_prior(p_old, 0.09, bad)


def test_brier_and_log_loss_known_values():
    y = np.array([1, 0, 1, 0])
    assert cal.brier(np.full(4, 0.5), y) == pytest.approx(0.25)
    assert cal.log_loss(np.full(4, 0.5), y) == pytest.approx(np.log(2))
    assert cal.brier(y.astype(float), y) == 0.0
    assert np.isfinite(cal.log_loss(1 - y.astype(float), y))  # clipped, not infinite


def test_ece_is_small_when_calibrated_and_large_when_not():
    rng = np.random.default_rng(1)
    p = rng.random(100_000)
    assert cal.ece(p, rng.random(100_000) < p) < 0.01
    assert cal.ece(p, rng.random(100_000) < 1 - p) > 0.2


def test_reliability_table_bins_have_equal_counts_and_sum_to_n():
    rng = np.random.default_rng(2)
    p = rng.random(1003)
    t = cal.reliability_table(p, rng.random(1003) < p, 10)
    assert t["n"].sum() == 1003 and t["n"].max() - t["n"].min() <= 1
    assert t["mean_predicted"].is_monotonic_increasing


def test_threshold_for_far_on_distinct_scores():
    neg = np.arange(1, 11) / 10.0  # 0.1 .. 1.0, ten negatives
    t = cal.threshold_for_far(neg, 0.2)
    assert np.mean(neg >= t) == pytest.approx(0.2)           # exactly two false alarms (0.9 and 1.0)
    assert 0.8 < t <= 0.9
    assert np.mean(neg >= cal.threshold_for_far(neg, 0.0)) == 0.0
    assert cal.threshold_for_far(neg, 1.0) == -np.inf
    assert np.mean(neg >= cal.threshold_for_far(neg, 0.05)) == 0.0  # 0.5 of a window is not allowed


def test_threshold_for_far_never_exceeds_the_target_even_with_ties():
    neg = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    t = cal.threshold_for_far(neg, 0.2)  # one false alarm allowed, but the three 1.0 scores are tied
    assert np.mean(neg >= t) <= 0.2
    rng = np.random.default_rng(0)
    for _ in range(50):
        v = np.round(rng.random(rng.integers(20, 300)), 1)  # many ties
        for far in (0.0, 0.01, 0.05, 0.1, 0.5):
            assert np.mean(v >= cal.threshold_for_far(v, far)) <= far + 1e-12
    with pytest.raises(ValueError):
        cal.threshold_for_far(np.array([]), 0.1)
    with pytest.raises(ValueError):
        cal.threshold_for_far(neg, 1.5)


def test_far_recall_curve_runs_from_origin_to_corner_and_is_monotone():
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.2
    s = rng.normal(0, 1, 2000) + 1.5 * y
    c = cal.far_recall_curve(s, y, max_points=100)
    assert len(c) <= 100 and c["far"].iloc[0] == 0 and c["recall"].iloc[0] == 0
    assert c["far"].iloc[-1] == 1 and c["recall"].iloc[-1] == 1
    assert c["far"].is_monotonic_increasing and c["recall"].is_monotonic_increasing


def test_crossfit_threshold_ignores_the_held_out_fold():
    rng = np.random.default_rng(0)
    n = 3000
    fold = rng.integers(0, 5, n)
    is_pos = rng.random(n) < 0.15
    s = rng.normal(0, 1, n) + 2.0 * is_pos
    grid = [0.05, 0.1]
    a = cal.crossfit_operating_points(s, is_pos, ~is_pos, fold, grid)
    s2 = s.copy()
    held = (fold == 2) & ~is_pos
    s2[held] += 5.0                                   # make the held-out fold's negatives look like targets
    b = cal.crossfit_operating_points(s2, is_pos, ~is_pos, fold, grid)
    ta = a[a["fold"] == 2].set_index("target_far")["threshold"]
    tb = b[b["fold"] == 2].set_index("target_far")["threshold"]
    pd.testing.assert_series_equal(ta, tb)             # fold 2's own threshold did not move
    assert (b[b["fold"] == 2]["far"] > a[a["fold"] == 2]["far"]).all()  # but its achieved false-alarm rate did
    assert len(a) == len(grid) * 5
    sm = cal.summarize_crossfit(a)
    assert list(sm["target_far"]) == grid and (sm["far_max"] >= sm["far_mean"]).all()
