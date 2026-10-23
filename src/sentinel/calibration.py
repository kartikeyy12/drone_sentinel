"""Score calibration and operating thresholds.

The SVM's decision values are not probabilities. A Platt calibrator (a logistic curve on the score) is fitted on
OUT-OF-FOLD scores, i.e. scores of recordings the scoring model never trained on. It is not `SVC(probability=True)`,
which fits its sigmoid on internal folds of the training windows (not grouped by recording).

Probabilities are calibrated for the class prevalence of the data they were fitted on (ESC-50 windows, about 9 %
aircraft). A deployment with a different target prevalence needs `adjust_prior`. The false-alarm threshold, not the
probability value, is what sets an operating point.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import expit, logit


class PlattCalibrator:
    """p = sigmoid(a * score + b), maximum-likelihood fit by Newton's method with a backtracking line search."""

    def __init__(self) -> None:
        self.a_, self.b_, self.prior_ = 1.0, 0.0, float("nan")

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        s, t = np.asarray(scores, np.float64), np.asarray(y, np.float64)
        if s.ndim != 1 or len(s) != len(t):
            raise ValueError("scores must be 1-D and match y")
        if not (0 < t.sum() < len(t)):
            raise ValueError("calibration needs both classes present")
        self.prior_ = float(t.mean())
        theta = np.array([1.0, logit(self.prior_)])

        def nll(th: np.ndarray) -> float:
            z = th[0] * s + th[1]
            return float(np.sum(np.logaddexp(0.0, z) - t * z))

        for _ in range(100):
            p = expit(theta[0] * s + theta[1])
            g = np.array([np.sum((p - t) * s), np.sum(p - t)])
            w = p * (1 - p) + 1e-12
            H = np.array([[np.sum(w * s * s), np.sum(w * s)], [np.sum(w * s), np.sum(w)]]) + 1e-9 * np.eye(2)
            step = np.linalg.solve(H, g)
            f0, k = nll(theta), 1.0
            while nll(theta - k * step) > f0 + 1e-12 and k > 1e-8:
                k /= 2
            theta = theta - k * step
            if np.max(np.abs(k * step)) < 1e-10:
                break
        self.a_, self.b_ = float(theta[0]), float(theta[1])
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return expit(self.a_ * np.asarray(scores, np.float64) + self.b_)


def adjust_prior(p: np.ndarray, old_prior: float, new_prior: float) -> np.ndarray:
    """Re-express probabilities calibrated at `old_prior` for a population with `new_prior` (Bayes, odds shift)."""
    for q in (old_prior, new_prior):
        if not 0 < q < 1:
            raise ValueError("priors must be in (0, 1)")
    z = logit(np.clip(np.asarray(p, np.float64), 1e-12, 1 - 1e-12)) + logit(new_prior) - logit(old_prior)
    return expit(z)


# ---- calibration quality -------------------------------------------------------------------------------------------------

def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of the predicted probability: count, mean prediction, observed frequency of the positive class."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    order = np.argsort(p, kind="mergesort")
    rows = []
    for b, idx in enumerate(np.array_split(order, n_bins)):
        if len(idx):
            rows.append({"bin": b, "n": int(len(idx)), "mean_predicted": float(p[idx].mean()), "observed": float(y[idx].mean())})
    return pd.DataFrame(rows)


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over equal-count bins: sum of (bin share) x |observed - predicted|."""
    t = reliability_table(p, y, n_bins)
    return float(np.sum(t["n"] * (t["observed"] - t["mean_predicted"]).abs()) / t["n"].sum())


# ---- thresholds from a false-alarm target -----------------------------------------------------------------------------------

def threshold_for_far(neg_scores: np.ndarray, far: float) -> float:
    """Smallest threshold t such that the share of negatives with score >= t is at most `far` (alarm when score >= t).

    The smallest such t keeps the most positives. Ties are handled by moving just above the tied value.
    """
    v = np.sort(np.asarray(neg_scores, np.float64))[::-1]
    n = len(v)
    if n == 0:
        raise ValueError("no negative scores")
    if not 0 <= far <= 1:
        raise ValueError("far must be in [0, 1]")
    k = int(np.floor(far * n + 1e-9))  # number of false alarms allowed
    if k >= n:
        return float("-inf")
    return float(np.nextafter(v[k], np.inf))


def far_recall_curve(scores: np.ndarray, is_pos: np.ndarray, max_points: int = 400) -> pd.DataFrame:
    """(false-alarm rate, recall, threshold) along the ROC of one score, thinned to at most max_points rows."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(is_pos, scores)
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).astype(int))
    return pd.DataFrame({"far": fpr[keep], "recall": tpr[keep], "threshold": thr[keep]})


def crossfit_operating_points(scores: np.ndarray, is_pos: np.ndarray, is_neg: np.ndarray, fold_id: np.ndarray,
                              far_grid: Sequence[float]) -> pd.DataFrame:
    """For each false-alarm target and each fold: pick the threshold on the OTHER folds' negatives, apply it to the held-out
    fold, record the false-alarm rate and recall achieved there. Shows whether a target transfers to unseen recordings."""
    rows = []
    for far in far_grid:
        for f in np.unique(fold_id):
            other, mine = fold_id != f, fold_id == f
            t = threshold_for_far(scores[other & is_neg], far)
            rows.append({"target_far": float(far), "fold": int(f), "threshold": t,
                         "far": float(np.mean(scores[mine & is_neg] >= t)), "recall": float(np.mean(scores[mine & is_pos] >= t))})
    return pd.DataFrame(rows)


def summarize_crossfit(cf: pd.DataFrame) -> pd.DataFrame:
    g = cf.groupby("target_far")
    return g.agg(far_mean=("far", "mean"), far_std=("far", "std"), far_max=("far", "max"), recall_mean=("recall", "mean"),
                 recall_std=("recall", "std"), recall_min=("recall", "min")).reset_index()
