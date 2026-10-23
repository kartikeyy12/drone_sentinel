"""Grouped cross-validation, fold-level preprocessing, SVM fitting and metrics.

Everything that learns from data (z-scoring, PCA, SVM) is fitted on the training part of a fold only. Folds are
StratifiedGroupKFold on the source-recording id, so no recording contributes windows to both sides of a split.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC

from .svdpca import SVDPCA

AMBIENT = "ambient"


# ---- folds --------------------------------------------------------------------------------------------------------

def check_folds(y: np.ndarray, groups: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Raise unless the folds are a leak-free partition in which every class is on both sides of every split."""
    n = len(y)
    seen = np.zeros(n, int)
    classes = set(y.tolist())
    for i, (tr, te) in enumerate(folds):
        shared = set(groups[tr].tolist()) & set(groups[te].tolist())
        if shared:
            raise ValueError(f"fold {i}: {len(shared)} source groups appear in both train and test, e.g. {sorted(shared)[:3]}")
        if set(y[tr].tolist()) != classes or set(y[te].tolist()) != classes:
            raise ValueError(f"fold {i}: a class is missing from train or test")
        seen[te] += 1
    if not np.all(seen == 1):
        raise ValueError("every window must be in exactly one test fold")


def make_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [(tr, te) for tr, te in sgkf.split(np.zeros(len(y)), y, groups)]
    check_folds(y, groups, folds)
    return folds


# ---- fold-level preprocessing --------------------------------------------------------------------------------------

class FoldFeatures:
    """Z-score and full PCA fitted on one training fold. PCA components are nested, so any k is a column slice."""

    def __init__(self, Xtr: np.ndarray, Xte: np.ndarray) -> None:
        self.mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)
        self.Ztr, self.Zte = (Xtr - self.mu) / self.sd, (Xte - self.mu) / self.sd
        self.pca = SVDPCA().fit(self.Ztr)
        self.Ptr, self.Pte = self.pca.transform(self.Ztr), self.pca.transform(self.Zte)

    def view(self, k: int | None) -> tuple[np.ndarray, np.ndarray]:
        """(train, test) matrices: all z-scored features if k is None, else the first k principal components."""
        if k is None:
            return self.Ztr, self.Zte
        k = min(int(k), self.Ptr.shape[1])
        return self.Ptr[:, :k], self.Pte[:, :k]


def fit_svm(A: np.ndarray, y: np.ndarray, svm: dict[str, Any], class_weight: str | None, seed: int) -> SVC:
    return SVC(kernel=svm["kernel"], C=float(svm["C"]), gamma=svm.get("gamma", "scale"), class_weight=class_weight,
               cache_size=500, random_state=seed).fit(A, y)


def class_scores(model: SVC, A: np.ndarray) -> np.ndarray:
    """Decision scores aligned with model.classes_, shape (n, n_classes); larger means more likely that class."""
    d = model.decision_function(A)
    if d.ndim == 1:  # binary: positive favours classes_[1]
        return np.column_stack([-d, d])
    return d


# ---- metrics -----------------------------------------------------------------------------------------------------

def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, S: np.ndarray, classes: list[str],
                 hard_negative: np.ndarray) -> dict[str, float]:
    """Balanced accuracy, per-class recall/precision/F1, AP and AUC of each non-ambient class, ambient specificity
    split into plain and hard-negative windows."""
    m: dict[str, float] = {"n_test": float(len(y_true)), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))}
    for j, c in enumerate(classes):
        t, p = y_true == c, y_pred == c
        tp = float(np.sum(t & p))
        rec = tp / t.sum() if t.any() else np.nan
        prec = tp / p.sum() if p.any() else 0.0
        m[f"recall_{c}"] = float(rec)
        m[f"precision_{c}"] = float(prec)
        m[f"f1_{c}"] = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        m[f"n_{c}"] = float(t.sum())
        if c != AMBIENT and t.any() and (~t).any():
            m[f"ap_{c}"] = float(average_precision_score(t, S[:, j]))
            m[f"auc_{c}"] = float(roc_auc_score(t, S[:, j]))
    amb = y_true == AMBIENT
    for name, sel in (("plain", amb & ~hard_negative), ("hard_negative", amb & hard_negative)):
        m[f"specificity_{name}"] = float(np.mean(y_pred[sel] == AMBIENT)) if sel.any() else np.nan
        m[f"n_{name}"] = float(sel.sum())
    return m


def aggregate(folds: list[dict[str, float]]) -> dict[str, float]:
    """Mean and standard deviation over folds (NaN-aware) for every metric except the counts, which are summed."""
    out: dict[str, float] = {}
    for k in folds[0]:
        v = np.array([f[k] for f in folds], float)
        if k.startswith("n_"):
            out[k] = float(np.nansum(v))
        else:
            out[f"{k}_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
            out[f"{k}_std"] = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else np.nan
    return out


def group_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, classes: list[str],
                       n_boot: int, seed: int, level: float = 0.95) -> dict[str, float]:
    """Percentile CI of balanced accuracy and per-class recall, resampling whole source groups with replacement."""
    ug, inv = np.unique(groups, return_inverse=True)
    tot = np.zeros((len(ug), len(classes)))
    hit = np.zeros_like(tot)
    for j, c in enumerate(classes):
        t = y_true == c
        np.add.at(tot[:, j], inv[t], 1.0)
        np.add.at(hit[:, j], inv[t & (y_pred == c)], 1.0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ug), size=(n_boot, len(ug)))
    T, H = tot[draws].sum(axis=1), hit[draws].sum(axis=1)  # (n_boot, n_classes)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = H / T
    ba = np.nanmean(rec, axis=1)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    out = {"balanced_accuracy_lo": float(np.percentile(ba, lo)), "balanced_accuracy_hi": float(np.percentile(ba, hi))}
    for j, c in enumerate(classes):
        r = rec[:, j][np.isfinite(rec[:, j])]
        out[f"recall_{c}_lo"], out[f"recall_{c}_hi"] = float(np.percentile(r, lo)), float(np.percentile(r, hi))
    return out


# ---- one fold, one config ------------------------------------------------------------------------------------------

def run_fold_config(ff: FoldFeatures, ytr: np.ndarray, yte: np.ndarray, hard_negative_te: np.ndarray, k: int | None,
                    svm: dict[str, Any], class_weight: str | None, seed: int
                    ) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Fit on the fold's training part, score its test part. Returns (metrics, predictions, class scores)."""
    A, B = ff.view(k)
    model = fit_svm(A, ytr, svm, class_weight, seed)
    pred = model.predict(B)
    S = class_scores(model, B)
    classes = [str(c) for c in model.classes_]
    return fold_metrics(yte, pred, S, classes, hard_negative_te), pred, S
