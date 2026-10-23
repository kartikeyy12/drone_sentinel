"""Helpers shared by scripts/train_cv.py and scripts/run_ablation.py."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import plotstyle
from .cv import AMBIENT, FoldFeatures, aggregate, fold_metrics, run_fold_config
from .dataset import Dataset, columns_with_prefix


def select_columns(ds: Dataset, cfg: dict[str, Any], drop_level_features: bool) -> np.ndarray:
    """Feature matrix with the level (loudness) features removed if asked."""
    if not drop_level_features:
        return ds.X
    return ds.X[:, ~columns_with_prefix(ds.names, cfg["m3"]["level_feature_prefixes"])]


def cv_run(X: np.ndarray, ds: Dataset, folds: list[tuple[np.ndarray, np.ndarray]], k: int | None, svm: dict[str, Any],
           class_weight: str | None, seed: int, y: np.ndarray | None = None) -> dict[str, Any]:
    """Grouped CV of one pipeline. Returns per-fold metrics, their aggregate and the out-of-fold predictions/scores.

    `y` overrides the labels (used by the permutation check); evaluation then uses the same overridden labels.
    """
    y = ds.y if y is None else y
    classes = sorted(set(y.tolist()))
    pred = np.empty(len(y), dtype=object)
    S = np.zeros((len(y), len(classes)))
    fold_id = np.zeros(len(y), int)
    per_fold = []
    for i, (tr, te) in enumerate(folds):
        ff = FoldFeatures(X[tr], X[te])
        m, p, s = run_fold_config(ff, y[tr], y[te], ds.hard_negative[te], k, svm, class_weight, seed)
        per_fold.append(m)
        pred[te], S[te], fold_id[te] = p, s, i
    return {"folds": per_fold, "aggregate": aggregate(per_fold), "pred": pred.astype(str), "scores": S, "fold_id": fold_id,
            "classes": classes}


def pooled_metrics(ds: Dataset, pred: np.ndarray, S: np.ndarray, classes: list[str]) -> dict[str, float]:
    return fold_metrics(ds.y, pred, S, classes, ds.hard_negative)


def permuted_labels(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle labels across whole source groups (every window of a group keeps one shared label)."""
    ug, inv = np.unique(groups, return_inverse=True)
    first = np.array([y[np.flatnonzero(inv == i)[0]] for i in range(len(ug))])
    mixed = {c: np.unique(y[inv == i]) for i, c in enumerate(ug) if len(np.unique(y[inv == i])) > 1}
    if mixed:
        raise ValueError(f"{len(mixed)} groups carry more than one label, e.g. {list(mixed)[:3]}")
    return rng.permutation(first)[inv]


def false_alarm_table(ds: Dataset, pred: np.ndarray, groups_as_rows: bool = False) -> pd.DataFrame:
    """Per source category: how many ambient windows were called a non-ambient class, and the rate."""
    amb = ds.y == AMBIENT
    df = pd.DataFrame({"category": ds.category[amb], "hard_negative": ds.hard_negative[amb], "false_alarm": pred[amb] != AMBIENT})
    t = df.groupby(["category", "hard_negative"]).agg(windows=("false_alarm", "size"), false_alarms=("false_alarm", "sum"),
                                                      false_alarm_rate=("false_alarm", "mean")).reset_index()
    return t.sort_values("false_alarm_rate", ascending=False).reset_index(drop=True)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], path: Path, title: str) -> Path:
    """Counts with row-normalised shading (single hue ramp carries the share, the numbers carry the counts)."""
    import matplotlib.pyplot as plt

    plotstyle.apply()
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.6, 4.2), constrained_layout=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if norm[i, j] > 0.55 else plotstyle.INK)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    ax.set_title(title, fontsize=9)
    return plotstyle.save(fig, path)
