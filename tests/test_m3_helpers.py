"""Helpers shared by the M3 scripts: permutation, false-alarm table, column selection, confusion plot, CV driver."""
from __future__ import annotations

import numpy as np
import pytest

from sentinel.config import load_config
from sentinel.cv import make_folds
from sentinel.dataset import Dataset
from sentinel.m3 import cv_run, false_alarm_table, permuted_labels, plot_confusion, select_columns

CFG = load_config()


def _ds(n_groups=60, per=6, d=8, seed=0):
    rng = np.random.default_rng(seed)
    g, y, X, hn, cat = [], [], [], [], []
    for i in range(n_groups):
        lab = "aircraft" if i % 5 == 0 else "ambient"
        centre = rng.standard_normal(d) * 0.4 + (1.5 if lab == "aircraft" else 0.0)
        for _ in range(per):
            g.append(f"g{i}"), y.append(lab), hn.append(lab == "ambient" and i % 4 == 0), cat.append("engine" if i % 4 == 0 else "rain")
            X.append(centre + rng.standard_normal(d))
    names = ["mfcc00_mean", "rms_mean", "contrast0_mean"] + [f"f{i}" for i in range(d - 3)]
    return Dataset("none", np.array(X), np.array(y), np.array(g), np.array(hn), np.array(cat), names, np.arange(len(y)), {})


def test_permuted_labels_keep_groups_whole_and_class_counts():
    ds = _ds()
    yp = permuted_labels(ds.y, ds.groups, np.random.default_rng(0))
    assert len(yp) == len(ds.y)
    for gname in np.unique(ds.groups):
        assert len(set(yp[ds.groups == gname])) == 1  # every window of a group shares one label
    assert sorted(set(zip(ds.groups, yp))).__len__() == len(set(ds.groups))
    assert (yp == "aircraft").sum() == (ds.y == "aircraft").sum()  # same number of aircraft windows (equal group sizes here)
    assert not np.array_equal(yp, ds.y)


def test_permuted_labels_rejects_groups_with_mixed_labels():
    y = np.array(["a", "b", "a"])
    with pytest.raises(ValueError, match="more than one label"):
        permuted_labels(y, np.array(["g", "g", "h"]), np.random.default_rng(0))


def test_label_permutation_drops_cv_to_chance():
    ds = _ds(n_groups=120)
    real = cv_run(ds.X, ds, make_folds(ds.y, ds.groups, 5, 0), None, {"kernel": "rbf", "C": 1.0}, "balanced", 0)
    yp = permuted_labels(ds.y, ds.groups, np.random.default_rng(1))
    perm = cv_run(ds.X, ds, make_folds(yp, ds.groups, 5, 0), None, {"kernel": "rbf", "C": 1.0}, "balanced", 0, y=yp)
    assert real["aggregate"]["balanced_accuracy_mean"] > 0.8
    assert abs(perm["aggregate"]["balanced_accuracy_mean"] - 0.5) < 0.15


def test_cv_run_predictions_cover_every_row_once_and_use_grouped_folds():
    ds = _ds()
    folds = make_folds(ds.y, ds.groups, 5, 0)
    run = cv_run(ds.X, ds, folds, 4, {"kernel": "linear", "C": 1.0}, "balanced", 0)
    assert set(run["pred"]) <= {"aircraft", "ambient"} and len(run["pred"]) == len(ds.y)
    assert sorted(set(run["fold_id"])) == list(range(5)) and run["scores"].shape == (len(ds.y), 2)
    for i, (tr, te) in enumerate(folds):
        assert np.all(run["fold_id"][te] == i)


def test_false_alarm_table_counts_and_sorting():
    ds = _ds()
    pred = ds.y.copy()
    idx = np.flatnonzero((ds.y == "ambient") & (ds.category == "engine"))
    pred[idx[:5]] = "aircraft"
    t = false_alarm_table(ds, pred)
    top = t.iloc[0]
    assert top["category"] == "engine" and top["false_alarms"] == 5 and bool(top["hard_negative"])
    assert top["false_alarm_rate"] == pytest.approx(5 / top["windows"])
    assert t["false_alarm_rate"].is_monotonic_decreasing


def test_select_columns_drops_exactly_the_level_features():
    ds = _ds()
    X = select_columns(ds, CFG, True)
    assert X.shape[1] == ds.X.shape[1] - 3
    np.testing.assert_array_equal(X, ds.X[:, 3:])
    assert select_columns(ds, CFG, False) is ds.X


def test_plot_confusion_writes_a_png(tmp_path):
    y = np.array(["aircraft", "ambient", "ambient", "aircraft"])
    p = plot_confusion(y, np.array(["aircraft", "ambient", "aircraft", "ambient"]), ["aircraft", "ambient"], tmp_path / "c.png", "t")
    assert p.exists() and p.stat().st_size > 1000
