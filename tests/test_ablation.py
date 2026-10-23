"""Ablation machinery: grid expansion, resume bookkeeping, nested selection hygiene, factor effects."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.ablation import (append_rows, expand_grid, factor_effects, fingerprint, fold_worker, load_rows,
                               missing_by_block_fold, nested_outer_worker, subset_worker, summarize_grid, cfg_id, block_key)
from sentinel.config import load_config
from sentinel.cv import make_folds

CFG = load_config()


def _toy(n_groups=70, per=6, d=10, seed=0):
    rng = np.random.default_rng(seed)
    g, y, X, hn = [], [], [], []
    for i in range(n_groups):
        lab = "aircraft" if i % 5 == 0 else "ambient"
        centre = rng.standard_normal(d) * 0.4 + (1.2 if lab == "aircraft" else 0.0)
        for _ in range(per):
            g.append(f"g{i}"), y.append(lab), hn.append(i % 7 == 0 and lab == "ambient"), X.append(centre + rng.standard_normal(d))
    return np.array(X), np.array(y), np.array(g), np.array(hn)


def test_expand_grid_size_ids_and_order_are_stable():
    grid = expand_grid(CFG)
    g = CFG["m3"]["grid"]
    assert len(grid) == len(g["variants"]) * len(g["drop_level_features"]) * len(g["pca"]) * len(g["svm"]) == 150
    ids = [cfg_id(c) for c in grid]
    assert len(set(ids)) == len(ids)
    assert grid == expand_grid(CFG)
    assert grid[0]["variant"] == "none" and grid[0]["pca"] is None


def test_fold_worker_returns_one_row_per_config_with_metrics():
    X, y, g, hn = _toy()
    tr, te = make_folds(y, g, 5, 0)[0]
    cfgs = [{"variant": "none", "drop_level": False, "pca": k, "kernel": "rbf", "C": 1.0} for k in (None, 3)]
    rows = fold_worker(X, y, hn, tr, te, cfgs, "balanced", 1)
    assert [r["pca"] for r in rows] == [None, 3] and all(0.5 < r["balanced_accuracy"] <= 1 for r in rows)
    assert all("specificity_hard_negative" in r for r in rows)


def test_resume_bookkeeping_finds_exactly_the_missing_cells(tmp_path):
    cfgs = [{"variant": "none", "drop_level": False, "pca": k, "kernel": "rbf", "C": 1.0} for k in (None, 5)]
    path = tmp_path / "rows.jsonl"
    assert len(load_rows(path, "fp")) == 0
    append_rows(path, [{**cfgs[0], "fold": 0, "balanced_accuracy": 0.7}, {**cfgs[1], "fold": 0, "balanced_accuracy": 0.6}], "fp")
    append_rows(path, [{**cfgs[0], "fold": 1, "balanced_accuracy": float("nan")}], "other")
    done = load_rows(path, "fp")
    assert len(done) == 2  # the row with another fingerprint is ignored
    todo = missing_by_block_fold(done, cfgs, 2)
    assert set(todo) == {("none|0", 1)} and len(todo[("none|0", 1)]) == 2  # fold 0 is complete, fold 1 is open
    done2 = load_rows(path, "other")
    assert len(done2) == 1 and np.isnan(done2["balanced_accuracy"].iloc[0]) or done2["balanced_accuracy"].iloc[0] is None


def test_fingerprint_changes_with_folds_seed_and_cache():
    X, y, g, hn = _toy()
    f1, f2 = make_folds(y, g, 5, 0), make_folds(y, g, 5, 1)
    base = fingerprint(CFG, {"none": "a"}, f1)
    assert base == fingerprint(CFG, {"none": "a"}, f1)
    assert base != fingerprint(CFG, {"none": "b"}, f1) and base != fingerprint(CFG, {"none": "a"}, f2)


def test_nested_selection_never_looks_at_outer_test_labels():
    X, y, g, hn = _toy()
    tr, te = make_folds(y, g, 5, 0)[0]
    cfgs = [{"variant": "none", "drop_level": False, "pca": k, "kernel": kern, "C": 1.0}
            for k in (None, 3) for kern in ("linear", "rbf")]
    Xs = {"none|0": X}
    a = nested_outer_worker(Xs, y, g, hn, tr, te, cfgs, 3, "balanced", 1, 0)
    y2 = y.copy()
    y2[te] = np.where(y2[te] == "aircraft", "ambient", "aircraft")  # corrupt only the outer TEST labels
    b = nested_outer_worker(Xs, y2, g, hn, tr, te, cfgs, 3, "balanced", 1, 0)
    assert a["chosen"] == b["chosen"] and np.array_equal(a["inner_ba"], b["inner_ba"])
    np.testing.assert_array_equal(a["pred"], b["pred"])  # predictions depend on training data only
    assert a["metrics"]["balanced_accuracy"] != b["metrics"]["balanced_accuracy"]
    assert len(a["inner_ba"]) == len(cfgs) and a["chosen"] in cfgs


def test_subset_worker_handles_single_feature_and_all_features():
    X, y, g, hn = _toy()
    tr, te = make_folds(y, g, 5, 0)[0]
    masks = [("all", np.ones(10, bool)), ("one", np.arange(10) == 2)]
    rows = subset_worker(X, y, hn, tr, te, masks, {"kernel": "rbf", "C": 1.0}, "balanced", 0)
    assert [r["subset"] for r in rows] == ["all", "one"] and rows[1]["n_features"] == 1


def test_summarize_and_factor_effects_on_a_hand_built_table():
    rows = []
    for variant in ("none", "specsub"):
        for pca in (None, 10):
            for fold in range(3):
                ba = 0.70 + (0.05 if variant == "specsub" else 0.0) + (0.02 if pca == 10 else 0.0) + 0.001 * fold
                rows.append({"variant": variant, "drop_level": False, "pca": pca, "kernel": "rbf", "C": 1.0, "fold": fold,
                             **{m: ba for m in ("balanced_accuracy", "recall_aircraft", "specificity_plain",
                                                "specificity_hard_negative", "precision_aircraft", "f1_aircraft",
                                                "ap_aircraft", "auc_aircraft")}})
    s = summarize_grid(pd.DataFrame(rows))
    assert len(s) == 4 and s["n_folds"].eq(3).all() and s["balanced_accuracy_mean"].is_monotonic_decreasing
    fe = factor_effects(s)
    v = fe[(fe.factor == "variant") & (fe.level == "specsub")].iloc[0]
    assert v["mean_diff"] == pytest.approx(0.05) and v["better_in"] == 2 and v["worse_in"] == 0
    p = fe[(fe.factor == "pca") & (fe.level == "10")].iloc[0]
    assert p["mean_diff"] == pytest.approx(0.02)
