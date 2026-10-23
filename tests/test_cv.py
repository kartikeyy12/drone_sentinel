"""Grouped folds, fold-level preprocessing, metrics, bootstrap and pipeline/slicing equivalence."""
from __future__ import annotations

import numpy as np
import pytest

from sentinel.cv import (FoldFeatures, aggregate, check_folds, class_scores, fit_svm, fold_metrics, group_bootstrap_ci,
                         make_folds, run_fold_config)
from sentinel.pipeline import FeaturePipeline
from sentinel.svdpca import SVDPCA


def _toy(n_groups=60, per=8, d=12, seed=0, shift=1.5):
    rng = np.random.default_rng(seed)
    groups, y, X = [], [], []
    for g in range(n_groups):
        lab = "aircraft" if g % 5 == 0 else "ambient"
        centre = rng.standard_normal(d) * 0.5 + (shift if lab == "aircraft" else 0.0)  # a per-source offset
        for _ in range(per):
            groups.append(f"g{g}")
            y.append(lab)
            X.append(centre + rng.standard_normal(d))
    return np.array(X), np.array(y), np.array(groups)


def test_make_folds_is_a_leak_free_partition_and_deterministic():
    X, y, g = _toy()
    f1, f2 = make_folds(y, g, 5, 7), make_folds(y, g, 5, 7)
    assert len(f1) == 5
    for (a, b), (c, d) in zip(f1, f2):
        np.testing.assert_array_equal(b, d)
        assert not set(g[a]) & set(g[b])
    assert sorted(np.concatenate([te for _, te in f1]).tolist()) == list(range(len(y)))
    assert any(not np.array_equal(a[1], b[1]) for a, b in zip(f1, make_folds(y, g, 5, 8)))  # the seed matters


def test_check_folds_detects_leaks_missing_classes_and_non_partitions():
    X, y, g = _toy()
    folds = make_folds(y, g, 5, 0)
    leaky = [(np.concatenate([folds[0][0], folds[0][1][:3]]), folds[0][1])] + folds[1:]
    with pytest.raises(ValueError, match="both train and test"):
        check_folds(y, g, leaky)
    one_class_test = [(np.flatnonzero(y == "ambient"), np.flatnonzero(y == "aircraft"))]
    with pytest.raises(ValueError, match="class is missing"):
        check_folds(y, g, one_class_test)
    with pytest.raises(ValueError, match="exactly one"):
        check_folds(y, g, folds[:4])


def test_groups_are_not_split_even_when_classes_are_imbalanced():
    X, y, g = _toy(n_groups=100)
    for tr, te in make_folds(y, g, 5, 1):
        assert not set(g[tr]) & set(g[te])
        assert (y[te] == "aircraft").any() and (y[tr] == "aircraft").any()


def test_foldfeatures_uses_training_statistics_only():
    X, y, g = _toy()
    tr, te = make_folds(y, g, 5, 0)[0]
    a = FoldFeatures(X[tr], X[te])
    b = FoldFeatures(X[tr], X[te] + 100.0)  # a different test set must not change anything learned
    np.testing.assert_array_equal(a.mu, b.mu)
    np.testing.assert_array_equal(a.pca.components_, b.pca.components_)
    np.testing.assert_allclose(a.Ztr.mean(axis=0), 0, atol=1e-10)


def test_foldfeatures_slice_equals_a_pca_fitted_with_k_components():
    X, y, g = _toy()
    tr, te = make_folds(y, g, 5, 0)[0]
    ff = FoldFeatures(X[tr], X[te])
    for k in (1, 4, 12):
        direct = SVDPCA(k).fit(ff.Ztr)
        A, B = ff.view(k)
        np.testing.assert_allclose(A, direct.transform(ff.Ztr), atol=1e-10)
        np.testing.assert_allclose(B, direct.transform(ff.Zte), atol=1e-10)
    A, B = ff.view(None)
    assert A.shape[1] == 12 and ff.view(99)[0].shape[1] == 12


def test_constant_feature_does_not_break_foldfeatures():
    X, y, g = _toy()
    X[:, 3] = 5.0
    tr, te = make_folds(y, g, 5, 0)[0]
    ff = FoldFeatures(X[tr], X[te])
    assert np.isfinite(ff.Ztr).all() and np.isfinite(ff.Pte).all()


def test_fold_metrics_hand_computed():
    y = np.array(["aircraft"] * 4 + ["ambient"] * 6)
    pred = np.array(["aircraft", "aircraft", "aircraft", "ambient", "ambient", "ambient", "ambient", "ambient", "aircraft", "ambient"])
    hn = np.array([False] * 4 + [False, False, False, False, True, True])
    S = np.column_stack([np.linspace(1, -1, 10), np.linspace(-1, 1, 10)])
    m = fold_metrics(y, pred, S, ["aircraft", "ambient"], hn)
    assert m["recall_aircraft"] == pytest.approx(0.75) and m["precision_aircraft"] == pytest.approx(0.75)
    assert m["specificity_plain"] == pytest.approx(1.0) and m["specificity_hard_negative"] == pytest.approx(0.5)
    assert m["balanced_accuracy"] == pytest.approx((0.75 + 5 / 6) / 2)
    assert m["f1_aircraft"] == pytest.approx(0.75) and m["auc_aircraft"] == pytest.approx(1.0)
    assert "ap_ambient" not in m  # AP and AUC are reported for non-ambient classes only


def test_aggregate_means_stds_and_sums_counts():
    a = aggregate([{"n_test": 10.0, "balanced_accuracy": 0.8}, {"n_test": 20.0, "balanced_accuracy": 0.6}])
    assert a["n_test"] == 30 and a["balanced_accuracy_mean"] == pytest.approx(0.7)
    assert a["balanced_accuracy_std"] == pytest.approx(np.std([0.8, 0.6], ddof=1))


def test_group_bootstrap_ci_brackets_the_estimate_and_widens_with_few_groups():
    rng = np.random.default_rng(0)
    classes = ["aircraft", "ambient"]

    def make(n_groups):
        g = np.repeat([f"g{i}" for i in range(n_groups)], 10)
        y = np.where(np.repeat(np.arange(n_groups) % 4 == 0, 10), "aircraft", "ambient")
        pred = np.where(rng.random(len(y)) < 0.8, y, np.where(y == "aircraft", "ambient", "aircraft"))
        return y, pred, g

    wide = group_bootstrap_ci(*make(12), classes, 500, 1)
    narrow = group_bootstrap_ci(*make(400), classes, 500, 1)
    assert wide["balanced_accuracy_hi"] - wide["balanced_accuracy_lo"] > narrow["balanced_accuracy_hi"] - narrow["balanced_accuracy_lo"]
    y, p, g = make(400)
    ba = np.mean([np.mean(p[y == c] == c) for c in classes])
    assert narrow["balanced_accuracy_lo"] < ba < narrow["balanced_accuracy_hi"]
    assert group_bootstrap_ci(y, p, g, classes, 200, 5) == group_bootstrap_ci(y, p, g, classes, 200, 5)


def test_class_scores_alignment_for_binary_and_multiclass():
    X, y, g = _toy()
    m = fit_svm(X, y, {"kernel": "linear", "C": 1.0}, "balanced", 0)
    S = class_scores(m, X)
    pred = m.classes_[np.argmax(S, axis=1)]
    np.testing.assert_array_equal(pred, m.predict(X))
    rng = np.random.default_rng(0)
    y3 = np.array(["a", "b", "c"] * 40)
    X3 = rng.standard_normal((120, 4)) + np.array([[0, 0, 0, 0], [3, 0, 0, 0], [0, 3, 0, 0]])[np.arange(120) % 3]
    m3 = fit_svm(X3, y3, {"kernel": "rbf", "C": 1.0}, None, 0)
    S3 = class_scores(m3, X3)
    assert S3.shape == (120, 3)
    assert np.mean(m3.classes_[np.argmax(S3, axis=1)] == y3) > 0.8


def test_pipeline_and_fast_path_give_identical_scores():
    X, y, g = _toy(d=20)
    names = [f"f{i}" for i in range(20)]
    tr, te = make_folds(y, g, 5, 0)[0]
    hn = np.zeros(len(y), bool)
    for k in (None, 6):
        for svm in ({"kernel": "rbf", "C": 1.0}, {"kernel": "linear", "C": 1.0}):
            ff = FoldFeatures(X[tr], X[te])
            fast_metrics, fast_pred, fast_S = run_fold_config(ff, y[tr], y[te], hn[te], k, svm, "balanced", 3)
            pipe = FeaturePipeline(names, svm, n_components=k, class_weight="balanced", seed=3).fit(X[tr], y[tr])
            np.testing.assert_allclose(pipe.decision(X[te]), fast_S, atol=1e-8)
            np.testing.assert_array_equal(pipe.predict(X[te]), fast_pred)
            assert fast_metrics["balanced_accuracy"] > 0.6


def test_pipeline_drops_level_features_and_validates_width():
    names = ["mfcc00_mean", "rms_mean", "contrast0_mean", "centroid_mean", "zcr_mean"]
    X, y, _ = _toy(d=5)
    p = FeaturePipeline(names, {"kernel": "linear", "C": 1.0}, drop_level_features=True,
                        level_prefixes=["mfcc00_", "rms_", "contrast"]).fit(X, y)
    assert [names[i] for i in p.columns_] == ["centroid_mean", "zcr_mean"]
    # changing a dropped column must not change the output
    X2 = X.copy()
    X2[:, 0] += 50.0
    np.testing.assert_allclose(p.decision(X2), p.decision(X), atol=1e-12)
    with pytest.raises(ValueError):
        p.decision(X[:, :3])
