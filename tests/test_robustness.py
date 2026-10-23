"""Noisy feature caches and the train-clean / test-noisy evaluation."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from sentinel import dataset as dataset_mod
from sentinel import robustness as rb
from sentinel.config import load_config
from sentinel.cv import make_folds
from sentinel.dataset import Dataset, GatedWindows
from sentinel.denoise import denoise
from sentinel.features import clip_features
from sentinel.m3 import cv_run
from sentinel.preprocess import preprocess_clip

CFG = load_config()
SR = CFG["audio"]["sr"]


def _cfg(**over):
    c = copy.deepcopy(CFG)
    c["m3"]["default"].update({"pca": 4, **over})
    return c


# ---- toy data ----------------------------------------------------------------------------------------------------------

def _ds(n_groups=60, per=6, d=8, seed=0):
    rng = np.random.default_rng(seed)
    g, y, X, hn, cat = [], [], [], [], []
    for i in range(n_groups):
        lab = "aircraft" if i % 5 == 0 else "ambient"
        centre = rng.standard_normal(d) * 0.4 + (1.5 if lab == "aircraft" else 0.0)
        for _ in range(per):
            g.append(f"g{i}"), y.append(lab), hn.append(lab == "ambient" and i % 4 == 0)
            cat.append("engine" if i % 4 == 0 else "rain"), X.append(centre + rng.standard_normal(d))
    names = ["mfcc00_mean", "rms_mean", "contrast0_mean"] + [f"f{i}" for i in range(d - 3)]
    return Dataset("none", np.array(X), np.array(y), np.array(g), np.array(hn), np.array(cat), names, np.arange(len(y)), {})


class FakeFeats:
    """Stands in for NoisyFeatures: a matrix per (variant, kind, snr), indexed by cache row."""

    def __init__(self, ds, gated_X, mats):
        n = len(ds.y)
        self.clean = np.vstack([ds.X, gated_X])
        assert n == len(ds.X)
        self.mats = mats

    def get(self, variant, kind, snr):
        return self.clean if kind is None else self.mats[(kind, snr)]


def _setup(shift_noisy=0.0):
    ds = _ds()
    n = len(ds.y)
    rng = np.random.default_rng(5)
    gX = rng.standard_normal((12, ds.X.shape[1]))
    gated = GatedWindows(row_id=np.arange(n, n + 12), groups=np.array([f"g{i}" for i in range(12)]),
                         category=np.array(["rain"] * 12))
    mats = {("white", -5.0): np.vstack([ds.X, gX]) + shift_noisy, ("pink", -5.0): np.vstack([ds.X, gX]) + 2 * shift_noisy}
    return ds, gated, FakeFeats(ds, gX, mats)


# ---- evaluation --------------------------------------------------------------------------------------------------------

def test_clean_condition_reproduces_the_plain_cv_result():
    cfg = _cfg()
    ds, gated, feats = _setup()
    folds = make_folds(ds.y, ds.groups, 5, 0)
    recs = rb.evaluate_conditions(cfg, ds, gated, folds, "none", feats, [rb.CLEAN], rb.clean_train(ds), 50, 0, "t")
    ref = cv_run(ds.X, ds, folds, 4, cfg["m3"]["default"]["svm"], cfg["m3"]["class_weight"], cfg["seed"])
    assert recs[0]["kind"] == "clean" and recs[0]["snr_db"] is None
    assert recs[0]["ba_fold_mean"] == pytest.approx(ref["aggregate"]["balanced_accuracy_mean"], abs=1e-9)
    assert recs[0]["ba_folds"] == pytest.approx([m["balanced_accuracy"] for m in ref["folds"]], abs=1e-9)


def test_noisy_condition_changes_only_the_test_side():
    cfg = _cfg()
    ds, gated, feats = _setup(shift_noisy=3.0)
    folds = make_folds(ds.y, ds.groups, 5, 0)
    conds = [rb.CLEAN, ("white", -5.0)]
    recs = rb.evaluate_conditions(cfg, ds, gated, folds, "none", feats, conds, rb.clean_train(ds), 50, 0, "t")
    clean, noisy = recs
    assert clean["ba_fold_mean"] > 0.85 and noisy["ba_fold_mean"] < clean["ba_fold_mean"] - 0.1
    # a pure shift of every feature by +3 (far above the unit noise) pushes everything to one side: recall or specificity collapses
    assert min(noisy["recall_aircraft"], noisy["specificity_plain"]) < 0.5
    assert 0.0 <= noisy["noise_only_far"] <= 1.0 and noisy["n_noise_only"] == 12


def test_fit_sees_only_the_matrix_given_by_train_X():
    cfg = _cfg()
    ds, gated, feats = _setup()
    folds = make_folds(ds.y, ds.groups, 5, 0)
    seen = []

    def spy(i, tr):
        seen.append((i, tr.copy()))
        return ds.X[tr]

    rb.evaluate_conditions(cfg, ds, gated, folds, "none", feats, [rb.CLEAN], spy, 20, 0, "t")
    assert [i for i, _ in seen] == list(range(5))
    for (i, tr), (tr0, te0) in zip(seen, folds):
        assert np.array_equal(tr, tr0) and not set(ds.groups[tr]) & set(ds.groups[te0])


def test_augmented_training_never_contains_the_held_out_noise_kind():
    ds, gated, feats = _setup(shift_noisy=1000.0)  # white rows are +1000, pink rows +2000, clean rows ~unit scale
    folds = make_folds(ds.y, ds.groups, 5, 0)
    fn = rb.augmented_train(ds, feats, "none", "white", ["white", "pink"], [-5.0], 0)
    mixed = False
    for i, (tr, _) in enumerate(folds):
        X = fn(i, tr)
        assert X.shape == ds.X[tr].shape and np.array_equal(X, fn(i, tr))  # same size, deterministic
        row_offset = np.round((X - ds.X[tr]).mean(axis=1) / 1000.0)
        assert set(np.unique(row_offset)) <= {0.0, 2.0}  # clean or pink, never white (offset 1000)
        for g in np.unique(ds.groups[tr]):
            assert len(np.unique(row_offset[ds.groups[tr] == g])) == 1  # one condition per recording
        mixed |= len(np.unique(row_offset)) == 2
    assert mixed


def test_gated_fold_uses_the_heldout_fold_and_a_fixed_fallback_for_unseen_groups():
    ds = _ds()
    folds = make_folds(ds.y, ds.groups, 5, 0)
    te_fold_of_g0 = next(i for i, (_, te) in enumerate(folds) if "g0" in set(ds.groups[te]))
    out = rb.gated_fold(np.array(["g0", "never_seen_b", "never_seen_a"]), ds.groups, folds)
    assert out[0] == te_fold_of_g0
    assert out[1] == 1 and out[2] == 0  # sorted order of the unknown groups, modulo the fold count


def test_paired_against_counts_folds():
    rows = []
    for v, base in (("none", [0.70, 0.70, 0.70]), ("wiener", [0.72, 0.69, 0.75])):
        rows.append({"kind": "white", "snr_db": -5.0, "variant": v, "ba_folds": base})
    t = rb.paired_against(pd.DataFrame(rows))
    r = t.iloc[0]
    assert r["variant"] == "wiener" and r["mean_diff"] == pytest.approx(0.02) and r["folds_better"] == 2 and r["n_folds"] == 3


# ---- noisy caches (small synthetic recordings in a temporary project root) -----------------------------------------------

@pytest.fixture()
def tiny(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dataset_mod, "PROJECT_ROOT", tmp_path)
    cfg = copy.deepcopy(CFG)
    cfg["m4"]["robustness"]["cache_dir"] = "noisy"
    rng = np.random.default_rng(0)
    rows = []
    for i in range(3):
        n = int(SR * (2.0 + 0.5 * i))
        t = np.arange(n) / SR
        x = 0.1 * np.sin(2 * np.pi * (300 + 200 * i) * t) + 0.02 * rng.standard_normal(n)
        p = tmp_path / f"c{i}.wav"
        sf.write(p, x, SR, subtype="FLOAT")
        rows.append({"path": str(p), "dataset": "esc50", "label": "aircraft" if i == 0 else "ambient", "group_id": f"g{i}",
                     "duration_s": n / SR, "sr": SR, "hard_negative": False, "category": "x"})
    return cfg, pd.DataFrame(rows), tmp_path


def test_process_clip_noisy_at_negligible_noise_equals_the_clean_pipeline():
    rng = np.random.default_rng(1)
    x = (0.1 * np.sin(2 * np.pi * 440 * np.arange(SR * 3) / SR) + 0.02 * rng.standard_normal(SR * 3)).astype(np.float32)
    p_ref = 0.005
    out, starts, snr = rb.process_clip_noisy(x, 0, "g", "white", [(300.0, "none"), (300.0, "svd_elbow")], p_ref, None, CFG)
    for v in ("none", "svd_elbow"):
        y = denoise(preprocess_clip(x, SR, CFG), v, SR, CFG)[0]
        X, st = clip_features(y, SR, CFG)
        np.testing.assert_allclose(out[(300.0, v)], X, rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(starts, st)
    assert snr[300.0] > 250  # nominal 300 dB against p_ref, the clip is far louder than the noise


def test_build_noisy_writes_aligned_caches_and_resumes(tiny):
    cfg, man, root = tiny
    p_ref = 0.004
    s1 = rb.build_noisy(cfg, man, "white", [5.0, -5.0], ["none"], p_ref, None, n_jobs=1)
    assert sorted(s1["built"]) == ["none@+5", "none@-5"]
    for snr in (5.0, -5.0):
        z = np.load(rb.noisy_path(cfg, "white", snr, "none"))
        assert z["X"].shape == (3 + 4 + 5, 146)  # 2.0, 2.5, 3.0 s clips give 3, 4, 5 windows
        assert list(np.unique(z["file_index"])) == [0, 1, 2] and len(z["clip_snr_db"]) == 3
    s2 = rb.build_noisy(cfg, man, "white", [5.0, -5.0], ["none"], p_ref, None, n_jobs=1)
    assert s2["built"] == []                                    # fresh caches are reused
    s3 = rb.build_noisy(cfg, man, "white", [5.0, -5.0], ["none"], p_ref * 2, None, n_jobs=1)
    assert len(s3["built"]) == 2                                # another reference level makes them stale
    s4 = rb.build_noisy(cfg, man, "white", [5.0, 0.0], ["none"], p_ref * 2, None, n_jobs=1)
    assert s4["built"] == ["none@+0"]                           # adding one SNR builds only that one
    a, b = np.load(rb.noisy_path(cfg, "white", 5.0, "none"))["X"], np.load(rb.noisy_path(cfg, "white", -5.0, "none"))["X"]
    assert not np.allclose(a, b)


def test_noisy_features_reject_a_cache_that_is_not_row_aligned(tiny):
    cfg, man, root = tiny
    fdir = root / cfg["features"]["cache_dir"]
    fdir.mkdir(parents=True)
    names = np.array([f"f{i}" for i in range(146)])
    np.savez(fdir / "features_none.npz", X=np.zeros((4, 146), np.float32), file_index=np.array([0, 0, 1, 1]),
             t_start=np.array([0, 0.5, 0, 0.5], np.float32), feature_names=names, label=np.array(["a"] * 4),
             group_id=np.array(["g"] * 4), dataset=np.array(["esc50"] * 4), hard_negative=np.zeros(4, bool),
             category=np.array(["x"] * 4))
    out = rb.noisy_path(cfg, "white", 5.0, "none")
    out.parent.mkdir(parents=True)
    np.savez(out, X=np.zeros((4, 146), np.float32), file_index=np.array([0, 0, 1, 1]),
             t_start=np.array([0, 0.25, 0, 0.5], np.float32), cache_key=np.array("k"))
    nf = rb.NoisyFeatures(cfg)
    with pytest.raises(ValueError, match="not row-aligned"):
        nf.get("none", "white", 5.0)
    with pytest.raises(FileNotFoundError, match="make robustness"):
        nf.get("none", "pink", 5.0)
    assert nf.get("none", None, None).shape == (4, 146)


def test_noisy_key_depends_on_everything_the_cache_depends_on(tiny):
    cfg, man, _ = tiny
    pool = [("a", "p1")]
    base = rb.noisy_key(cfg, "ambient", 5.0, "none", man, 0.01, pool)
    assert base == rb.noisy_key(cfg, "ambient", 5.0, "none", man, 0.01, pool)
    assert base != rb.noisy_key(cfg, "pink", 5.0, "none", man, 0.01, pool)
    assert base != rb.noisy_key(cfg, "ambient", 0.0, "none", man, 0.01, pool)
    assert base != rb.noisy_key(cfg, "ambient", 5.0, "wiener", man, 0.01, pool)
    assert base != rb.noisy_key(cfg, "ambient", 5.0, "none", man, 0.02, pool)
    assert base != rb.noisy_key(cfg, "ambient", 5.0, "none", man, 0.01, pool + [("b", "p2")])
    c2 = copy.deepcopy(cfg)
    c2["features"]["n_mfcc"] = 13
    assert base != rb.noisy_key(c2, "ambient", 5.0, "none", man, 0.01, pool)
