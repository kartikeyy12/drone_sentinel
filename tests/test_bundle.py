"""The saved model: build from out-of-fold scores, round trip, compatibility check, gating, offline scoring, latency helpers."""
from __future__ import annotations

import copy
import json
import warnings

import numpy as np
import pytest

from sentinel import bundle as bd
from sentinel.calibration import threshold_for_far
from sentinel.config import config_hash, load_config
from sentinel.cv import make_folds
from sentinel.dataset import Dataset
from sentinel.denoise import denoise
from sentinel.features import clip_features, feature_names, window_features, window_rms_mean
from sentinel.m3 import cv_run
from sentinel.preprocess import preprocess_clip
from sentinel.windowing import window_signal

CFG0 = load_config()
SR = CFG0["audio"]["sr"]


def _cfg(pca=4, **final):
    c = copy.deepcopy(CFG0)
    c["m4"]["final"]["config"] = {"variant": "none", "drop_level_features": False, "pca": pca,
                                  "svm": {"kernel": "rbf", "C": 1.0}, **final}
    return c


def _toy_ds(n_groups=60, per=6, d=8, seed=0):
    rng = np.random.default_rng(seed)
    g, y, X, hn, cat = [], [], [], [], []
    for i in range(n_groups):
        lab = "aircraft" if i % 5 == 0 else "ambient"
        centre = rng.standard_normal(d) * 0.4 + (1.5 if lab == "aircraft" else 0.0)
        for _ in range(per):
            g.append(f"g{i}"), y.append(lab), hn.append(False), cat.append("x"), X.append(centre + rng.standard_normal(d))
    names = ["mfcc00_mean", "rms_mean", "contrast0_mean"] + [f"f{i}" for i in range(d - 3)]
    return Dataset("none", np.array(X), np.array(y), np.array(g), np.array(hn), np.array(cat), names, np.arange(len(y)), {})


def _toy_model(cfg=None):
    cfg = cfg or _cfg()
    ds = _toy_ds()
    run = cv_run(ds.X, ds, make_folds(ds.y, ds.groups, 5, 0), 4, {"kernel": "rbf", "C": 1.0}, "balanced", 0)
    return cfg, ds, run, bd.build_model(cfg, ds, run["scores"], run["classes"])


def test_build_model_calibrates_on_out_of_fold_scores_and_sets_the_far_threshold():
    cfg, ds, run, m = _toy_model()
    assert m.classes == ["aircraft", "ambient"] and m.targets == ["aircraft"]
    p_oof = m.calibrators["aircraft"].predict(run["scores"][:, 0])
    amb = ds.y == "ambient"
    assert np.mean(p_oof[amb] >= m.thresholds["aircraft"]) <= cfg["m4"]["operating"]["target_window_far"]
    assert m.thresholds["aircraft"] == threshold_for_far(p_oof[amb], cfg["m4"]["operating"]["target_window_far"])
    p = m.probabilities(ds.X)
    assert p.shape == (len(ds.y), 1) and p.min() > 0 and p.max() < 1
    assert p[ds.y == "aircraft"].mean() > p[amb].mean() + 0.2          # it separates the classes
    S = m.pipeline.decision(ds.X)[:, 0]
    assert np.all(np.diff(p[np.argsort(S), 0]) >= -1e-12)               # monotone in the score


def test_the_final_fit_uses_every_window_but_calibration_never_sees_in_sample_scores():
    cfg, ds, run, m = _toy_model()
    assert m.extra["n_training_windows"] == len(ds.y)
    in_sample = m.pipeline.decision(ds.X)[:, 0]
    # in-sample scores are more separated than out-of-fold ones, which is why calibration must use out-of-fold scores
    sep = lambda s: s[ds.y == "aircraft"].mean() - s[ds.y == "ambient"].mean()
    assert sep(in_sample) > sep(run["scores"][:, 0])


def test_build_model_rejects_mismatched_out_of_fold_inputs():
    cfg, ds, run, _ = _toy_model()
    with pytest.raises(ValueError, match="class order"):
        bd.build_model(cfg, ds, run["scores"], ["ambient", "aircraft"])
    with pytest.raises(ValueError, match="do not match"):
        bd.build_model(cfg, ds, run["scores"][:-1], run["classes"])
    c2 = copy.deepcopy(cfg)
    c2["m4"]["positive_classes"] = ["drone"]
    with pytest.raises(ValueError):
        bd.build_model(c2, ds, run["scores"], run["classes"])


def test_final_config_defaults_to_m3_default_and_can_be_overridden():
    assert bd.final_config(CFG0) is CFG0["m3"]["default"]
    assert bd.final_config(_cfg(pca=7))["pca"] == 7


def test_save_load_roundtrip_gives_identical_probabilities(tmp_path):
    cfg, ds, run, m = _toy_model()
    path = tmp_path / "models" / "pipeline.joblib"
    size = bd.save_model(m, path)
    assert size == path.stat().st_size > 1000
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["classes"] == ["aircraft", "ambient"] and meta["config_hash"] == config_hash(cfg)
    assert meta["thresholds"]["aircraft"] == pytest.approx(m.thresholds["aircraft"])
    m2 = bd.load_model(path, cfg)
    np.testing.assert_array_equal(m.probabilities(ds.X), m2.probabilities(ds.X))
    fp = bd.model_footprint(m2, path)
    assert fp["joblib_file_bytes"] == size and fp["array_bytes_total"] > 0 and sum(fp["n_support_vectors"]) > 0


def test_load_rejects_a_config_whose_features_differ_but_not_one_that_only_changes_the_operating_point(tmp_path):
    cfg, ds, run, m = _toy_model()
    path = tmp_path / "p.joblib"
    bd.save_model(m, path)
    bad = copy.deepcopy(cfg)
    bad["preprocess"]["bandpass"]["low_hz"] = 150
    with pytest.raises(ValueError, match="make final"):
        bd.load_model(path, bad)
    bad = copy.deepcopy(cfg)
    bad["features"]["n_mfcc"] = 13
    with pytest.raises(ValueError, match="config changed"):
        bd.load_model(path, bad)
    ok = copy.deepcopy(cfg)
    ok["m4"]["operating"]["target_window_far"] = 0.01
    ok["m4"]["robustness"]["snr_db"] = [3]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bd.load_model(path, ok)  # same features: loads
    with pytest.raises(ValueError, match="does not contain"):
        import joblib
        joblib.dump({"not": "a model"}, tmp_path / "x.joblib")
        bd.load_model(tmp_path / "x.joblib")


def test_gate_zeroes_probability_and_blocks_alerts_labels_pick_the_strongest_class():
    cfg, ds, run, m = _toy_model()
    p, gated = m.score_features(ds.X[:6], np.array([1.0, 1e-5, 1.0, 1.0, 1e-9, 1.0]))
    assert gated.tolist() == [False, True, False, False, True, False]
    assert np.all(p[gated] == 0.0) and np.all(p[~gated] > 0)
    forced = np.full((4, 1), 0.99)
    assert m.alerts(forced).all() and not m.alerts(forced, np.array([0, 1, 0, 1], bool))[[1, 3]].any()
    assert m.labels(forced, np.array([0, 1, 0, 1], bool)).tolist() == ["aircraft", "ambient", "aircraft", "ambient"]
    assert m.labels(np.full((2, 1), 1e-6)).tolist() == ["ambient", "ambient"]


def test_multiclass_labels_choose_the_alerting_class_with_the_highest_probability():
    cfg, ds, run, m = _toy_model()
    m.targets, m.thresholds = ["aircraft", "drone"], {"aircraft": 0.5, "drone": 0.4}
    p = np.array([[0.9, 0.6], [0.6, 0.9], [0.3, 0.2], [0.3, 0.45]])
    assert m.labels(p).tolist() == ["aircraft", "drone", "ambient", "drone"]


# ---- the real feature chain (small synthetic recordings) ---------------------------------------------------------------

def _clip(kind: str, seed: int, dur=4.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(SR * dur)
    t = np.arange(n) / SR
    if kind == "aircraft":
        x = sum(0.05 / k * np.sin(2 * np.pi * 150 * k * t + rng.random() * 6) for k in range(1, 12)) + 0.01 * rng.standard_normal(n)
    else:
        x = rng.uniform(0.02, 0.08) * rng.standard_normal(n)
    return x.astype(np.float32)


@pytest.fixture(scope="module")
def real_model():
    cfg = _cfg(pca=10)
    names = feature_names(cfg["features"])
    Xs, ys, gs = [], [], []
    for i in range(16):
        kind = "aircraft" if i % 4 == 0 else "ambient"
        y = preprocess_clip(_clip(kind, i), SR, cfg)
        X, _ = clip_features(y, SR, cfg)
        Xs.append(X)
        ys += [kind] * len(X)
        gs += [f"g{i}"] * len(X)
    X = np.vstack(Xs).astype(np.float64)
    ds = Dataset("none", X, np.array(ys), np.array(gs), np.zeros(len(ys), bool), np.array(["x"] * len(ys)), names,
                 np.arange(len(ys)), {})
    run = cv_run(ds.X, ds, make_folds(ds.y, ds.groups, 4, 0), 10, {"kernel": "rbf", "C": 1.0}, "balanced", 0)
    return cfg, bd.build_model(cfg, ds, run["scores"], run["classes"])


def test_window_rms_mean_is_the_rms_mean_feature():
    names = feature_names(CFG0["features"])
    y = preprocess_clip(_clip("aircraft", 3, 2.0), SR, CFG0)
    wins, _ = window_signal(y, SR, 1.0, 0.5)
    for w in wins:
        f = window_features(w, SR, CFG0["features"])
        assert window_rms_mean(w, CFG0["features"]) == pytest.approx(float(f[names.index("rms_mean")]), rel=1e-6)


def test_score_audio_matches_the_feature_cache_chain_and_gates_silence(real_model):
    cfg, m = real_model
    x = _clip("aircraft", 99, 3.0)
    out = m.score_audio(x)
    y = preprocess_clip(x, SR, cfg)
    X, starts = clip_features(denoise(y, "none", SR, cfg)[0], SR, cfg)
    np.testing.assert_array_equal(out["t_start"], starts)
    np.testing.assert_allclose(out["p"], m.probabilities(X), rtol=1e-12)
    assert not out["gated"].any() and len(out["label"]) == len(starts) == 5
    silent = m.score_audio(np.zeros(SR * 3, np.float32))
    assert silent["gated"].all() and np.all(silent["p"] == 0) and set(silent["label"]) == {"ambient"}
    assert len(m.score_audio(np.zeros(SR // 2, np.float32))["p"]) == 0          # shorter than one window: nothing to score


def test_the_synthetic_target_is_separated_from_the_synthetic_background(real_model):
    cfg, m = real_model
    pa = np.mean([m.score_audio(_clip("aircraft", 200 + i, 3.0))["p"].mean() for i in range(4)])
    pn = np.mean([m.score_audio(_clip("ambient", 300 + i, 3.0))["p"].mean() for i in range(4)])
    assert pa > 0.5 > pn


def test_benchmark_stages_and_summary(real_model):
    cfg, m = real_model
    y = preprocess_clip(_clip("aircraft", 5, 5.0), SR, cfg)
    wins, _ = window_signal(y, SR, 1.0, 0.5)
    st = bd.benchmark_stages(m, wins, warmup=2)
    assert all(len(v) == len(wins) - 2 for v in st.values()) and all((v > 0).all() for v in st.values())
    np.testing.assert_allclose(st["total"], st["features"] + st["decision"] + st["calibration"])
    np.testing.assert_allclose(st["total_with_gate"], st["total"] + st["gate_rms"])
    s = bd.stage_summary(st, 0.5)
    assert s["total"]["median_ms"] <= s["total"]["p95_ms"] <= s["total"]["max_ms"]
    assert s["total"]["share_of_hop_median"] == pytest.approx(s["total"]["median_ms"] / 500.0)
