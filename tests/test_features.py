"""Feature vector layout and sanity, windowing, and the npz cache."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import sentinel.features as feat
from sentinel.config import load_config
from sentinel.features import build_feature_cache, cache_key, clip_features, feature_names, window_features
from sentinel.windowing import window_signal

SR = 16000
CFG = load_config()
FC = CFG["features"]


def _tone(f, secs=1.0, amp=0.3):
    t = np.arange(int(SR * secs)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_names_match_vector():
    v = window_features(_tone(1000), SR, FC)
    names = feature_names(FC)
    assert v.shape == (len(names),) == (146,) and len(set(names)) == 146
    assert v.dtype == np.float32 and np.isfinite(v).all()


def test_deterministic():
    x = _tone(700)
    np.testing.assert_array_equal(window_features(x, SR, FC), window_features(x, SR, FC))


def test_centroid_tracks_tone_frequency():
    d = lambda f: dict(zip(feature_names(FC), window_features(_tone(f), SR, FC)))  # noqa: E731
    assert abs(d(1000)["centroid_mean"] - 1000) < 150
    assert d(3000)["centroid_mean"] > d(1000)["centroid_mean"] + 1500


def test_flatness_noise_above_tone():
    rng = np.random.default_rng(0)
    names = feature_names(FC)
    fl = names.index("flatness_mean")
    noise = (0.1 * rng.standard_normal(SR)).astype(np.float32)
    assert window_features(noise, SR, FC)[fl] > 10 * window_features(_tone(1000), SR, FC)[fl]


def test_rms_scales_with_amplitude():
    names = feature_names(FC)
    r = names.index("rms_mean")
    a, b = window_features(_tone(1000, amp=0.1), SR, FC)[r], window_features(_tone(1000, amp=0.2), SR, FC)[r]
    assert abs(b / a - 2) < 0.05


def test_silence_is_finite():
    assert np.isfinite(window_features(np.zeros(SR, np.float32), SR, FC)).all()


def test_window_signal_layout():
    x = np.arange(SR * 3, dtype=np.float32)
    w, t = window_signal(x, SR, 1.0, 0.5)
    assert w.shape == (5, SR) and list(t) == [0, 0.5, 1.0, 1.5, 2.0]
    assert w[1, 0] == SR // 2 and w[-1, -1] == len(x) - 1
    assert window_signal(x[: SR - 1], SR)[0].shape[0] == 0


def test_clip_features_counts():
    X, starts = clip_features(_tone(500, 3.0), SR, CFG)
    assert X.shape == (5, 146) and len(starts) == 5
    assert clip_features(_tone(500, 0.5), SR, CFG)[0].shape == (0, 146)


@pytest.fixture
def manifest(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i, (label, dur) in enumerate([("aircraft", 2.0), ("ambient", 3.0), ("ambient", 0.6)]):
        p = tmp_path / f"c{i}.wav"
        sf.write(str(p), (0.1 * rng.standard_normal(int(SR * dur))).astype(np.float32), SR)
        rows.append({"path": str(p), "dataset": "t", "label": label, "group_id": f"g{i}", "duration_s": dur,
                     "sr": SR, "hard_negative": False, "category": label})
    return pd.DataFrame(rows)


def test_cache_build_hit_and_content(manifest, tmp_path):
    out = tmp_path / "f.npz"
    s1 = build_feature_cache(CFG, manifest, "none", out, n_jobs=1)
    assert s1["cache"] == "built" and s1["n_windows"] == 3 + 5 + 0
    assert s1["files_without_a_full_window"] == 1
    assert s1["windows_by_class"] == {"aircraft": 3, "ambient": 5}
    with np.load(out) as z:
        assert z["X"].shape == (8, 146) and list(z["feature_names"]) == feature_names(FC)
        assert list(z["label"]) == ["aircraft"] * 3 + ["ambient"] * 5
        assert list(z["group_id"][:3]) == ["g0"] * 3 and z["t_start"][1] == 0.5
    assert build_feature_cache(CFG, manifest, "none", out, n_jobs=1)["cache"] == "hit"


def test_cache_not_recomputed_on_hit(manifest, tmp_path, monkeypatch):
    out = tmp_path / "f.npz"
    build_feature_cache(CFG, manifest, "none", out, n_jobs=1)

    def boom(*a, **k):
        raise AssertionError("recomputed")

    monkeypatch.setattr(feat, "process_file", boom)
    assert build_feature_cache(CFG, manifest, "none", out, n_jobs=1)["cache"] == "hit"


def test_cache_invalidation(manifest):
    base = cache_key(CFG, "none", manifest)
    c2 = copy.deepcopy(CFG)
    c2["features"]["n_mfcc"] = 13
    assert cache_key(c2, "none", manifest) != base
    c3 = copy.deepcopy(CFG)
    c3["features"]["n_jobs"] = 1  # runtime-only setting must not invalidate
    assert cache_key(c3, "none", manifest) == base
    assert cache_key(CFG, "specsub", manifest) != base
    c4 = copy.deepcopy(CFG)
    c4["denoise"]["wiener"]["gain_floor"] = 0.3  # irrelevant to specsub
    assert cache_key(c4, "specsub", manifest) == cache_key(CFG, "specsub", manifest)
    assert cache_key(c4, "wiener", manifest) != cache_key(CFG, "wiener", manifest)
    assert cache_key(CFG, "none", manifest.iloc[:2]) != base


def test_all_variants_build(manifest, tmp_path):
    for v in ("specsub", "wiener", "svd_energy", "svd_elbow"):
        s = build_feature_cache(CFG, manifest, v, tmp_path / f"{v}.npz", n_jobs=1)
        assert s["n_windows"] == 8 and np.isfinite(np.load(tmp_path / f"{v}.npz")["X"]).all()
