"""Helpers behind `make stream-check`: causal features and their cache, the CausalFeatures adapter, SNR mixtures and summaries."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from helpers_stream import SR, noise, tone_stack
from sentinel import robustness as rb
from sentinel import streamcheck as sc
from sentinel.config import load_config
from sentinel.features import window_features
from sentinel.noise import inband_power
from sentinel.preprocess import StreamingBandpass
from sentinel.windowing import n_windows

CFG = load_config()


def test_causal_filter_equals_the_streaming_filter_in_any_chunking():
    x = tone_stack(2.0) + noise(2.0)
    bp = StreamingBandpass(sc.causal_sos(CFG))
    chunked = np.concatenate([bp.process(x[i:i + 777]) for i in range(0, len(x), 777)])
    np.testing.assert_allclose(sc.causal_filter(x, CFG), chunked, atol=1e-6)


def test_causal_clip_features_cover_every_window_and_match_the_single_window_computation():
    x = tone_stack(3.0) + noise(3.0)
    X, starts = sc.causal_clip_features(x, CFG)
    assert len(X) == len(starts) == n_windows(3.0) and starts[1] == pytest.approx(0.5)
    y = sc.causal_filter(x, CFG)
    np.testing.assert_allclose(X[2], window_features(y[SR:2 * SR], SR, CFG["features"]), atol=1e-5)


def _fake_clean_cache(files, cfg):
    """What load_cache(cfg, 'none') would return for these clips: file_index and t_start per window."""
    fi, ts = [], []
    for i, f in enumerate(files):
        n = n_windows(len(sf.read(f)[0]) / SR)
        fi += [i] * n
        ts += list(np.arange(n) * 0.5)
    return {"file_index": np.array(fi, np.int32), "t_start": np.array(ts, np.float32)}


@pytest.fixture
def mini(tmp_path, monkeypatch):
    cfg = copy.deepcopy(CFG)
    cfg["m5"]["check"]["cache_dir"] = str(tmp_path / "causal")
    files = []
    for i, x in enumerate([tone_stack(3.0, seed=1), noise(2.5, seed=2)]):
        p = tmp_path / f"c{i}.wav"
        sf.write(p, x, SR)
        files.append(str(p))
    man = pd.DataFrame({"path": files, "label": ["aircraft", "ambient"], "group_id": ["a", "b"], "dataset": "esc50"})
    ref = _fake_clean_cache(files, cfg)
    monkeypatch.setattr(sc, "load_cache", lambda c, v: ref)
    monkeypatch.setattr(rb, "load_cache", lambda c, v: {**ref, "X": np.zeros((len(ref["file_index"]), 3))})
    return cfg, man, ref


def test_build_causal_is_row_aligned_and_reused_until_something_changes(mini):
    cfg, man, ref = mini
    out = sc.build_causal(cfg, man, n_jobs=1)
    assert out["built"] is True
    with np.load(sc.causal_path(cfg)) as z:
        assert z["X"].shape == (len(ref["file_index"]), 146) and np.array_equal(z["file_index"], ref["file_index"])
    assert sc.build_causal(cfg, man, n_jobs=1)["built"] is False
    assert sc.build_causal(cfg, man, n_jobs=1, force=True)["built"] is True
    changed = copy.deepcopy(cfg)
    changed["preprocess"]["bandpass"]["low_hz"] = 150
    assert sc.causal_key(changed, man) != sc.causal_key(cfg, man)
    assert sc.build_causal(changed, man, n_jobs=1)["built"] is True


def test_build_causal_refuses_a_clean_cache_that_does_not_line_up(mini, monkeypatch):
    cfg, man, ref = mini
    bad = {"file_index": ref["file_index"][:-1], "t_start": ref["t_start"][:-1]}
    monkeypatch.setattr(sc, "load_cache", lambda c, v: bad)
    with pytest.raises(ValueError, match="row-aligned"):
        sc.build_causal(cfg, man, n_jobs=1)


def test_causal_features_adapter_serves_the_causal_matrix_and_checks_alignment(mini):
    cfg, man, ref = mini
    sc.build_causal(cfg, man, n_jobs=1)
    f = sc.CausalFeatures(cfg)
    Xc = f.get("none", "causal", None)
    assert Xc.shape == (len(ref["file_index"]), 146) and Xc.dtype == np.float64
    assert f.get("none", "causal", None) is Xc                                    # cached
    with pytest.raises(ValueError, match="`none` variant"):
        f.get("wiener", "causal", None)
    with np.load(sc.causal_path(cfg)) as z:                                       # corrupt the alignment
        d = {k: z[k] for k in z.files}
    d["t_start"] = d["t_start"][::-1].copy()
    np.savez(sc.causal_path(cfg), **d)
    with pytest.raises(ValueError, match="not row-aligned"):
        sc.CausalFeatures(cfg).get("none", "causal", None)
    sc.causal_path(cfg).unlink()
    with pytest.raises(FileNotFoundError, match="stream-check"):
        sc.CausalFeatures(cfg).get("none", "causal", None)


# ---- SNR ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("snr", [0.0, 10.0, 20.0])
def test_snr_scene_sets_the_clip_power_relative_to_the_noise_power(snr):
    clip = tone_stack(5.0, seed=3)
    x, start, c = sc.snr_scene(clip, snr, CFG, seed=1, noise_power=1e-4)
    assert start == 30 * SR and len(x) == len(clip) + 45 * SR
    assert 10 * np.log10(np.mean(c ** 2) / 1e-4) == pytest.approx(snr, abs=0.6)    # causal filter vs the zero-phase reference
    assert 10 * np.log10(inband_power(x[:start - SR], SR, CFG) / 1e-4) == pytest.approx(0.0, abs=1.5)   # the lead is noise only


def test_snr_scene_is_deterministic_in_the_seed():
    clip = tone_stack(2.0)
    a = sc.snr_scene(clip, 5.0, CFG, seed=4)[0]
    assert np.array_equal(a, sc.snr_scene(clip, 5.0, CFG, seed=4)[0]) and not np.array_equal(a, sc.snr_scene(clip, 5.0, CFG, seed=5)[0])


def test_snr_estimate_is_close_to_the_truth_on_a_known_mixture(toy):
    cfg, model = toy
    clip = tone_stack(6.0, seed=5)
    r = sc.snr_estimate_errors(model, cfg, clip, 10.0, seed=2)
    assert len(r["estimate"]) == len(r["truth"]) == len(r["flagged"]) == 11         # (6 - 1) / 0.5 + 1 windows lie entirely inside a 6 s clip
    s = sc.summarize_snr_errors([r])
    assert s["n_windows"] > 5 and abs(s["bias_db"]) < 2.5 and s["mae_db"] < 3.0 and r["flagged"].all()


def test_summarize_snr_errors_filters_the_truth_range_and_counts_missing_estimates():
    r = {"estimate": np.array([10.0, 12.0, np.nan, 5.0, -20.0]), "truth": np.array([9.0, 10.0, 8.0, 40.0, -15.0]),
         "flagged": np.zeros(5, bool)}
    s = sc.summarize_snr_errors([r])
    # kept: windows 0, 1 (finite estimate, truth in range) and window 4 (truth -15 is below -10: dropped)
    assert s["n_windows"] == 2 and s["n_no_estimate"] == 1 and s["bias_db"] == pytest.approx(1.5) and s["mae_db"] == pytest.approx(1.5)
    assert sc.summarize_snr_errors([{"estimate": np.array([np.nan]), "truth": np.array([1.0]), "flagged": np.zeros(1, bool)}]) == {"n_windows": 0}


def test_engine_speed_reports_keep_up_on_a_short_signal(toy):
    cfg, model = toy
    out = sc.engine_speed(model, cfg, tone_stack(6.0) + noise(6.0))
    assert out["n_windows"] == 11 and out["hop_ms"] == 500.0 and out["window_ms_median"] > 0
    assert out["real_time_factor"] < 1.0 and out["window_ms_max"] >= out["window_ms_median"]
