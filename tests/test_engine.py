"""Streaming engine: chunking invariance, equivalence with batch causal scoring, event records, gating, noise floor."""
from __future__ import annotations

import copy
import re
from datetime import datetime

import numpy as np
import pytest
from scipy import signal

from helpers_stream import SR, noise, scene, tone_stack
from sentinel import alerting as al
from sentinel.engine import NoiseFloor, StreamEngine, run_offline, utc_iso
from sentinel.features import feature_names, window_features
from sentinel.preprocess import bandpass_sos
from sentinel.windowing import n_windows, window_signal

SPEC_KEYS = ["sensor_id", "event_id", "label", "t_start", "t_end", "duration_s", "peak_prob", "mean_prob", "top_alt_class",
             "snr_est_db", "model_version"]


def p_of(wins):
    return np.array([w.p["aircraft"] for w in wins])


def test_utc_iso_formats_milliseconds_and_rounds():
    assert utc_iso(0.0) == "1970-01-01T00:00:00.000Z"
    assert utc_iso(1760179503.5) == "2025-10-11T10:45:03.500Z"
    assert utc_iso(1.9996) == "1970-01-01T00:00:02.000Z"


def test_noise_floor_needs_min_frames_and_keeps_only_the_history():
    f = NoiseFloor(percentile=10, history_frames=100, min_frames=50)
    f.update(np.ones(49))
    assert f.value is None
    f.update(np.ones(1))
    assert f.value == pytest.approx(1.0)
    f.update(np.full(100, 4.0))                      # the old frames fall out of the history
    assert f.n_frames == 100 and f.value == pytest.approx(4.0)
    f.update(np.zeros(0))
    assert f.n_frames == 100


def test_window_count_and_times_match_the_offline_windowing(toy):
    cfg, model = toy
    x = scene(4, 4, 4)
    wins, _, eng = run_offline(model, cfg, x)
    assert len(wins) == n_windows(len(x) / SR)
    assert [w.t_start for w in wins] == pytest.approx(list(np.arange(len(wins)) * 0.5))
    assert all(w.t_end - w.t_start == pytest.approx(1.0) for w in wins)
    assert eng.stream_time == pytest.approx(len(x) / SR)


@pytest.mark.parametrize("chunk_s", [1.0, 0.0173, 0.5, 3.3])
def test_results_do_not_depend_on_the_chunk_size(toy, chunk_s):
    cfg, model = toy
    x = scene(3, 5, 3)
    ref_w, ref_r, _ = run_offline(model, cfg, x, chunk_s=0.25)
    w, r, _ = run_offline(model, cfg, x, chunk_s=chunk_s)
    assert len(w) == len(ref_w)
    np.testing.assert_allclose(p_of(w), p_of(ref_w), atol=1e-6)
    assert [(e["label"], e["t_start"], e["t_end"]) for e in r] == [(e["label"], e["t_start"], e["t_end"]) for e in ref_r]


def test_engine_equals_batch_scoring_of_the_causally_filtered_signal(toy):
    cfg, model = toy
    c = model.cfg_used
    x = scene(3, 5, 3)
    sos = bandpass_sos(SR, c["preprocess"]["bandpass"]["low_hz"], c["preprocess"]["bandpass"]["high_hz"], c["preprocess"]["bandpass"]["order"])
    y = signal.sosfilt(sos, x).astype(np.float32)                      # causal, zero initial state, no DC removal
    wins, _ = window_signal(y, SR, 1.0, 0.5)
    X = np.stack([window_features(w, SR, c["features"]) for w in wins])
    p, gated = model.score_features(X, X[:, feature_names(c["features"]).index("rms_mean")])
    w, _, _ = run_offline(model, cfg, x)
    np.testing.assert_allclose(p_of(w), p[:, 0], atol=1e-5)
    assert [r.gated for r in w] == gated.tolist()


def test_int16_input_equals_float_input(toy):
    cfg, model = toy
    x = np.clip(scene(2, 3, 2), -0.99, 0.99)
    q = np.round(x * 32768).astype(np.int16)
    a, b = StreamEngine(model, cfg, t0_wall=0), StreamEngine(model, cfg, t0_wall=0)
    wa, _ = a.push(q)
    wb, _ = b.push(q.astype(np.float32) / 32768.0)
    np.testing.assert_allclose(p_of(wa), p_of(wb), atol=1e-7)


def test_a_tone_in_noise_gives_one_event_with_the_spec_fields(toy):
    cfg, model = toy
    wins, recs, eng = run_offline(model, cfg, scene(10, 12, 10), t0_wall=1_760_000_000.0)
    assert len(recs) == 1
    e = recs[0]
    assert list(e) == SPEC_KEYS
    assert e["sensor_id"] == "acoustic-01" and e["event_id"] == 1 and e["label"] == "aircraft"
    assert e["model_version"] == model.config_hash and e["top_alt_class"] == "ambient"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", e["t_start"])
    start = datetime.fromisoformat(e["t_start"].replace("Z", "+00:00")).timestamp() - 1_760_000_000.0   # seconds since t0
    assert abs(start - 10.0) <= 1.5 and abs(e["duration_s"] - 12.0) <= 2.5
    assert e["peak_prob"] > 0.9 and 0 < e["mean_prob"] <= e["peak_prob"]
    assert e["snr_est_db"] is not None and e["snr_est_db"] > 5          # tone is far above the noise floor
    assert eng.n_events == 1 and eng.active == ()


def test_event_ids_count_up_and_noise_alone_gives_no_event(toy):
    cfg, model = toy
    assert run_offline(model, cfg, noise(30))[1] == []
    x = np.concatenate([scene(8, 8, 8), scene(2.0, 8, 8)])
    recs = run_offline(model, cfg, x)[1]
    assert [r["event_id"] for r in recs] == [1, 2]
    assert recs[0]["t_end"] <= recs[1]["t_start"]


def test_flush_writes_an_event_that_is_still_open_at_the_end_of_the_stream(toy):
    cfg, model = toy
    x = np.concatenate([noise(8, seed=1), tone_stack(10, seed=2) + noise(10, seed=3)])
    eng = StreamEngine(model, cfg, t0_wall=0)
    _, live = eng.push(x)
    assert live == [] and eng.active == ("aircraft",)
    done = eng.flush()
    assert len(done) == 1 and done[0]["duration_s"] >= 8
    assert eng.active == () and eng.flush() == []


def test_digital_silence_is_gated_and_gives_no_snr_or_events(toy):
    cfg, model = toy
    wins, recs, eng = run_offline(model, cfg, np.zeros(SR * 12, np.float32))
    assert recs == [] and all(w.gated and w.label == al.SILENCE for w in wins)
    assert all(w.snr_db is None for w in wins) and eng.n_gated == len(wins)
    assert all(w.p["aircraft"] == 0.0 for w in wins)


def test_noise_floor_is_frozen_while_an_event_is_active(toy):
    cfg, model = toy
    eng = StreamEngine(model, cfg, t0_wall=0)
    eng.push(noise(15, seed=1))
    before = eng.floor.value
    assert before is not None and before > 0
    eng.push(tone_stack(15, seed=2) + noise(15, seed=3))
    assert eng.active == ("aircraft",)
    assert eng.floor.value == pytest.approx(before, rel=0.5)           # tone power is ~10^3 times the noise power


def test_event_snr_tracks_the_true_snr_on_the_synthetic_scene(toy):
    cfg, model = toy
    noise_amp = 0.03
    for tone_amp in (0.1, 0.4):
        _, recs, _ = run_offline(model, cfg, scene(15, 12, 10, amp=tone_amp, noise_amp=noise_amp))
        assert len(recs) == 1
        # in-band power of the tone stack (mean of squares) against that of the noise
        sos = bandpass_sos(SR, 100, 7000, 4)
        sig = signal.sosfilt(sos, tone_stack(12, amp=tone_amp, seed=2))[SR:]
        nse = signal.sosfilt(sos, noise(12, noise_amp, 3))[SR:]
        true_db = 10 * np.log10(np.mean(sig ** 2) / np.mean(nse ** 2))
        assert recs[0]["snr_est_db"] == pytest.approx(true_db, abs=3.0)


def test_only_the_none_variant_is_supported(toy):
    cfg, model = toy
    m = copy.copy(model)
    m.variant = "wiener"
    with pytest.raises(NotImplementedError, match="none"):
        StreamEngine(m, cfg)


def test_a_config_with_different_features_is_refused(toy):
    cfg, model = toy
    bad = copy.deepcopy(cfg)
    bad["preprocess"]["bandpass"]["low_hz"] = 150
    with pytest.raises(ValueError, match="make final"):
        StreamEngine(model, bad)


def test_reject_option_only_changes_the_window_label_not_the_events(toy):
    cfg, model = toy
    c2 = copy.deepcopy(cfg)
    c2["m5"]["alert"]["reject_below"] = 0.999999
    x = scene(6, 8, 6)
    w1, r1, _ = run_offline(model, cfg, x)
    w2, r2, _ = run_offline(model, c2, x)
    assert r1 == r2 and any(w.label == al.UNKNOWN for w in w2) and not any(w.label == al.UNKNOWN for w in w1)


def test_custom_alert_params_are_used(toy):
    cfg, model = toy
    strict = {"aircraft": al.AlertParams(t_on=0.5, t_off=0.3, min_event_s=60.0)}
    assert run_offline(model, cfg, scene(6, 8, 6), params=strict)[1] == []
