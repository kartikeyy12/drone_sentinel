"""Bench distance: file discovery, spreading prediction, SNR measurement, and an end-to-end run on synthetic recordings."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from helpers_stream import SR, noise, tone_stack
from sentinel import bench


def test_find_bench_files_sorted_numerically_and_ignores_others(tmp_path):
    for n in ("bench_8m.wav", "bench_1m.wav", "bench_2.5m.wav", "bench_noise.wav", "other.wav", "bench_4m.txt"):
        (tmp_path / n).write_bytes(b"")
    assert [d for d, _ in bench.find_bench_files(tmp_path)] == [1.0, 2.5, 8.0]


def test_prediction_loses_six_db_per_doubling_and_is_anchored():
    assert bench.predicted_snr_db(1, 1, 30.0) == pytest.approx(30.0)
    assert bench.predicted_snr_db(2, 1, 30.0) == pytest.approx(30.0 - 20 * np.log10(2))
    assert bench.predicted_snr_db(8, 2, 20.0, alpha_db_per_m=0.1) == pytest.approx(20.0 - 20 * np.log10(4) - 0.6)


def test_measured_snr_with_a_noise_reference_matches_the_power_ratio(toy):
    cfg, _ = toy
    n = noise(5.0, 0.02, 7)
    sig = tone_stack(5.0, amp=0.2, seed=3)
    x = (sig + n).astype(np.float32)
    got = bench.measure_snr_db(bench.inband(x, cfg), cfg, bench.inband(n, cfg))
    pn = float(np.mean(bench.inband(n, cfg) ** 2))
    pt = float(np.mean(bench.inband(x, cfg) ** 2))
    assert got["snr_db"] == pytest.approx(10 * np.log10((pt - pn) / pn), abs=0.2)
    assert got["noise_reference"] == "silent-speaker recording"


def _rec(sig, bg, lead=8.0, tail=4.0):
    """Recording with a noise-only lead-in and tail around the playback (what the bench instructions ask for)."""
    n_lead, n_tail = int(lead * SR), int(tail * SR)
    x = bg.copy()
    x[n_lead:len(x) - n_tail] += sig[:len(x) - n_tail - n_lead]
    return x.astype(np.float32)


def test_end_to_end_louder_recording_has_higher_snr_and_detection(toy, tmp_path):
    cfg, model = toy
    sig = tone_stack(40.0, amp=0.3, seed=5)
    bg = noise(32.0, 0.02, 9)
    for d, g in ((1.0, 1.0), (4.0, 0.25)):                      # 12 dB quieter signal at 4 m
        sf.write(tmp_path / f"bench_{d:g}m.wav", _rec(g * sig, bg), SR)
    rows = [bench.evaluate_file(model, cfg, p) for _, p in bench.find_bench_files(tmp_path)]
    assert rows[0]["snr_db"] > rows[1]["snr_db"] + 6
    assert rows[0]["n_active_windows"] >= 30 and rows[0]["window_detection_rate"] >= 0.9
    assert rows[0]["window_detection_rate"] >= (rows[1]["window_detection_rate"] or 0.0)
    assert rows[0]["file"] == "bench_1m.wav" and rows[0]["noise_reference"].endswith("own frame power")
    assert rows[0]["noise_floor_is_own_percentile"] is True


def test_a_recording_with_no_noise_only_frames_is_flagged_not_trusted(toy, tmp_path):
    cfg, model = toy
    sf.write(tmp_path / "bench_1m.wav", (tone_stack(20.0, amp=0.3, seed=5) + noise(20.0, 0.02, 9)).astype(np.float32), SR)
    row = bench.evaluate_file(model, cfg, tmp_path / "bench_1m.wav")
    assert row["noise_floor_is_own_percentile"] is True and row["snr_db"] is not None and row["snr_db"] < 3.0
