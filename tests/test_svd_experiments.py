"""Synthetic signals, SNR mixing, metrics and the fixed-rank SVD used in the M2 experiment."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.config import load_config
from sentinel.denoise import stft, svd_denoise_stft
from sentinel.svd_experiments import (build_cases, evaluate_pair, fixed_rank_stft, magnitude_snr_db, method_list,
                                      mix_at_snr, rank_curve, real_clip_rank_stats, run_case, synth_bursts,
                                      synth_rotor, time_snr_db, usable_noise)

SR = 16000
CFG = load_config()
M2 = CFG["m2"]["synthetic"]


def test_synth_signals_are_deterministic_unit_rms_and_finite():
    for fn, p in ((synth_rotor, M2["rotor"]), (synth_bursts, M2["bursts"])):
        a = fn(SR, 2.0, p, np.random.default_rng(3))
        b = fn(SR, 2.0, p, np.random.default_rng(3))
        c = fn(SR, 2.0, p, np.random.default_rng(4))
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)
        assert len(a) == 2 * SR and np.isfinite(a).all()
        assert np.sqrt(np.mean(a ** 2)) == pytest.approx(1.0, rel=1e-9)


def test_rotor_energy_sits_on_harmonics_of_the_fundamental():
    x = synth_rotor(SR, 4.0, {**M2["rotor"], "drift_hz": 0.0, "am_depth": 0.0}, np.random.default_rng(0))
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    near = np.zeros(len(freqs), bool)
    for h in range(1, 40):
        near |= np.abs(freqs - h * M2["rotor"]["f0_hz"]) < 3.0
    assert spec[near].sum() / spec.sum() > 0.99


def test_mix_at_snr_is_exact_for_any_target():
    rng = np.random.default_rng(0)
    clean, noise = rng.standard_normal(SR), rng.standard_normal(SR) * 7
    for snr in (-5, 0, 10, 20):
        noisy, n = mix_at_snr(clean, noise, snr)
        assert 10 * np.log10(np.mean(clean ** 2) / np.mean(n ** 2)) == pytest.approx(snr, abs=1e-9)
        np.testing.assert_allclose(noisy - clean, n)
    with pytest.raises(ValueError):
        mix_at_snr(clean, np.zeros(SR), 0)


def test_metrics_perfect_and_noisy():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(1000)
    assert time_snr_db(s, s) > 100  # floor only avoids division by zero
    assert time_snr_db(s, s + s) == pytest.approx(0.0, abs=1e-9)  # error power equals signal power
    S = np.abs(rng.standard_normal((10, 10)))
    assert magnitude_snr_db(S, 2 * S) == pytest.approx(0.0, abs=1e-9)
    assert magnitude_snr_db(S, S * 1.1) == pytest.approx(20.0, abs=1e-6)


def test_usable_noise_rejects_silence_and_padding():
    x = np.random.default_rng(0).standard_normal(1000)
    assert usable_noise(x, 0.01)
    assert not usable_noise(np.zeros(1000), 0.01)
    y = x.copy()
    y[500:] = 0.0
    assert not usable_noise(y, 0.01)


def test_fixed_rank_matches_the_energy_rule_when_the_rank_agrees():
    rng = np.random.default_rng(0)
    F, T = 257, 120  # one block (block_frames >= T)
    mag = np.outer(rng.random(F) + 0.5, np.abs(np.sin(np.linspace(0, 5, T))) + 0.5) + 0.2 * rng.random((F, T))
    Z = mag * np.exp(1j * rng.uniform(-np.pi, np.pi, (F, T)))
    Zh, info = svd_denoise_stft(Z, "energy", 0.95, T)
    np.testing.assert_allclose(fixed_rank_stft(Z, info[0]["k"], T), Zh, atol=1e-10)


def test_fixed_rank_full_rank_is_identity_on_magnitudes_and_error_is_monotone():
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((64, 80)) + 1j * rng.standard_normal((64, 80))
    np.testing.assert_allclose(np.abs(fixed_rank_stft(Z, 64, 80)), np.abs(Z), atol=1e-9)
    # Eckart-Young on the magnitude matrix: the Frobenius error cannot increase with rank
    # (the max(.,0) clip can only reduce it further since magnitudes are non-negative)
    errs = [np.linalg.norm(np.abs(fixed_rank_stft(Z, k, 80)) - np.abs(Z)) for k in (1, 2, 4, 8, 16, 32, 64)]
    assert all(b <= a + 1e-9 for a, b in zip(errs, errs[1:]))
    np.testing.assert_allclose(np.angle(fixed_rank_stft(Z, 3, 80)), np.angle(Z), atol=1e-6)


def _small_cfg():
    import copy

    c = copy.deepcopy(CFG)
    c["m2"]["sweep"]["energy"] = [0.9, 0.99]
    return c


def test_evaluate_pair_none_has_zero_gain_and_methods_are_complete():
    c = _small_cfg()
    rng = np.random.default_rng(0)
    clean = synth_rotor(SR, 3.0, M2["rotor"], rng) * 0.1
    noisy = clean + 0.1 * rng.standard_normal(len(clean))
    rows = evaluate_pair(clean, noisy, c)
    assert [r["method"] for r in rows] == [m[0] for m in method_list(c)]
    none = rows[0]
    assert none["mag_snr_gain_db"] == pytest.approx(0.0, abs=0.05)  # only the STFT round trip differs
    assert all(np.isfinite(r["mag_snr_gain_db"]) and np.isfinite(r["time_snr_gain_db"]) for r in rows)


def test_svd_improves_a_steady_tone_in_white_noise():
    """The claim the experiment tests, in its simplest form: steady content is low rank, noise is not."""
    c = _small_cfg()
    rng = np.random.default_rng(0)
    clean = synth_rotor(SR, 4.0, {**M2["rotor"], "drift_hz": 0.0}, rng)
    noisy, _ = mix_at_snr(clean, rng.standard_normal(len(clean)), 0.0)
    gains = {r["method"]: r["mag_snr_gain_db"] for r in evaluate_pair(clean, noisy, c)}
    assert gains["svd_energy_0.90"] > 1.0
    assert gains["svd_elbow"] > 1.0


def test_rank_curve_shapes_and_first_rank_beats_noisy_input_for_steady_signal():
    rng = np.random.default_rng(0)
    clean = synth_rotor(SR, 4.0, {**M2["rotor"], "drift_hz": 0.0}, rng)
    noisy, _ = mix_at_snr(clean, rng.standard_normal(len(clean)), 0.0)
    rc = rank_curve(clean, noisy, CFG, [1, 2, 8, 64])
    assert rc["ks"] == [1, 2, 8, 64] and len(rc["mag_snr_gain_db"]) == 4
    assert len(rc["sv_clean"]) == len(rc["sv_noisy"]) > 100
    assert rc["mag_snr_gain_db"][0] > 0
    assert rc["sv_clean"][0] >= rc["sv_clean"][1] >= rc["sv_clean"][-1]


def test_build_cases_and_run_case_on_synthetic_noise_only_manifest():
    c = _small_cfg()
    c["m2"]["synthetic"]["duration_s"] = 3.0
    c["m2"]["synthetic"]["snr_db"] = [0]
    c["m2"]["synthetic"]["noise"]["ambient_categories"] = []
    empty = pd.DataFrame(columns=["path", "label", "category"])
    cases = build_cases(c, empty, np.random.default_rng(0))
    assert {k["signal"] for k in cases} == {"rotor", "bursts"} and all(k["noise_kind"] == "white" for k in cases)
    rows = run_case(cases[0], c)
    assert len(rows) == len(method_list(c)) and all(r["snr_db"] == 0 for r in rows)
    assert rows[0]["input_time_snr_db"] == pytest.approx(0.0, abs=1e-6)  # band-passed SNR is exactly the target


def test_real_clip_stats_centering_and_ranks():
    rng = np.random.default_rng(0)
    x = (0.05 * rng.standard_normal(SR * 4)).astype(np.float32)
    s = real_clip_rank_stats(x, _small_cfg())
    assert {"svd_elbow", "svd_energy_0.90", "svd_energy_0.99", "centering"} <= set(s)
    cen = s["centering"]
    assert cen["first_share_uncentered"] > cen["first_share_centered"]  # the mean level dominates the raw spectrogram
    assert s["svd_energy_0.90"]["k_mean"] <= s["svd_energy_0.99"]["k_mean"]
    assert stft(x, 512, 128).shape[0] == 257
