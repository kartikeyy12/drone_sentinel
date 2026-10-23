"""Denoising variants: shape/finite, effect on noise vs tone, causality, SVD rank rules."""
from __future__ import annotations

import numpy as np
import pytest

from sentinel.config import load_config
from sentinel.denoise import (VARIANTS, _block_edges, denoise, elbow_index, noise_psd_rolling, rank_for_energy,
                              stft, svd_denoise_stft)

SR = 16000
CFG = load_config()


def _noise_then_tone(secs_noise=3.0, secs_tone=3.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(SR * (secs_noise + secs_tone))
    x = 0.05 * rng.standard_normal(n)
    t = np.arange(n) / SR
    k = int(SR * secs_noise)
    # a tone that switches on later, so the rolling estimate has noise-only history
    x[k:] += 0.3 * np.sin(2 * np.pi * 1000 * t[k:])
    return x.astype(np.float32), k


def _band_power(x, f, width=60):
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    return spec[(freqs > f - width) & (freqs < f + width)].sum()


@pytest.mark.parametrize("variant", VARIANTS)
def test_shape_dtype_finite(variant):
    x, _ = _noise_then_tone()
    y, _ = denoise(x, variant, SR, CFG)
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all()


def test_none_is_identity_copy():
    x, _ = _noise_then_tone()
    y, _ = denoise(x, "none", SR, CFG)
    np.testing.assert_array_equal(y, x)
    assert y is not x


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        denoise(np.zeros(100, np.float32), "magic", SR, CFG)


@pytest.mark.parametrize("variant", ["specsub", "wiener"])
def test_noise_only_part_is_reduced_and_late_tone_kept(variant):
    x, k = _noise_then_tone()
    y, _ = denoise(x, variant, SR, CFG)
    seg = slice(SR // 2, k - SR // 4)  # noise-only, after warm-up
    reduction_db = 10 * np.log10(np.mean(x[seg] ** 2) / np.mean(y[seg] ** 2))
    assert reduction_db > 6
    tone_seg = slice(k + SR // 2, k + 2 * SR)
    kept = _band_power(y[tone_seg], 1000) / _band_power(x[tone_seg], 1000)
    assert kept > 0.5  # most of the tone energy at 1 kHz survives


def test_stationary_tone_is_absorbed_into_noise_estimate():
    """Documents the known weakness: a hum present for the whole window looks like noise."""
    rng = np.random.default_rng(0)
    t = np.arange(SR * 6) / SR
    x = (0.3 * np.sin(2 * np.pi * 1000 * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)
    y, _ = denoise(x, "specsub", SR, CFG)
    kept = _band_power(y[SR:], 1000) / _band_power(x[SR:], 1000)
    assert kept < 0.2


def test_noise_estimate_is_causal_after_warmup():
    rng = np.random.default_rng(0)
    P = rng.random((257, 400)) + 0.1
    warm, win = 30, 100
    base = noise_psd_rolling(P, win, 0.2, 8, warm)
    P2 = P.copy()
    P2[:, 250:] = rng.random((257, 150)) * 50  # change the future of t = 249
    changed = noise_psd_rolling(P2, win, 0.2, 8, warm)
    np.testing.assert_array_equal(base[:, warm:250], changed[:, warm:250])


def test_noise_estimate_picks_quiet_frames():
    P = np.ones((10, 100)) * 5.0
    P[:, 40:50] = 1.0  # 10 quiet frames = 10 %
    N = noise_psd_rolling(P, 100, 0.1, 1, 10)
    assert np.allclose(N[:, 99], 1.0)


def test_rank_for_energy():
    s = np.array([10.0, 5.0, 1.0, 0.5])
    e = s ** 2 / (s ** 2).sum()
    assert rank_for_energy(s, 0.5) == 1
    assert rank_for_energy(s, float(e[0] + e[1]) - 1e-6) == 2
    assert rank_for_energy(s, 1.0) == 4
    ks = [rank_for_energy(s, q) for q in (0.5, 0.9, 0.99, 0.999)]
    assert ks == sorted(ks)


def test_elbow_index_finds_the_knee():
    s = np.concatenate([[100, 90, 80, 70], np.full(46, 1.0)]) + np.linspace(0.5, 0, 50)
    assert elbow_index(s) == 4
    assert elbow_index(np.array([3.0, 2.0])) == 2


def test_block_edges_cover_everything_and_merge_tail():
    edges = _block_edges(1000, 250)
    assert edges[0][0] == 0 and edges[-1][1] == 1000 and all(b > a for a, b in edges)
    assert all(edges[i][1] == edges[i + 1][0] for i in range(len(edges) - 1))
    assert _block_edges(1010, 250)[-1] == (750, 1010)  # tiny tail merged
    assert _block_edges(100, 250) == [(0, 100)]


def test_svd_denoise_recovers_low_rank_spectrogram():
    rng = np.random.default_rng(0)
    F, T = 257, 200
    clean = np.outer(rng.random(F) + 0.5, np.abs(np.sin(np.linspace(0, 6, T))) + 0.5)
    noisy_mag = clean + 0.3 * rng.random((F, T))
    Z = noisy_mag * np.exp(1j * rng.uniform(-np.pi, np.pi, (F, T)))
    Zh, info = svd_denoise_stft(Z, "energy", 0.95, T)
    assert np.linalg.norm(np.abs(Zh) - clean) < np.linalg.norm(noisy_mag - clean)
    np.testing.assert_allclose(np.angle(Zh), np.angle(Z), atol=1e-6)  # noisy phase kept
    assert info[0]["k"] < info[0]["rank"]


def test_svd_variants_report_rank():
    x, _ = _noise_then_tone()
    for v in ("svd_energy", "svd_elbow"):
        _, info = denoise(x, v, SR, CFG)
        assert info["blocks"] and 1 <= info["mean_k"] <= 257


def test_stft_roundtrip_is_near_perfect():
    from sentinel.denoise import istft

    rng = np.random.default_rng(0)
    x = rng.standard_normal(SR).astype(np.float32)
    y = istft(stft(x, 512, 128), 512, 128, len(x))
    np.testing.assert_allclose(y, x, atol=1e-4)


def test_all_zero_block_is_handled_without_warnings():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert rank_for_energy(np.zeros(5), 0.95) == 1
        assert elbow_index(np.zeros(10)) == 1
        x = np.zeros(SR * 3, np.float32)
        for v in VARIANTS:
            y, _ = denoise(x, v, SR, CFG)
            assert y.shape == x.shape and not y.any()


def test_half_silent_clip_does_not_break_svd_blocks():
    import warnings

    rng = np.random.default_rng(0)
    x = np.zeros(SR * 6, np.float32)
    x[: SR * 3] = 0.1 * rng.standard_normal(SR * 3)  # second half is digital silence, as in padded clips
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for v in ("svd_energy", "svd_elbow"):
            y, info = denoise(x, v, SR, CFG)
            assert np.isfinite(y).all() and min(b["k"] for b in info["blocks"]) == 1
