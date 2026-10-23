"""Noise generation and SNR scaling: spectral slopes, exact in-band power, determinism and pairing across SNRs."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from sentinel.config import load_config
from sentinel.noise import (add_noise, clip_noise, colored_noise, inband_power, noise_target_power, reference_power)

CFG = load_config()
SR = CFG["audio"]["sr"]


def _slope(x: np.ndarray, lo: float = 200.0, hi: float = 6000.0) -> float:
    f, p = signal.welch(x, fs=SR, nperseg=4096)
    m = (f >= lo) & (f <= hi)
    return float(np.polyfit(np.log10(f[m]), np.log10(p[m]), 1)[0])


@pytest.mark.parametrize("kind,expected", [("white", 0.0), ("pink", -1.0), ("brown", -2.0)])
def test_colored_noise_has_the_requested_spectral_slope(kind, expected):
    x = colored_noise(kind, 2 ** 18, np.random.default_rng(0))
    assert _slope(x) == pytest.approx(expected, abs=0.12)
    assert abs(x.mean()) < 0.05 * x.std()


def test_colored_noise_rejects_non_synthetic_kind():
    with pytest.raises(ValueError, match="synthetic"):
        colored_noise("ambient", 100, np.random.default_rng(0))


def _clip(n=SR * 3, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    # 50 Hz hum is below the band-pass and must not count towards the in-band level
    return (0.1 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.sin(2 * np.pi * 1500 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
            + 0.01 * rng.standard_normal(n)).astype(np.float32)


def test_inband_power_ignores_content_below_the_band():
    x = _clip()
    expected = 0.5 * 0.1 ** 2 + 0.5 * 0.05 ** 2 + 0.01 ** 2
    assert inband_power(x, SR, CFG) == pytest.approx(expected, rel=0.03)


@pytest.mark.parametrize("kind", ["white", "pink", "brown"])
def test_added_noise_has_exactly_the_requested_inband_power(kind):
    x = _clip()
    noise = clip_noise(kind, len(x), 3, "g", 7, None, SR)
    p_noise = 2.5e-4
    noisy, snr = add_noise(x, noise, p_noise, SR, CFG)
    added = noisy.astype(np.float64) - x.astype(np.float64)
    assert inband_power(added, SR, CFG) == pytest.approx(p_noise, rel=2e-3)
    assert snr == pytest.approx(10 * np.log10(inband_power(x, SR, CFG) / p_noise), abs=1e-9)
    assert noisy.dtype == np.float32 and len(noisy) == len(x)


def test_noise_target_power_follows_the_db_definition():
    assert noise_target_power(1.0, 0.0) == pytest.approx(1.0)
    assert noise_target_power(1.0, 10.0) == pytest.approx(0.1)
    assert noise_target_power(1.0, -10.0) == pytest.approx(10.0)


def test_noise_is_deterministic_and_the_same_shape_at_every_snr():
    x = _clip()
    a = clip_noise("pink", len(x), 5, "g", 11, None, SR)
    assert np.array_equal(a, clip_noise("pink", len(x), 5, "g", 11, None, SR))
    assert not np.array_equal(a, clip_noise("pink", len(x), 6, "g", 11, None, SR))   # other clip
    assert not np.array_equal(a, clip_noise("white", len(x), 5, "g", 11, None, SR))  # other kind
    assert not np.array_equal(a, clip_noise("pink", len(x), 5, "g", 12, None, SR))   # other seed
    n1 = add_noise(x, a, 1e-3, SR, CFG)[0].astype(np.float64) - x
    n2 = add_noise(x, a, 1e-1, SR, CFG)[0].astype(np.float64) - x
    assert np.corrcoef(n1, n2)[0, 1] > 0.999  # paired: same noise, only scaled
    assert np.std(n2) / np.std(n1) == pytest.approx(10.0, rel=0.02)


def test_ambient_noise_never_uses_the_clips_own_source_group(tmp_path):
    paths = []
    for i, f in enumerate((300.0, 900.0)):
        p = tmp_path / f"n{i}.wav"
        sf.write(p, 0.3 * np.sin(2 * np.pi * f * np.arange(SR * 2) / SR), SR)
        paths.append(str(p))
    pool = [("gA", paths[0]), ("gB", paths[1])]
    for idx in range(6):
        own_a = clip_noise("ambient", SR, idx, "gA", 1, pool, SR)
        own_b = clip_noise("ambient", SR, idx, "gB", 1, pool, SR)
        assert np.argmax(np.abs(np.fft.rfft(own_a))) * SR / len(own_a) == pytest.approx(900.0, abs=2)  # from gB
        assert np.argmax(np.abs(np.fft.rfft(own_b))) * SR / len(own_b) == pytest.approx(300.0, abs=2)  # from gA
    assert len(clip_noise("ambient", SR * 5, 0, "gA", 1, pool, SR)) == SR * 5  # tiled to length
    with pytest.raises(ValueError, match="outside this clip"):
        clip_noise("ambient", SR, 0, "gA", 1, [("gA", paths[0])], SR)
    with pytest.raises(ValueError, match="needs a noise pool"):
        clip_noise("ambient", SR, 0, "gA", 1, None, SR)


def test_reference_power_is_the_median_of_clip_powers(tmp_path):
    ps = []
    for i, a in enumerate((0.1, 0.2, 0.4)):
        p = tmp_path / f"c{i}.wav"
        sf.write(p, a * np.sin(2 * np.pi * 1000 * np.arange(SR * 2) / SR), SR, subtype="FLOAT")
        ps.append(str(p))
    assert reference_power(ps, SR, CFG) == pytest.approx(0.5 * 0.2 ** 2, rel=0.03)
    with pytest.raises(ValueError):
        reference_power([], SR, CFG)
