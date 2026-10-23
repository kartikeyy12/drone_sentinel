"""Band-pass behaviour: pass/stop bands, zero phase, streaming equivalence, resampling, DC."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from sentinel.preprocess import (StreamingBandpass, bandpass_offline, bandpass_sos, load_audio, remove_dc,
                                 resample)

SR = 16000


@pytest.fixture(scope="module")
def sos():
    return bandpass_sos(SR, 100, 7000, 4)


def _tone(f, secs=2.0, amp=1.0):
    t = np.arange(int(SR * secs)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def _gain_db(y, x):
    mid = slice(SR // 2, -SR // 2)  # ignore edge transients
    return 20 * np.log10(np.sqrt(np.mean(y[mid] ** 2)) / np.sqrt(np.mean(x[mid] ** 2)))


def test_passband_is_flat(sos):
    for f in (300, 1000, 3000):
        x = _tone(f)
        assert abs(_gain_db(bandpass_offline(x, sos), x)) < 0.5


def test_low_stopband_attenuated(sos):
    x = _tone(30)
    assert _gain_db(bandpass_offline(x, sos), x) < -40


def test_high_edge_rolls_off(sos):
    x_mid, x_hi = _tone(1000), _tone(7800)
    drop = _gain_db(bandpass_offline(x_hi, sos), x_hi) - _gain_db(bandpass_offline(x_mid, sos), x_mid)
    assert drop < -8


def test_offline_is_zero_phase(sos):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(SR).astype(np.float32)
    y = bandpass_offline(x, sos)
    xc = signal.correlate(y, x, mode="full")
    lag = np.argmax(xc) - (len(x) - 1)
    assert lag == 0


def test_causal_never_responds_before_the_input(sos):
    x = np.zeros(SR, np.float32)
    x[SR // 2] = 1.0
    causal = StreamingBandpass(sos).process(x)
    assert np.all(causal[: SR // 2] == 0)
    zero_phase = bandpass_offline(x, sos)  # symmetric response: it rings before the impulse too
    assert np.abs(zero_phase[SR // 2 - 200: SR // 2 - 1]).max() > 1e-4


def test_streaming_chunks_equal_one_shot(sos):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(SR * 2).astype(np.float32)
    one_shot = signal.sosfilt(sos, x).astype(np.float32)
    sb = StreamingBandpass(sos)
    chunks = np.array_split(x, [100, 1777, 4000, 4001, 20000])
    streamed = np.concatenate([sb.process(c) for c in chunks])
    np.testing.assert_allclose(streamed, one_shot, atol=1e-5)


def test_streaming_reset(sos):
    x = _tone(1000, 0.5)
    sb = StreamingBandpass(sos)
    a = sb.process(x)
    sb.reset()
    np.testing.assert_array_equal(sb.process(x), a)


def test_remove_dc():
    x = _tone(500, 1.0) + 0.7
    assert abs(remove_dc(x).mean()) < 1e-6


def test_bandpass_removes_dc(sos):
    x = np.full(SR, 0.5, np.float32)
    assert np.abs(bandpass_offline(x, sos)[SR // 4: -SR // 4]).max() < 1e-3


def test_short_input_does_not_crash(sos):
    assert len(bandpass_offline(np.ones(10, np.float32), sos)) == 10
    assert len(bandpass_offline(np.ones(2, np.float32), sos)) == 2


def test_resample_keeps_frequency_and_length():
    fs = 44100
    t = np.arange(fs) / fs
    x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    y = resample(x, fs, SR)
    assert len(y) == SR
    spec = np.abs(np.fft.rfft(y))
    assert abs(np.argmax(spec) * SR / len(y) - 1000) < 2


def test_load_audio_mono_and_rate(tmp_path):
    import soundfile as sf

    stereo = np.stack([_tone(500, 1.0), _tone(500, 1.0)], axis=1)
    p = tmp_path / "s.wav"
    sf.write(str(p), stereo, SR)
    y = load_audio(p, 8000)
    assert y.ndim == 1 and len(y) == 8000
