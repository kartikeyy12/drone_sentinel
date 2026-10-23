"""Additive noise for the robustness experiment.

Design:
  * The noise level of a condition is ONE absolute in-band power, set from a reference level and a nominal SNR:
        P_noise = P_ref / 10**(snr_db / 10)
    P_ref is the median in-band power (100-7000 Hz, after DC removal) of the aircraft clips. Every clip, aircraft or
    ambient, gets noise at that same absolute level. This models a microphone sitting in a noisy place, where the
    background does not get quieter because a quiet source is being recorded. Consequence: a faint aircraft clip sees a
    worse SNR than the nominal one, a loud one a better one. `clip_snr_db` reports the SNR each clip actually got.
  * Noise is added to the raw clip, before the band-pass and the denoiser, so the whole chain sees it.
  * The noise waveform of a clip depends only on (seed, kind, clip index), not on the SNR or the denoising variant,
    so conditions are paired: the same noise shape, scaled.
  * Kinds: white, pink (power ~ 1/f) and brown (power ~ 1/f^2) are synthetic. `ambient` is a real ESC-50 recording from a
    different source group, cropped or tiled to the clip length. ESC-50 ambient classes are also in the training data,
    so the SVM has seen that kind of sound as a negative example; read the `ambient` result with that in mind.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .preprocess import load_audio, preprocess_clip

KINDS = ("white", "pink", "brown", "ambient")
_KIND_ID = {k: i for i, k in enumerate(KINDS)}
NOISE_CODE_VERSION = "m4-1"  # bump when this module changes what the noisy audio is


def colored_noise(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean Gaussian noise with a power spectrum ~ f^0 (white), f^-1 (pink) or f^-2 (brown). Unit-free."""
    if kind not in ("white", "pink", "brown"):
        raise ValueError(f"not a synthetic noise kind: {kind!r}")
    w = rng.standard_normal(n)
    if kind == "white":
        return w
    spec = np.fft.rfft(w)
    f = np.fft.rfftfreq(n)
    f[0] = f[1]  # avoid dividing by zero, the DC bin is zeroed below
    spec = spec / f ** {"pink": 0.5, "brown": 1.0}[kind]  # amplitude exponent = half the power exponent
    spec[0] = 0.0
    return np.fft.irfft(spec, n)


def inband_power(x: np.ndarray, sr: int, cfg: dict[str, Any]) -> float:
    """Mean power after the pipeline's own DC removal and band-pass. Linear in x, so P(a*x) = a^2 P(x)."""
    y = preprocess_clip(np.asarray(x, dtype=np.float32), sr, cfg).astype(np.float64)
    return float(np.mean(y ** 2))


def reference_power(paths: Sequence[str], sr: int, cfg: dict[str, Any]) -> float:
    """Median in-band power of the given clips (the aircraft clips)."""
    if not len(paths):
        raise ValueError("no reference clips")
    return float(np.median([inband_power(load_audio(p, sr), sr, cfg) for p in paths]))


def noise_target_power(p_ref: float, snr_db: float) -> float:
    return float(p_ref / 10 ** (snr_db / 10.0))


def clip_noise(kind: str, n: int, clip_index: int, group_id: str, seed: int,
               pool: Sequence[tuple[str, str]] | None, sr: int) -> np.ndarray:
    """Unscaled noise of length n for one clip. Deterministic in (seed, kind, clip_index).

    `pool` (kind == ambient only) lists (group_id, path) of candidate noise recordings. The clip's own source group is
    never used, so a recording is not mixed with itself or with another clip of the same source.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), _KIND_ID[kind], int(clip_index)]))
    if kind != "ambient":
        return colored_noise(kind, n, rng)
    if not pool:
        raise ValueError("kind 'ambient' needs a noise pool")
    cand = [i for i, (g, _) in enumerate(pool) if g != group_id]
    if not cand:
        raise ValueError("noise pool has no recording outside this clip's source group")
    x = load_audio(pool[int(rng.choice(cand))][1], sr).astype(np.float64)
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / len(x))))
    start = int(rng.integers(0, len(x) - n + 1))
    return x[start:start + n]


def add_noise(clip: np.ndarray, noise: np.ndarray, p_noise: float, sr: int, cfg: dict[str, Any],
              pn: float | None = None, pc: float | None = None) -> tuple[np.ndarray, float]:
    """clip + g * noise with g chosen so the in-band power of the added noise is exactly p_noise.

    `pn` and `pc` (in-band power of the noise and of the clip) can be passed in when already known.
    Returns (noisy float32, in-band SNR of this clip in dB: P(clip) / p_noise).
    """
    pn = inband_power(noise, sr, cfg) if pn is None else pn
    if pn <= 0:
        raise ValueError("noise has zero in-band power")
    g = np.sqrt(p_noise / pn)
    pc = inband_power(clip, sr, cfg) if pc is None else pc
    snr = float(10 * np.log10(pc / p_noise)) if pc > 0 else float("-inf")
    return (clip.astype(np.float64) + g * noise).astype(np.float32), snr
