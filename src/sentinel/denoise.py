"""Noise reduction implemented from scratch (NumPy/SciPy only), as comparable variants.

Variants (all work on an STFT, keep the noisy phase, and return a waveform of the input length):
  none        identity
  specsub     spectral subtraction, Boll style (power domain, over-subtraction alpha, spectral floor beta)
  wiener      Wiener-style gain with a decision-directed a priori SNR estimate
  svd_energy  truncated SVD of the magnitude spectrogram, k = smallest rank reaching an energy share
  svd_elbow   truncated SVD of the magnitude spectrogram, k = elbow of the singular value curve

specsub and wiener share a rolling noise PSD estimate: the mean power of the quietest frames in a
trailing window. After the warm-up it only uses frames up to the current one, so the same code
can run frame by frame in the live engine.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal

VARIANTS = ("none", "specsub", "wiener", "svd_energy", "svd_elbow")
_EPS = 1e-12


def stft(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Hann STFT, shape (n_fft // 2 + 1, frames)."""
    return signal.stft(x, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
                       boundary="zeros", padded=True)[2]


def istft(Z: np.ndarray, n_fft: int, hop: int, length: int) -> np.ndarray:
    """Inverse of stft(), trimmed or zero-padded to `length` samples."""
    y = signal.istft(Z, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=True)[1]
    y = y[:length]
    return np.pad(y, (0, length - len(y))).astype(np.float32)


def noise_psd_rolling(P: np.ndarray, window_frames: int, quiet_fraction: float,
                      update_every: int, warm_frames: int) -> np.ndarray:
    """Rolling noise power estimate, shape of P (freq, frames).

    For frame t >= warm_frames: mean power of the quietest `quiet_fraction` of frames in
    [t - window_frames + 1, t], refreshed every `update_every` frames (uses frames <= t only).
    For t < warm_frames: the same statistic over the first warm_frames frames (a short learning period).
    """
    F, T = P.shape
    energy = P.sum(axis=0)
    N = np.empty_like(P)

    def quiet_mean(lo: int, hi: int) -> np.ndarray:  # frames lo..hi inclusive
        seg = energy[lo:hi + 1]
        n = max(1, int(math.ceil(quiet_fraction * len(seg))))
        idx = np.argpartition(seg, n - 1)[:n]
        return P[:, lo + idx].mean(axis=1)

    warm = min(max(warm_frames, 1), T)
    N[:, :warm] = quiet_mean(0, warm - 1)[:, None]
    for t in range(warm, T, update_every):
        est = quiet_mean(max(0, t - window_frames + 1), t)
        N[:, t:t + update_every] = est[:, None]
    return N


def spectral_subtraction(Z: np.ndarray, N: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Boll-style power spectral subtraction. Gain is capped at 1 so no bin is amplified."""
    P = np.abs(Z) ** 2
    Pc = np.maximum(P - alpha * N, beta * N)
    gain = np.minimum(1.0, np.sqrt(Pc / np.maximum(P, _EPS)))
    return Z * gain


def wiener_gain(Z: np.ndarray, N: np.ndarray, dd_alpha: float, gain_floor: float) -> np.ndarray:
    """Wiener-style gain xi / (1 + xi) with a decision-directed a priori SNR (Ephraim and Malah style)."""
    P = np.abs(Z) ** 2
    gamma = P / np.maximum(N, _EPS)                       # a posteriori SNR
    G = np.empty_like(P)
    prev = np.zeros(P.shape[0])                           # |S_hat(t-1)|^2 / N(t-1)
    for t in range(P.shape[1]):
        xi = dd_alpha * prev + (1 - dd_alpha) * np.maximum(gamma[:, t] - 1.0, 0.0)
        G[:, t] = np.maximum(xi / (1.0 + xi), gain_floor)
        prev = (G[:, t] ** 2) * gamma[:, t]
    return Z * G


def rank_for_energy(s: np.ndarray, energy: float) -> int:
    """Smallest k such that the first k singular values hold at least `energy` of the total s^2."""
    c = np.cumsum(s ** 2)
    if c[-1] <= 0:  # all-zero block (digital silence): nothing to rank, keep one component
        return 1
    return int(min(len(s), np.searchsorted(c / c[-1], energy - 1e-12) + 1))


def elbow_index(s: np.ndarray) -> int:
    """Number of singular values to keep, from the elbow.

    The elbow is the point farthest from the chord joining the first and last singular values;
    the components before it (the steep part of the curve) are kept, at least one.
    """
    n = len(s)
    if s[0] <= 0:  # all-zero block
        return 1
    if n < 3 or s[0] <= s[-1]:
        return n
    x = np.arange(n) / (n - 1)
    y = (s - s[-1]) / (s[0] - s[-1])
    return max(1, int(np.argmax(np.abs(x + y - 1.0))))


def _block_edges(T: int, block: int) -> list[tuple[int, int]]:
    """Consecutive [start, end) blocks of `block` frames; a short tail merges into the previous block."""
    starts = list(range(0, T, block))
    if len(starts) > 1 and T - starts[-1] < block // 2:
        starts.pop()
    return [(a, b) for a, b in zip(starts, starts[1:] + [T])]


def svd_denoise_stft(Z: np.ndarray, mode: str, energy: float, block_frames: int
                     ) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Low-rank approximation of |Z| per block of frames, noisy phase kept. Returns (Z_hat, per-block info)."""
    out = np.empty_like(Z)
    info: list[dict[str, int]] = []
    for a, b in _block_edges(Z.shape[1], block_frames):
        mag = np.abs(Z[:, a:b])
        U, s, Vt = np.linalg.svd(mag, full_matrices=False)
        k = rank_for_energy(s, energy) if mode == "energy" else elbow_index(s)
        low = np.maximum((U[:, :k] * s[:k]) @ Vt[:k], 0.0)
        out[:, a:b] = low * np.exp(1j * np.angle(Z[:, a:b]))
        info.append({"k": int(k), "rank": int(len(s)), "frames": int(b - a)})
    return out, info


def denoise(x: np.ndarray, variant: str, sr: int, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a denoising variant to a (band-passed) waveform. Returns (waveform, info)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    if variant == "none" or len(x) == 0:
        return x.copy(), {}
    d = cfg["denoise"]
    n_fft, hop = d["n_fft"], d["hop"]
    Z = stft(x, n_fft, hop)
    if variant in ("specsub", "wiener"):
        nz = d["noise"]
        N = noise_psd_rolling(np.abs(Z) ** 2, int(round(nz["window_s"] * sr / hop)), nz["quiet_fraction"],
                              nz["update_every"], int(round(nz["warm_s"] * sr / hop)))
        if variant == "specsub":
            Zh, info = spectral_subtraction(Z, N, d["specsub"]["alpha"], d["specsub"]["beta"]), {}
        else:
            Zh, info = wiener_gain(Z, N, d["wiener"]["dd_alpha"], d["wiener"]["gain_floor"]), {}
    else:
        block = max(1, int(round(d["svd"]["block_s"] * sr / hop)))
        Zh, blocks = svd_denoise_stft(Z, variant.split("_")[1], d["svd"]["energy"], block)
        info = {"blocks": blocks, "mean_k": float(np.mean([b["k"] for b in blocks]))}
    return istft(Zh, n_fft, hop, len(x)), info
