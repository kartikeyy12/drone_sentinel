"""M2 experiments for the SVD denoiser.

Two kinds of evidence, kept apart on purpose:

1. Controlled: a synthetic clean signal (steady rotor-like harmonics, or impulsive bursts) is mixed with
   noise at a known SNR, so the clean magnitude spectrogram is known and every denoiser can be scored
   against it. The synthetic signals are NOT drone recordings. They exist to show how each method treats
   steady tonal content and impulsive content.
2. Descriptive: on real ESC-50 clips there is no clean reference, so only rank, retained power and the
   effect of centring are reported. Nothing there is a quality score.

Metrics (both in dB, positive is better):
  magnitude SNR gain  10 log10(sum |S|^2 / sum (|Y|-|S|)^2) of the output minus the same for the noisy input,
                      with S the clean STFT magnitude and Y the STFT magnitude of the output waveform.
  time SNR gain       10 log10(sum s^2 / sum (y-s)^2) of the output minus the same for the noisy input.
The denoisers keep the noisy phase, so the time-domain gain is limited by it; the magnitude gain is what the
feature extractor mostly sees.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

from .denoise import _block_edges, denoise, stft
from .preprocess import bandpass_offline, bandpass_sos, load_audio
from .config import PROJECT_ROOT

_EPS = 1e-12


# ---- synthetic clean signals ---------------------------------------------------------------------------------

def _unit_rms(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    r = float(np.sqrt(np.mean(x ** 2)))
    return x / r if r > 0 else x


def synth_rotor(sr: int, duration_s: float, p: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """Harmonic stack (amplitude 1/h) on a slowly drifting fundamental, with amplitude modulation. Unit RMS."""
    n = int(round(sr * duration_s))
    t = np.arange(n) / sr
    f_inst = p["f0_hz"] + p["drift_hz"] * np.sin(2 * np.pi * p["drift_rate_hz"] * t + rng.uniform(0, 2 * np.pi))
    phi = 2 * np.pi * np.cumsum(f_inst) / sr + rng.uniform(0, 2 * np.pi)
    hmax = max(1, int(p["max_hz"] // (p["f0_hz"] + abs(p["drift_hz"]))))
    x = np.zeros(n)
    for h in range(1, hmax + 1):
        x += np.sin(h * phi) / h
    x *= 1.0 + p["am_depth"] * np.sin(2 * np.pi * p["am_rate_hz"] * t + rng.uniform(0, 2 * np.pi))
    return _unit_rms(x)


def synth_bursts(sr: int, duration_s: float, p: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """Jittered train of exponentially decaying band-limited noise bursts. Unit RMS."""
    n = int(round(sr * duration_s))
    n_bursts = max(1, int(duration_s * p["rate_hz"]))
    sos = signal.butter(4, p["band_hz"], btype="bandpass", fs=sr, output="sos")
    length = int(sr * 8 * p["decay_ms"] / 1000)
    env = np.exp(-np.arange(length) / (sr * p["decay_ms"] / 1000))
    x = np.zeros(n)
    centres = (np.arange(n_bursts) + rng.uniform(0.1, 0.6, n_bursts)) / p["rate_hz"]
    for c in centres:
        a = int(c * sr)
        seg = signal.sosfilt(sos, rng.standard_normal(length)) * env
        b = min(n, a + length)
        x[a:b] += seg[: b - a]
    return _unit_rms(x)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale `noise` so that 10 log10(P_clean / P_noise) = snr_db. Returns (noisy, scaled_noise)."""
    pc, pn = float(np.mean(clean ** 2)), float(np.mean(noise ** 2))
    if pn <= 0:
        raise ValueError("noise has zero power")
    g = np.sqrt(pc / (pn * 10 ** (snr_db / 10.0)))
    return clean + g * noise, g * noise


def usable_noise(x: np.ndarray, max_zero_fraction: float) -> bool:
    """A noise clip is usable if it has power and is not padded with digital silence."""
    return bool(np.mean(x ** 2) > 1e-12 and np.mean(x == 0.0) <= max_zero_fraction)


def pick_noise_clips(manifest: pd.DataFrame, categories: list[str], per_category: int, rng: np.random.Generator,
                     sr: int, n_samples: int, max_zero_fraction: float) -> list[dict[str, Any]]:
    """Seeded choice of real ambient clips (band-limited later), cropped or tiled to n_samples. Skips padded clips."""
    out: list[dict[str, Any]] = []
    for cat in categories:
        pool = manifest[(manifest["label"] == "ambient") & (manifest["category"] == cat)]
        pool = pool.iloc[rng.permutation(len(pool))]
        got = 0
        for _, row in pool.iterrows():
            p = PROJECT_ROOT / row["path"] if not str(row["path"]).startswith("/") else row["path"]
            x = load_audio(p, sr)
            if not usable_noise(x, max_zero_fraction):
                continue
            if len(x) < n_samples:
                x = np.tile(x, int(np.ceil(n_samples / len(x))))
            out.append({"name": f"{cat}:{row['path'].split('/')[-1]}", "category": cat, "x": x[:n_samples].astype(np.float64)})
            got += 1
            if got == per_category:
                break
    return out


# ---- metrics -------------------------------------------------------------------------------------------------

def magnitude_snr_db(clean_mag: np.ndarray, test_mag: np.ndarray) -> float:
    return float(10 * np.log10(np.sum(clean_mag ** 2) / max(np.sum((test_mag - clean_mag) ** 2), _EPS)))


def time_snr_db(clean: np.ndarray, test: np.ndarray) -> float:
    return float(10 * np.log10(np.sum(clean ** 2) / max(np.sum((test - clean) ** 2), _EPS)))


# ---- fixed-rank SVD (the Eckart-Young sweep) ---------------------------------------------------------------

def fixed_rank_stft(Z: np.ndarray, k: int, block_frames: int) -> np.ndarray:
    """Rank-k approximation of |Z| per block (same blocks as svd_denoise_stft), noisy phase kept."""
    out = np.empty_like(Z)
    for a, b in _block_edges(Z.shape[1], block_frames):
        U, s, Vt = np.linalg.svd(np.abs(Z[:, a:b]), full_matrices=False)
        kk = int(min(max(1, k), len(s)))
        low = np.maximum((U[:, :kk] * s[:kk]) @ Vt[:kk], 0.0)
        out[:, a:b] = low * np.exp(1j * np.angle(Z[:, a:b]))
    return out


def block_singular_values(x: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Singular values of |STFT| for the first SVD block of x."""
    d = cfg["denoise"]
    Z = stft(x, d["n_fft"], d["hop"])
    block = max(1, int(round(d["svd"]["block_s"] * cfg["audio"]["sr"] / d["hop"])))
    a, b = _block_edges(Z.shape[1], block)[0]
    return np.linalg.svd(np.abs(Z[:, a:b]), compute_uv=False)


def _istft_len(Z: np.ndarray, cfg: dict[str, Any], length: int) -> np.ndarray:
    from .denoise import istft

    d = cfg["denoise"]
    return istft(Z, d["n_fft"], d["hop"], length)


def rank_curve(clean: np.ndarray, noisy: np.ndarray, cfg: dict[str, Any], ks: list[int]) -> dict[str, Any]:
    """Magnitude SNR gain versus fixed rank k for one clean/noisy pair, plus the first-block singular values."""
    d, sr = cfg["denoise"], cfg["audio"]["sr"]
    S = np.abs(stft(clean, d["n_fft"], d["hop"]))
    Zn = stft(noisy, d["n_fft"], d["hop"])
    block = max(1, int(round(d["svd"]["block_s"] * sr / d["hop"])))
    base = magnitude_snr_db(S, np.abs(Zn))
    gains = []
    for k in ks:
        y = _istft_len(fixed_rank_stft(Zn, k, block), cfg, len(noisy))
        gains.append(magnitude_snr_db(S, np.abs(stft(y, d["n_fft"], d["hop"]))) - base)
    return {"ks": [int(k) for k in ks], "mag_snr_gain_db": [float(g) for g in gains],
            "input_mag_snr_db": float(base),
            "sv_clean": block_singular_values(clean, cfg).tolist(), "sv_noisy": block_singular_values(noisy, cfg).tolist()}


# ---- controlled sweep -------------------------------------------------------------------------------------------

def method_list(cfg: dict[str, Any]) -> list[tuple[str, str, float | None]]:
    """(name, variant, svd energy override). `none` is the band-passed noisy input itself."""
    ms: list[tuple[str, str, float | None]] = [("none", "none", None), ("specsub", "specsub", None),
                                               ("wiener", "wiener", None), ("svd_elbow", "svd_elbow", None)]
    ms += [(f"svd_energy_{e:.2f}", "svd_energy", float(e)) for e in cfg["m2"]["sweep"]["energy"]]
    return ms


def evaluate_pair(clean: np.ndarray, noisy: np.ndarray, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Score every method on one clean/noisy pair. clean and noisy must already be band-passed float arrays."""
    d, sr = cfg["denoise"], cfg["audio"]["sr"]
    S = np.abs(stft(clean, d["n_fft"], d["hop"]))
    in_mag = magnitude_snr_db(S, np.abs(stft(noisy, d["n_fft"], d["hop"])))
    in_time = time_snr_db(clean, noisy)
    rows = []
    for name, variant, energy in method_list(cfg):
        c = cfg
        if energy is not None:
            c = copy.deepcopy(cfg)
            c["denoise"]["svd"]["energy"] = energy
        y, info = denoise(noisy.astype(np.float32), variant, sr, c)
        y = y.astype(np.float64)
        rows.append({"method": name, "k_mean": info.get("mean_k", np.nan),
                     "mag_snr_gain_db": magnitude_snr_db(S, np.abs(stft(y, d["n_fft"], d["hop"]))) - in_mag,
                     "time_snr_gain_db": time_snr_db(clean, y) - in_time,
                     "input_mag_snr_db": in_mag, "input_time_snr_db": in_time})
    return rows


def run_case(spec: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """One (signal, noise, snr) case. `spec` holds clean, noise and labels so cases can run in worker processes."""
    sos = bandpass_sos(cfg["audio"]["sr"], cfg["preprocess"]["bandpass"]["low_hz"],
                       cfg["preprocess"]["bandpass"]["high_hz"], cfg["preprocess"]["bandpass"]["order"])
    clean = bandpass_offline(spec["clean"].astype(np.float32), sos).astype(np.float64)
    noise = bandpass_offline(spec["noise"].astype(np.float32), sos).astype(np.float64)
    noisy, _ = mix_at_snr(clean, noise, spec["snr_db"])
    rows = evaluate_pair(clean, noisy, cfg)
    for r in rows:
        r.update(signal=spec["signal"], noise_kind=spec["noise_kind"], noise_id=spec["noise_id"],
                 realization=spec["realization"], snr_db=spec["snr_db"])
    return rows


def build_cases(cfg: dict[str, Any], manifest: pd.DataFrame, rng: np.random.Generator) -> list[dict[str, Any]]:
    """All (signal, noise, snr) cases of the controlled sweep, with arrays attached."""
    m = cfg["m2"]["synthetic"]
    sr = cfg["audio"]["sr"]
    n = int(round(sr * m["duration_s"]))
    rng_clean, rng_noise = rng.spawn(2)  # independent streams: changing `realizations` must not change the noise picks
    cleans = []
    for r in range(m["realizations"]):
        cleans.append(("rotor", r, synth_rotor(sr, m["duration_s"], m["rotor"], rng_clean)))
        cleans.append(("bursts", r, synth_bursts(sr, m["duration_s"], m["bursts"], rng_clean)))
    noises: list[tuple[str, str, np.ndarray]] = []
    nz = m["noise"]
    if nz["white"]:
        noises.append(("white", "white", rng_noise.standard_normal(n)))
    for c in pick_noise_clips(manifest, nz["ambient_categories"], nz["clips_per_category"], rng_noise, sr, n,
                              nz["max_zero_fraction"]):
        noises.append(("ambient", c["name"], c["x"]))
    return [{"signal": s, "realization": r, "clean": x, "noise_kind": nk, "noise_id": nid, "noise": nx, "snr_db": float(snr)}
            for (s, r, x) in cleans for (nk, nid, nx) in noises for snr in m["snr_db"]]


# ---- descriptive statistics on real clips (no ground truth) -----------------------------------------------------

def real_clip_rank_stats(x: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    """For one preprocessed real clip: rank kept and retained power per SVD setting, and the effect of centring.

    Retained power is 10 log10(P_out / P_in) of the waveform. Centring removes the per-frequency time mean of the
    block's magnitude spectrogram before the SVD: `first_share_*` is the energy share of the first singular value and
    `k95_*` the rank needed for 95 % of the energy, averaged over blocks.
    """
    d, sr = cfg["denoise"], cfg["audio"]["sr"]
    out: dict[str, Any] = {}
    p_in = float(np.mean(x ** 2)) + _EPS
    for name, variant, energy in method_list(cfg):
        if variant not in ("svd_energy", "svd_elbow"):
            continue
        c = cfg
        if energy is not None:
            c = copy.deepcopy(cfg)
            c["denoise"]["svd"]["energy"] = energy
        y, info = denoise(x, variant, sr, c)
        out[name] = {"k_mean": info["mean_k"], "power_change_db": float(10 * np.log10((np.mean(y ** 2) + _EPS) / p_in))}
    Z = stft(x, d["n_fft"], d["hop"])
    block = max(1, int(round(d["svd"]["block_s"] * sr / d["hop"])))
    fu, ku, fc, kc = [], [], [], []
    for a, b in _block_edges(Z.shape[1], block):
        mag = np.abs(Z[:, a:b])
        for centred, f_list, k_list in ((False, fu, ku), (True, fc, kc)):
            m = mag - mag.mean(axis=1, keepdims=True) if centred else mag
            s = np.linalg.svd(m, compute_uv=False)
            tot = float(np.sum(s ** 2))
            if tot <= 0:
                continue
            f_list.append(float(s[0] ** 2 / tot))
            k_list.append(int(np.searchsorted(np.cumsum(s ** 2) / tot, 0.95 - 1e-12) + 1))
    out["centering"] = {"first_share_uncentered": float(np.mean(fu)) if fu else np.nan,
                        "k95_uncentered": float(np.mean(ku)) if ku else np.nan,
                        "first_share_centered": float(np.mean(fc)) if fc else np.nan,
                        "k95_centered": float(np.mean(kc)) if kc else np.nan}
    return out
