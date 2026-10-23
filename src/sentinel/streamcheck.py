"""Helpers for scripts/check_streaming.py: causal feature cache, engine speed, and the SNR estimator on known mixtures.

The causal feature cache holds the features the LIVE engine would compute for every window of every ESC-50 clip: causal
band-pass from zero state, no DC removal, same windows as the clean cache. Scoring those with a model trained on the
zero-phase features is the train/inference mismatch of the live system, measured with the same grouped folds as M3.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal

from . import alerting as al
from .config import PROJECT_ROOT
from .dataset import load_cache
from .engine import StreamEngine, run_offline
from .features import CODE_VERSION, cache_key, clip_features, feature_config
from .noise import colored_noise, inband_power
from .preprocess import bandpass_sos, load_audio
from .robustness import NoisyFeatures

CAUSAL_CODE_VERSION = "m5-1"


def causal_sos(cfg: dict[str, Any]) -> np.ndarray:
    bp = cfg["preprocess"]["bandpass"]
    return bandpass_sos(cfg["audio"]["sr"], bp["low_hz"], bp["high_hz"], bp["order"])


def causal_filter(x: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """What the engine's StreamingBandpass produces for the whole signal in one go (zero initial state)."""
    return signal.sosfilt(causal_sos(cfg), np.asarray(x, np.float32)).astype(np.float32)


def causal_clip_features(x: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Features of every full window of a clip as the live engine computes them. Returns (X, window start times)."""
    return clip_features(causal_filter(x, cfg), cfg["audio"]["sr"], cfg)


def causal_path(cfg: dict[str, Any]) -> Path:
    return PROJECT_ROOT / cfg["m5"]["check"]["cache_dir"] / "features_causal_none.npz"


def causal_key(cfg: dict[str, Any], manifest: pd.DataFrame) -> str:
    blob = {"clean": cache_key(cfg, "none", manifest), "causal_code": CAUSAL_CODE_VERSION, "feature_code": CODE_VERSION,
            "features": feature_config(cfg, "none")}
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _job(path: str, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return causal_clip_features(load_audio(PROJECT_ROOT / path, cfg["audio"]["sr"]), cfg)


def build_causal(cfg: dict[str, Any], manifest: pd.DataFrame, n_jobs: int = -1, force: bool = False) -> dict[str, Any]:
    """Build (or reuse) the causal feature cache, row-aligned with the clean `none` cache."""
    p, key = causal_path(cfg), causal_key(cfg, manifest)
    if p.exists() and not force:
        with np.load(p, allow_pickle=False) as z:
            if str(z["cache_key"]) == key:
                return {"built": False, "wall_seconds": 0.0, "path": str(p)}
    t0 = time.perf_counter()
    res = Parallel(n_jobs=n_jobs, batch_size=8)(delayed(_job)(path, cfg) for path in manifest["path"])
    X = np.concatenate([r[0] for r in res]).astype(np.float32)
    fidx = np.concatenate([np.full(len(r[0]), i, np.int32) for i, r in enumerate(res)])
    starts = np.concatenate([r[1] for r in res])
    ref = load_cache(cfg, "none")
    if not (np.array_equal(fidx, ref["file_index"]) and np.array_equal(starts, ref["t_start"])):
        raise ValueError("causal features are not row-aligned with the clean cache; run make features")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, X=X, file_index=fidx, t_start=starts, cache_key=np.array(key))
    return {"built": True, "wall_seconds": round(time.perf_counter() - t0, 1), "path": str(p)}


class CausalFeatures(NoisyFeatures):
    """NoisyFeatures that also serves the condition ("causal", None): the live engine's features of the clean audio."""

    def get(self, variant: str, kind: str | None, snr: float | None) -> np.ndarray:
        if kind != "causal":
            return super().get(variant, kind, snr)
        key = (variant, kind, snr)
        if key not in self._m:
            if variant != "none":
                raise ValueError("causal features exist for the `none` variant only")
            p = causal_path(self.cfg)
            if not p.exists():
                raise FileNotFoundError(f"{p} not found; run make stream-check")
            with np.load(p, allow_pickle=False) as z:
                if not (np.array_equal(z["file_index"], self._ref["file_index"]) and np.array_equal(z["t_start"], self._ref["t_start"])):
                    raise ValueError("causal cache is not row-aligned with the clean cache; rebuild with make stream-check")
                self._m[key] = z["X"].astype(np.float64)
        return self._m[key]


# ---- speed ---------------------------------------------------------------------------------------------------------------

def engine_speed(model, cfg: dict[str, Any], x: np.ndarray) -> dict[str, Any]:
    """Real-time factor and per-window time of the engine on `x`, one process, whatever BLAS threads the machine uses."""
    t0 = time.perf_counter()
    wins, _, eng = run_offline(model, cfg, x)
    wall = time.perf_counter() - t0
    ms = np.array(eng.compute_ms)
    steady = ms[1:] if len(ms) > 1 else ms
    sr = eng.sr
    return {"audio_seconds": round(len(x) / sr, 2), "wall_seconds": round(wall, 3), "real_time_factor": round(wall / (len(x) / sr), 4),
            "n_windows": int(len(ms)), "first_window_ms": round(float(ms[0]), 2),
            "window_ms_median": round(float(np.median(steady)), 2), "window_ms_p95": round(float(np.percentile(steady, 95)), 2),
            "window_ms_max": round(float(steady.max()), 2), "hop_ms": round(1000.0 * eng.hop / sr, 1)}


# ---- SNR estimator ---------------------------------------------------------------------------------------------------------

def snr_scene(clip: np.ndarray, nominal_snr_db: float, cfg: dict[str, Any], seed: int, noise_power: float = 1e-4,
              lead_s: float = 30.0, tail_s: float = 15.0, kind: str = "pink"):
    """Noise, then `clip` mixed into the same noise, then noise again.

    The noise has in-band power `noise_power`; the clip is scaled so its own mean in-band power is `nominal_snr_db` above it.
    Returns (audio, clip_start_sample, clip_alone_filtered) where the last is the scaled clip through the causal filter.
    """
    sr = cfg["audio"]["sr"]
    rng = np.random.default_rng(seed)
    n = int((lead_s + tail_s) * sr) + len(clip)
    noise = colored_noise(kind, n, rng)
    noise *= np.sqrt(noise_power / inband_power(noise, sr, cfg))
    c = np.asarray(clip, np.float64)
    c = c * np.sqrt(noise_power * 10 ** (nominal_snr_db / 10) / inband_power(c, sr, cfg))
    start = int(lead_s * sr)
    mix = noise.copy()
    mix[start:start + len(c)] += c
    return mix.astype(np.float32), start, causal_filter(c, cfg).astype(np.float64)


def snr_estimate_errors(model, cfg: dict[str, Any], clip: np.ndarray, nominal_snr_db: float, seed: int,
                        noise_power: float = 1e-4) -> dict[str, np.ndarray]:
    """Run the engine on a known mixture and compare its per-window SNR estimate with the truth.

    Truth of a window inside the clip: 10 log10(power of the clip alone in that window / noise power). Windows are
    those lying entirely inside the clip. The estimate is the engine's own (power subtraction against its rolling floor).
    """
    sr = cfg["audio"]["sr"]
    x, start, c = snr_scene(clip, nominal_snr_db, cfg, seed, noise_power)
    wins, _, eng = run_offline(model, cfg, x)
    est, truth, flagged = [], [], []
    for w in wins:
        a, b = int(round(w.t_start * sr)), int(round(w.t_end * sr))
        if a >= start and b <= start + len(c):
            pw = float(np.mean(c[a - start:b - start] ** 2))
            truth.append(10 * np.log10(max(pw, 1e-30) / noise_power))
            est.append(np.nan if w.snr_db is None else w.snr_db)
            flagged.append(any(w.p[t] >= model.thresholds[t] for t in w.p))
    return {"estimate": np.array(est), "truth": np.array(truth), "flagged": np.array(flagged, bool)}


def summarize_snr_errors(rows: Sequence[dict[str, np.ndarray]], lowest_db: float = -20.0,
                         truth_range: tuple[float, float] = (-10.0, 30.0)) -> dict[str, Any]:
    est = np.concatenate([r["estimate"] for r in rows])
    truth = np.concatenate([r["truth"] for r in rows])
    ok = np.isfinite(est) & (truth >= truth_range[0]) & (truth <= truth_range[1])
    err = est[ok] - truth[ok]
    if not len(err):
        return {"n_windows": 0}
    return {"n_windows": int(ok.sum()), "n_no_estimate": int(np.sum(~np.isfinite(est))), "bias_db": float(np.median(err)),
            "mae_db": float(np.mean(np.abs(err))), "p10_err_db": float(np.percentile(err, 10)),
            "p90_err_db": float(np.percentile(err, 90)), "median_truth_db": float(np.median(truth[ok])),
            "median_estimate_db": float(np.median(est[ok])), "truth_range_db": list(truth_range)}
