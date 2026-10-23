"""Per-window feature extraction and the .npz feature cache.

Feature vector (146 values) = mean and std over the window's frames of:
  20 MFCC, 20 delta, 20 delta-delta, spectral centroid, bandwidth, rolloff, flatness,
  7 spectral contrast bands, zero-crossing rate, RMS.
Note: MFCC 0, RMS and contrast carry loudness information, which can differ between data sources.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import PROJECT_ROOT
from .denoise import denoise
from .preprocess import load_audio, preprocess_clip
from .windowing import window_signal

log = logging.getLogger(__name__)
CODE_VERSION = "m1-1"  # bump when feature or preprocessing code changes in a way the config does not capture


def feature_names(fcfg: dict[str, Any]) -> list[str]:
    """Names in the same order as window_features() output."""
    rows = ([f"mfcc{i:02d}" for i in range(fcfg["n_mfcc"])] + [f"dmfcc{i:02d}" for i in range(fcfg["n_mfcc"])]
            + [f"ddmfcc{i:02d}" for i in range(fcfg["n_mfcc"])]
            + ["centroid", "bandwidth", "rolloff", "flatness"]
            + [f"contrast{i}" for i in range(fcfg["contrast_bands"] + 1)] + ["zcr", "rms"])
    return [f"{r}_{s}" for s in ("mean", "std") for r in rows]


def window_features(win: np.ndarray, sr: int, fcfg: dict[str, Any]) -> np.ndarray:
    """Feature vector of one window (float32)."""
    n_fft, hop = fcfg["n_fft"], fcfg["hop"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = np.abs(librosa.stft(win, n_fft=n_fft, hop_length=hop))
        mfcc = librosa.feature.mfcc(y=win, sr=sr, n_mfcc=fcfg["n_mfcc"], n_fft=n_fft, hop_length=hop,
                                    n_mels=fcfg["n_mels"], fmin=fcfg["fmin"], fmax=fcfg["fmax"])
        w = fcfg["delta_width"]
        rows = np.vstack([
            mfcc, librosa.feature.delta(mfcc, width=w, order=1), librosa.feature.delta(mfcc, width=w, order=2),
            librosa.feature.spectral_centroid(S=S, sr=sr),
            librosa.feature.spectral_bandwidth(S=S, sr=sr),
            librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=fcfg["rolloff_percent"]),
            librosa.feature.spectral_flatness(S=S),
            librosa.feature.spectral_contrast(S=S, sr=sr, fmin=fcfg["fmin"], n_bands=fcfg["contrast_bands"]),
            librosa.feature.zero_crossing_rate(win, frame_length=n_fft, hop_length=hop),
            librosa.feature.rms(S=S, frame_length=n_fft),
        ])
    return np.nan_to_num(np.concatenate([rows.mean(axis=1), rows.std(axis=1)])).astype(np.float32)


def clip_features(x: np.ndarray, sr: int, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Features for every full window of an (already preprocessed and denoised) clip."""
    wins, starts = window_signal(x, sr, cfg["audio"]["window_s"], cfg["audio"]["hop_s"])
    dim = len(feature_names(cfg["features"]))
    if len(wins) == 0:
        return np.zeros((0, dim), np.float32), starts
    return np.stack([window_features(w, sr, cfg["features"]) for w in wins]), starts


def process_file(path: str | Path, variant: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Load, preprocess, denoise and featurise one file. Returns features and timings."""
    sr = cfg["audio"]["sr"]
    p = Path(path)
    x = load_audio(p if p.is_absolute() else PROJECT_ROOT / p, sr)
    y = preprocess_clip(x, sr, cfg)
    t0 = time.perf_counter()
    y, info = denoise(y, variant, sr, cfg)
    t1 = time.perf_counter()
    X, starts = clip_features(y, sr, cfg)
    t2 = time.perf_counter()
    return {"X": X, "starts": starts, "audio_s": len(x) / sr, "t_denoise": t1 - t0, "t_feat": t2 - t1,
            "svd_mean_k": info.get("mean_k")}


def feature_config(cfg: dict[str, Any], variant: str) -> dict[str, Any]:
    """The config sections that determine the features of a variant (and nothing else)."""
    d = cfg["denoise"]
    parts = {"none": [], "specsub": ["noise", "specsub"], "wiener": ["noise", "wiener"],
             "svd_energy": ["svd"], "svd_elbow": ["svd"]}[variant]
    rel = {"audio": cfg["audio"], "preprocess": cfg["preprocess"],
           "denoise": {"n_fft": d["n_fft"], "hop": d["hop"], **{p: d[p] for p in parts}},
           "features": {k: v for k, v in cfg["features"].items() if k not in ("n_jobs", "cache_dir")}}
    if variant == "svd_elbow":  # the energy target is unused by the elbow rule
        rel["denoise"]["svd"] = {"block_s": d["svd"]["block_s"]}
    return rel


def window_rms_mean(win: np.ndarray, fcfg: dict[str, Any]) -> float:
    """Mean frame RMS of one window, computed exactly as the `rms_mean` feature (the silence gate statistic)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = np.abs(librosa.stft(win, n_fft=fcfg["n_fft"], hop_length=fcfg["hop"]))
        return float(librosa.feature.rms(S=S, frame_length=fcfg["n_fft"]).mean())


def cache_key(cfg: dict[str, Any], variant: str, manifest: pd.DataFrame) -> str:
    """Hash of everything the cached features depend on."""
    rel = feature_config(cfg, variant)
    man = hashlib.sha256(manifest[["path", "label", "group_id"]].to_csv(index=False).encode()).hexdigest()
    blob = json.dumps({"cfg": rel, "variant": variant, "manifest": man, "code": CODE_VERSION}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_feature_cache(cfg: dict[str, Any], manifest: pd.DataFrame, variant: str, out: Path,
                        n_jobs: int = -1, force: bool = False) -> dict[str, Any]:
    """Compute (or reuse) the feature cache for one variant. Returns a summary dict (also stored in the npz)."""
    key = cache_key(cfg, variant, manifest)
    if out.exists() and not force:
        with np.load(out, allow_pickle=False) as z:
            if str(z["cache_key"]) == key:
                log.info("[%s] cache hit %s", variant, out.name)
                return {**json.loads(str(z["summary"])), "cache": "hit"}
    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, batch_size=8)(
        delayed(process_file)(p, variant, cfg) for p in manifest["path"])
    Xs, fidx, starts = [], [], []
    for i, r in enumerate(results):
        Xs.append(r["X"])
        fidx.append(np.full(len(r["X"]), i, np.int32))
        starts.append(r["starts"])
    X = np.concatenate(Xs).astype(np.float32)
    fidx = np.concatenate(fidx)
    n_win = np.array([len(r["X"]) for r in results])
    audio_s = float(sum(r["audio_s"] for r in results))
    summary = {
        "variant": variant, "cache_key": key, "n_files": int(len(manifest)), "n_windows": int(len(X)),
        "feature_dim": int(X.shape[1]), "files_without_a_full_window": int((n_win == 0).sum()),
        "windows_by_class": {k: int(v) for k, v in manifest.assign(n=n_win).groupby("label")["n"].sum().items()},
        "audio_seconds": round(audio_s, 2),
        "denoise_rtf": round(sum(r["t_denoise"] for r in results) / max(audio_s, 1e-9), 5),
        "feature_ms_per_window": round(1000 * sum(r["t_feat"] for r in results) / max(len(X), 1), 3),
        "mean_svd_rank_kept": (None if variant not in ("svd_energy", "svd_elbow") else
                               round(float(np.mean([r["svd_mean_k"] for r in results if r["svd_mean_k"]])), 2)),
        "wall_seconds": round(time.perf_counter() - t0, 1), "n_jobs": n_jobs,
        "note": "RTF and ms/window are summed per-file CPU time, measured on the machine that ran this",
    }
    m = manifest.iloc[fidx]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, X=X, file_index=fidx, t_start=np.concatenate(starts), label=m["label"].to_numpy().astype(str),
        group_id=m["group_id"].to_numpy().astype(str), dataset=m["dataset"].to_numpy().astype(str),
        hard_negative=m["hard_negative"].to_numpy().astype(bool), category=m["category"].to_numpy().astype(str),
        feature_names=np.array(feature_names(cfg["features"])), cache_key=np.array(key),
        summary=np.array(json.dumps(summary)))
    return {**summary, "cache": "built"}
