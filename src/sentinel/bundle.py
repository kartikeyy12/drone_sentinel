"""The saved model (`models/pipeline.joblib`): everything needed to turn audio into window probabilities.

A SentinelModel holds the fitted FeaturePipeline (z-score, SVDPCA, SVM), one Platt calibrator per target class, the
window-level alarm thresholds, the silence gate and a snapshot of the settings that define the features. Loading it
against a config whose feature settings differ raises instead of silently producing different features.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .calibration import PlattCalibrator, threshold_for_far
from .config import config_hash
from .dataset import Dataset
from .denoise import denoise
from .features import CODE_VERSION, clip_features, feature_config, window_features, window_rms_mean
from .pipeline import FeaturePipeline
from .preprocess import preprocess_clip
from .windowing import window_signal

BUNDLE_VERSION = "m4-1"
AMBIENT = "ambient"


def final_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """The pipeline settings of the final model: m3.default unless m4.final.config is a mapping."""
    c = cfg["m4"]["final"]["config"]
    return cfg["m3"]["default"] if c == "default" else c


def pipeline_hash(cfg: dict[str, Any], variant: str) -> str:
    """Hash of what decides the features of a window (and the gate): audio, filter, denoiser, feature settings, code."""
    blob = {"features": feature_config(cfg, variant), "code": CODE_VERSION, "gate": cfg["m3"]["silence_gate_rms"]}
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _versions() -> dict[str, str]:
    import librosa
    import scipy
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "librosa": librosa.__version__, "joblib": joblib.__version__}


class SentinelModel:
    def __init__(self, *, variant: str, pipeline: FeaturePipeline, calibrators: dict[str, PlattCalibrator],
                 thresholds: dict[str, float], silence_gate_rms: float, cfg_used: dict[str, Any], cfg: dict[str, Any],
                 target_far: float, extra: dict[str, Any] | None = None) -> None:
        self.variant, self.pipeline, self.calibrators = variant, pipeline, calibrators
        self.thresholds, self.silence_gate_rms, self.target_far = dict(thresholds), float(silence_gate_rms), float(target_far)
        self.targets = list(calibrators)
        self.classes = list(pipeline.classes_)
        self.cfg_used = cfg_used
        self.pipeline_hash = pipeline_hash(cfg, variant)
        self.config_hash = config_hash(cfg)
        self.versions = _versions()
        self.built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.extra = dict(extra or {})
        missing = [c for c in self.targets if c not in self.classes]
        if missing:
            raise ValueError(f"target classes {missing} are not classes of the pipeline {self.classes}")

    # ---- compatibility ----------------------------------------------------------------------------------------------
    def check_compatible(self, cfg: dict[str, Any]) -> None:
        """Raise if `cfg` would compute different features than the ones this model was fitted on."""
        now = pipeline_hash(cfg, self.variant)
        if now != self.pipeline_hash:
            raise ValueError(f"config changed since this model was built (feature settings hash {now}, model {self.pipeline_hash}); "
                             "rebuild with: make final")
        v = _versions()
        for k in ("sklearn", "numpy"):
            if v[k].split(".")[:2] != self.versions[k].split(".")[:2]:
                warnings.warn(f"model built with {k} {self.versions[k]}, running {v[k]}; results may differ", stacklevel=2)

    # ---- scoring ---------------------------------------------------------------------------------------------------
    def probabilities(self, X: np.ndarray) -> np.ndarray:
        """Calibrated probability of each target class for feature rows X, shape (n, n_targets). No gating here."""
        S = self.pipeline.decision(X)
        return np.column_stack([self.calibrators[c].predict(S[:, self.classes.index(c)]) for c in self.targets])

    def score_features(self, X: np.ndarray, rms_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Probabilities with the silence gate applied. `rms_mean` is the band-passed (before denoising) mean frame RMS of
        each window; for the `none` variant it is the `rms_mean` feature itself. Returns (p, gated); gated rows get p = 0."""
        p = self.probabilities(X)
        gated = np.asarray(rms_mean, np.float64) < self.silence_gate_rms
        p[gated] = 0.0
        return p, gated

    def alerts(self, p: np.ndarray, gated: np.ndarray | None = None) -> np.ndarray:
        """Boolean (n, n_targets): probability at or above the class threshold and not gated."""
        a = p >= np.array([self.thresholds[c] for c in self.targets])[None, :]
        return a if gated is None else a & ~np.asarray(gated)[:, None]

    def labels(self, p: np.ndarray, gated: np.ndarray | None = None) -> np.ndarray:
        """Window label: the alerting target class with the highest probability, else ambient."""
        a = self.alerts(p, gated)
        best = np.argmax(np.where(a, p, -1.0), axis=1)
        return np.where(a.any(axis=1), np.array(self.targets, dtype=object)[best], AMBIENT).astype(str)

    def score_audio(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Offline scoring of a 16 kHz mono clip, the same chain as the feature caches (zero-phase band-pass, clip-level
        denoiser, 1 s windows at 0.5 s hop). The live engine will use causal versions of the filter and the denoiser."""
        c = self.cfg_used
        sr = c["audio"]["sr"]
        full = {"audio": c["audio"], "preprocess": c["preprocess"], "denoise": c["denoise"], "features": c["features"]}
        y = preprocess_clip(np.asarray(x, np.float32), sr, full)
        wins, starts = window_signal(y, sr, c["audio"]["window_s"], c["audio"]["hop_s"])
        rms = np.array([window_rms_mean(w, c["features"]) for w in wins])
        X, _ = clip_features(denoise(y, self.variant, sr, full)[0], sr, full)
        if len(X) == 0:
            return {"t_start": starts, "gated": np.zeros(0, bool), "p": np.zeros((0, len(self.targets))), "label": np.zeros(0, str)}
        p, gated = self.score_features(X, rms)
        return {"t_start": starts, "rms_mean": rms, "gated": gated, "p": p, "label": self.labels(p, gated)}

    # ---- description -----------------------------------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        pipe = self.pipeline
        return {
            "bundle_version": BUNDLE_VERSION, "built_at": self.built_at, "variant": self.variant, "classes": self.classes,
            "targets": self.targets, "silence_gate_rms": self.silence_gate_rms, "target_window_far": self.target_far,
            "thresholds": self.thresholds,
            "calibrators": {c: {"a": k.a_, "b": k.b_, "fitted_at_prior": k.prior_} for c, k in self.calibrators.items()},
            "n_features_in": len(pipe.names), "n_features_used": int(len(pipe.columns_)), "n_components": pipe.n_components,
            "drop_level_features": pipe.drop_level_features, "svm": pipe.svm, "class_weight": pipe.class_weight,
            "n_support_vectors": [int(n) for n in pipe.model_.n_support_], "pipeline_hash": self.pipeline_hash,
            "config_hash": self.config_hash, "code_version": CODE_VERSION, "versions": self.versions, **self.extra}


def build_model(cfg: dict[str, Any], ds: Dataset, oof_scores: np.ndarray, oof_classes: list[str]) -> SentinelModel:
    """Fit the final pipeline on ALL non-holdout windows and calibrate it on out-of-fold scores.

    `oof_scores` are the class scores of the same pipeline settings from grouped cross-validation (every window scored by a
    model that never saw its recording); `oof_classes` gives their column order.
    """
    fc = final_config(cfg)
    variant = fc["variant"]
    pipe = FeaturePipeline(ds.names, fc["svm"], fc["pca"], fc["drop_level_features"], cfg["m3"]["level_feature_prefixes"],
                           cfg["m3"]["class_weight"], cfg["seed"]).fit(ds.X, ds.y)
    if list(oof_classes) != pipe.classes_:
        raise ValueError(f"out-of-fold class order {oof_classes} differs from the pipeline's {pipe.classes_}")
    if oof_scores.shape != (len(ds.y), len(pipe.classes_)):
        raise ValueError("out-of-fold scores do not match the dataset")
    target_far = float(cfg["m4"]["operating"]["target_window_far"])
    cals, thr = {}, {}
    for c in cfg["m4"]["positive_classes"]:
        s = oof_scores[:, pipe.classes_.index(c)]
        cals[c] = PlattCalibrator().fit(s, ds.y == c)
        thr[c] = threshold_for_far(cals[c].predict(s)[ds.y == AMBIENT], target_far)
    used = {k: cfg[k] for k in ("audio", "preprocess", "denoise", "features")}
    used["m3"] = {k: cfg["m3"][k] for k in ("silence_gate_rms", "class_weight", "level_feature_prefixes")}
    return SentinelModel(variant=variant, pipeline=pipe, calibrators=cals, thresholds=thr,
                         silence_gate_rms=cfg["m3"]["silence_gate_rms"], cfg_used=used, cfg=cfg, target_far=target_far,
                         extra={"final_config": fc, "n_training_windows": int(len(ds.y)),
                                "training_windows": {c: int(np.sum(ds.y == c)) for c in ds.classes}})


def save_model(model: SentinelModel, path: Path) -> int:
    """Write the joblib file and a readable JSON description next to it. Returns the joblib size in bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    path.with_suffix(".json").write_text(json.dumps(model.describe(), indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path.stat().st_size


def load_model(path: Path, cfg: dict[str, Any] | None = None) -> SentinelModel:
    m = joblib.load(path)
    if not isinstance(m, SentinelModel):
        raise ValueError(f"{path} does not contain a SentinelModel")
    if cfg is not None:
        m.check_compatible(cfg)
    return m


# ---- latency and size ---------------------------------------------------------------------------------------------------

def benchmark_stages(model: SentinelModel, windows: np.ndarray, warmup: int = 10) -> dict[str, np.ndarray]:
    """Per-window wall time in ms of each stage, one window at a time, single process. `windows` are band-passed 1 s windows.

    Stages: gate_rms (the gate statistic computed on its own), features, decision (z-score + PCA + SVM, one row),
    calibration. For the `none` variant the gate statistic is the rms_mean feature, so the engine does not pay for
    gate_rms; `total` = features + decision + calibration and `total_with_gate` adds it. The denoiser is not part of
    these timings (it is a streaming component, measured in M5).
    """
    c = model.cfg_used
    sr, fcfg = c["audio"]["sr"], c["features"]
    out: dict[str, list[float]] = {k: [] for k in ("gate_rms", "features", "decision", "calibration")}
    for i, w in enumerate(windows):
        t0 = time.perf_counter_ns()
        window_rms_mean(w, fcfg)
        t1 = time.perf_counter_ns()
        f = window_features(w, sr, fcfg)
        t2 = time.perf_counter_ns()
        S = model.pipeline.decision(f[None, :].astype(np.float64))
        t3 = time.perf_counter_ns()
        for k in model.targets:
            model.calibrators[k].predict(S[:, model.classes.index(k)])
        t4 = time.perf_counter_ns()
        if i >= warmup:
            for k, a, b in (("gate_rms", t0, t1), ("features", t1, t2), ("decision", t2, t3), ("calibration", t3, t4)):
                out[k].append((b - a) / 1e6)
    r = {k: np.array(v) for k, v in out.items()}
    r["total"] = r["features"] + r["decision"] + r["calibration"]
    r["total_with_gate"] = r["total"] + r["gate_rms"]
    return r


def stage_summary(stages: dict[str, np.ndarray], hop_s: float) -> dict[str, dict[str, float]]:
    """mean / median / p95 / max in ms per stage, plus the real-time factor of the median total against the hop."""
    out = {k: {"mean_ms": float(v.mean()), "median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95)),
               "max_ms": float(v.max()), "n": int(len(v))} for k, v in stages.items()}
    for k in ("total", "total_with_gate"):
        out[k]["share_of_hop_median"] = float(np.median(stages[k]) / (1000.0 * hop_s))
    return out


def model_footprint(model: SentinelModel, path: Path) -> dict[str, Any]:
    pipe = model.pipeline
    arrays = {"svm_support_vectors": pipe.model_.support_vectors_, "svm_dual_coef": pipe.model_.dual_coef_,
              "pca_components": None if pipe.pca_ is None else pipe.pca_.components_, "z_mean": pipe.mu_, "z_scale": pipe.sd_}
    nb = {k: int(v.nbytes) for k, v in arrays.items() if v is not None}
    t0 = time.perf_counter()
    joblib.load(path)
    return {"joblib_file_bytes": int(path.stat().st_size), "array_bytes": nb, "array_bytes_total": int(sum(nb.values())),
            "n_support_vectors": [int(n) for n in pipe.model_.n_support_], "feature_dim_into_svm": int(pipe.model_.support_vectors_.shape[1]),
            "load_seconds": round(time.perf_counter() - t0, 4), "platform": platform.platform(), "python": sys.version.split()[0]}
