"""Synthetic signals and a toy model with the real 146-feature layout, for the streaming tests (no ESC-50 needed)."""
from __future__ import annotations

import copy

import numpy as np

from sentinel import bundle as bd
from sentinel.config import load_config
from sentinel.cv import make_folds
from sentinel.dataset import Dataset
from sentinel.features import feature_names, window_features
from sentinel.m3 import cv_run

SR = 16000


def tone_stack(seconds: float, f0: float = 150.0, amp: float = 0.2, seed: int = 0) -> np.ndarray:
    """Harmonic stack (rotor-like) with a little noise: the stand-in for an aircraft."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28)) / k for k in range(1, 9))
    return (amp * x / 2.0 + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def noise(seconds: float, amp: float = 0.03, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(int(seconds * SR))
    return (amp * np.convolve(w, np.ones(4) / 4, mode="same")).astype(np.float32)   # mildly low-passed noise


def scene(pre_s=10.0, event_s=12.0, post_s=10.0, amp=0.2, noise_amp=0.03) -> np.ndarray:
    return np.concatenate([noise(pre_s, noise_amp, 1), tone_stack(event_s, amp=amp, seed=2) + noise(event_s, noise_amp, 3),
                           noise(post_s, noise_amp, 4)])


def build_toy_full(n_per_class: int = 40, seed: int = 0):
    """Train a small model on real features of synthetic windows. Returns (cfg, model, dataset, folds)."""
    cfg = copy.deepcopy(load_config())
    cfg["m4"]["final"]["config"] = {"variant": "none", "drop_level_features": False, "pca": 6, "svm": {"kernel": "rbf", "C": 1.0}}
    fc = cfg["features"]
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for i in range(n_per_class):
        for lab in ("aircraft", "ambient"):
            if lab == "aircraft":
                a = tone_stack(2.0, f0=float(rng.uniform(80, 250)), amp=float(rng.uniform(0.05, 0.4)), seed=int(rng.integers(1 << 30)))
                a = a + noise(2.0, float(rng.uniform(0.005, 0.05)), int(rng.integers(1 << 30)))
            else:
                a = noise(2.0, float(rng.uniform(0.005, 0.1)), int(rng.integers(1 << 30)))
            for s in (0, SR // 2, SR):
                X.append(window_features(a[s:s + SR], SR, fc))
                y.append(lab)
                g.append(f"{lab}{i}")
    n = len(y)
    ds = Dataset("none", np.array(X, np.float64), np.array(y), np.array(g), np.zeros(n, bool), np.array(["x"] * n),
                 feature_names(fc), np.arange(n), {})
    folds = make_folds(ds.y, ds.groups, 5, seed)
    run = cv_run(ds.X, ds, folds, 6, {"kernel": "rbf", "C": 1.0}, "balanced", seed)
    model = bd.build_model(cfg, ds, run["scores"], run["classes"])
    # the toy classes are separable, so the FAR-based threshold lands at ~1e-12; the engine tests need a usual one
    model.thresholds = {"aircraft": 0.5}
    return cfg, model, ds, folds


def build_toy(n_per_class: int = 40, seed: int = 0):
    """(cfg, model) only."""
    return build_toy_full(n_per_class, seed)[:2]
