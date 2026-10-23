"""Noise robustness: noisy feature caches and train-clean / test-noisy evaluation with grouped cross-validation.

Protocol
  * Features of every recording are recomputed with noise added to the raw audio (see noise.py), for each
    (noise kind, SNR, denoising variant). Windows line up row for row with the clean caches.
  * The model of each fold is fitted on CLEAN training windows only and scored on the NOISY windows of the held-out
    recordings, using the same StratifiedGroupKFold folds as `make train`. The `clean` row therefore reproduces the
    M3 default result for the same variant.
  * Near-silent ambient windows (the silence gate) are not scored in the main metrics, as in M3. Under noise they
    become windows that contain only the added noise, so they are reported separately as `noise_only_far`: the share
    of them the SVM calls a target class.
  * Augmentation study: train on clean windows plus windows corrupted with the OTHER noise kinds, test on the held-out
    kind. This shows whether noise-aware training transfers to a noise it has not seen.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import PROJECT_ROOT
from .cv import AMBIENT, aggregate, fold_metrics, group_bootstrap_ci
from .dataset import Dataset, GatedWindows, load_cache
from .denoise import denoise
from .features import cache_key, clip_features
from .noise import (NOISE_CODE_VERSION, add_noise, clip_noise, inband_power, noise_target_power)
from .pipeline import FeaturePipeline
from .preprocess import load_audio, preprocess_clip
from .svd_experiments import usable_noise

log = logging.getLogger(__name__)
Condition = tuple[str | None, float | None]   # (noise kind, snr_db); (None, None) is the clean condition
CLEAN: Condition = (None, None)


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ---- noise pool and caches -------------------------------------------------------------------------------------------

def build_noise_pool(cfg: dict[str, Any], manifest: pd.DataFrame) -> list[tuple[str, str]]:
    """(group_id, path) of the real ambient recordings usable as additive noise. Holdout datasets are excluded."""
    a, sr = cfg["m4"]["robustness"]["ambient_noise"], cfg["audio"]["sr"]
    sel = manifest[(manifest["label"] == AMBIENT) & manifest["category"].isin(a["categories"])
                   & ~manifest["dataset"].isin(cfg["splits"]["holdout_datasets"])].sort_values("path")
    return [(str(r.group_id), str(r.path)) for r in sel.itertuples()
            if usable_noise(load_audio(_abs(r.path), sr), a["max_zero_fraction"])]


def noisy_path(cfg: dict[str, Any], kind: str, snr: float, variant: str) -> Path:
    return PROJECT_ROOT / cfg["m4"]["robustness"]["cache_dir"] / f"{variant}__{kind}__snr{snr:+g}.npz"


def noisy_key(cfg: dict[str, Any], kind: str, snr: float, variant: str, manifest: pd.DataFrame, p_ref: float,
              pool: Sequence[tuple[str, str]] | None) -> str:
    """Hash of everything a noisy cache depends on: the clean cache key, the noise recipe and the reference level."""
    blob: dict[str, Any] = {"clean": cache_key(cfg, variant, manifest), "noise_code": NOISE_CODE_VERSION, "kind": kind,
                            "snr": float(snr), "seed": cfg["seed"], "p_ref": f"{p_ref:.10e}"}
    if kind == "ambient":
        blob["pool"] = hashlib.sha256(json.dumps(list(pool or [])).encode()).hexdigest()
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:16]


def _is_fresh(path: Path, key: str) -> bool:
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as z:
        return str(z["cache_key"]) == key


def process_clip_noisy(x: np.ndarray, clip_index: int, group: str, kind: str, conds: Sequence[tuple[float, str]],
                       p_ref: float, pool: Sequence[tuple[str, str]] | None, cfg: dict[str, Any]
                       ) -> tuple[dict[tuple[float, str], np.ndarray], np.ndarray, dict[float, float]]:
    """Window features of one raw clip for each (snr, variant). Returns (features, window start times, clip SNR in dB)."""
    sr = cfg["audio"]["sr"]
    noise = clip_noise(kind, len(x), clip_index, group, cfg["seed"], pool, sr)
    pn, pc = inband_power(noise, sr, cfg), inband_power(x, sr, cfg)
    out: dict[tuple[float, str], np.ndarray] = {}
    snr_of: dict[float, float] = {}
    starts = np.zeros(0, np.float32)
    for snr in sorted({s for s, _ in conds}):
        noisy, snr_of[snr] = add_noise(x, noise, noise_target_power(p_ref, snr), sr, cfg, pn, pc)
        y = preprocess_clip(noisy, sr, cfg)
        for v in [v for s, v in conds if s == snr]:
            out[(snr, v)], starts = clip_features(denoise(y, v, sr, cfg)[0], sr, cfg)
    return out, starts, snr_of


def _file_job(i: int, path: str, group: str, kind: str, conds: Sequence[tuple[float, str]], p_ref: float,
              pool: Sequence[tuple[str, str]] | None, cfg: dict[str, Any]):
    return process_clip_noisy(load_audio(_abs(path), cfg["audio"]["sr"]), i, group, kind, conds, p_ref, pool, cfg)


def build_noisy(cfg: dict[str, Any], manifest: pd.DataFrame, kind: str, snrs: Sequence[float], variants: Sequence[str],
                p_ref: float, pool: Sequence[tuple[str, str]] | None, n_jobs: int = -1, force: bool = False
                ) -> dict[str, Any]:
    """Build whichever (snr, variant) caches of one noise kind are missing or stale, in a single pass over the files."""
    keys = {(float(s), v): noisy_key(cfg, kind, s, v, manifest, p_ref, pool) for s in snrs for v in variants}
    todo = [sv for sv, k in keys.items() if force or not _is_fresh(noisy_path(cfg, kind, sv[0], sv[1]), k)]
    if not todo:
        return {"kind": kind, "built": [], "wall_seconds": 0.0}
    t0 = time.perf_counter()
    res = Parallel(n_jobs=n_jobs, batch_size=8)(
        delayed(_file_job)(i, p, g, kind, todo, p_ref, pool, cfg)
        for i, (p, g) in enumerate(zip(manifest["path"], manifest["group_id"])))
    for sv in todo:
        Xs = [r[0][sv] for r in res]
        fidx = np.concatenate([np.full(len(x), i, np.int32) for i, x in enumerate(Xs)])
        out = noisy_path(cfg, kind, sv[0], sv[1])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, X=np.concatenate(Xs).astype(np.float32), file_index=fidx,
                            t_start=np.concatenate([r[1] for r in res]), cache_key=np.array(keys[sv]),
                            clip_snr_db=np.array([r[2][sv[0]] for r in res]), kind=np.array(kind),
                            snr_db=np.array(sv[0]), variant=np.array(sv[1]))
    return {"kind": kind, "built": [f"{v}@{s:+g}" for s, v in todo], "wall_seconds": round(time.perf_counter() - t0, 1)}


class NoisyFeatures:
    """Feature matrices over all cache rows for a (variant, condition), checked to be row-aligned with the clean cache."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self._m: dict[tuple, np.ndarray] = {}
        self._ref = load_cache(cfg, "none")

    def get(self, variant: str, kind: str | None, snr: float | None) -> np.ndarray:
        key = (variant, kind, snr)
        if key not in self._m:
            if kind is None:
                self._m[key] = load_cache(self.cfg, variant)["X"].astype(np.float64)
            else:
                p = noisy_path(self.cfg, kind, snr, variant)
                if not p.exists():
                    raise FileNotFoundError(f"{p} not found. Run: make robustness")
                with np.load(p, allow_pickle=False) as z:
                    if not (np.array_equal(z["file_index"], self._ref["file_index"])
                            and np.array_equal(z["t_start"], self._ref["t_start"])):
                        raise ValueError(f"{p.name} is not row-aligned with the clean cache; rebuild with make robustness")
                    self._m[key] = z["X"].astype(np.float64)
        return self._m[key]


# ---- evaluation --------------------------------------------------------------------------------------------------------

def gated_fold(gated_groups: np.ndarray, ds_groups: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Fold index whose model scores each gated window: the fold in which its source group is held out.

    A group that has no kept window at all (it never enters any training or test set) gets a fold by sorted order;
    every fold's model is equally unfamiliar with it.
    """
    g2f = {g: i for i, (_, te) in enumerate(folds) for g in np.unique(ds_groups[te])}
    unknown = sorted(set(gated_groups.tolist()) - set(g2f))
    g2f.update({g: j % len(folds) for j, g in enumerate(unknown)})
    return np.array([g2f[g] for g in gated_groups], dtype=int)


def _pipeline(cfg: dict[str, Any], names: list[str]) -> FeaturePipeline:
    d = cfg["m3"]["default"]
    return FeaturePipeline(names, d["svm"], d["pca"], d["drop_level_features"], cfg["m3"]["level_feature_prefixes"],
                           cfg["m3"]["class_weight"], cfg["seed"])


def evaluate_conditions(cfg: dict[str, Any], ds: Dataset, gated: GatedWindows, folds: list[tuple[np.ndarray, np.ndarray]],
                        variant: str, feats: NoisyFeatures, conditions: Sequence[Condition],
                        train_X: Callable[[int, np.ndarray], np.ndarray], n_boot: int, seed: int, tag: str,
                        label: str | None = None) -> list[dict[str, Any]]:
    """Fit one pipeline per fold on `train_X(fold, train_rows)` and score every condition on the held-out rows.

    The pipeline settings are m3.default (variant aside). `label` overrides the variant name in the records (used when
    the same features are scored with different pipeline settings). Returns one record per condition.
    """
    classes, n = ds.classes, len(ds.y)
    pos = cfg["m4"]["positive_classes"][0]
    gfold = gated_fold(gated.groups, ds.groups, folds)
    pred = {c: np.empty(n, dtype=object) for c in conditions}
    S = {c: np.zeros((n, len(classes))) for c in conditions}
    fm: dict[Condition, list[dict[str, float]]] = {c: [] for c in conditions}
    npred = {c: np.empty(len(gated.row_id), dtype=object) for c in conditions}
    for i, (tr, te) in enumerate(folds):
        pipe = _pipeline(cfg, ds.names).fit(train_X(i, tr), ds.y[tr])
        if pipe.classes_ != classes:
            raise ValueError(f"fold {i}: class order {pipe.classes_} differs from {classes}")
        sel = gfold == i
        for c in conditions:
            Xall = feats.get(variant, *c)
            Xte = Xall[ds.row_id[te]]
            pred[c][te], S[c][te] = pipe.predict(Xte), pipe.decision(Xte)
            fm[c].append(fold_metrics(ds.y[te], pred[c][te].astype(str), S[c][te], classes, ds.hard_negative[te]))
            if sel.any():
                npred[c][sel] = pipe.predict(Xall[gated.row_id[sel]])
    recs = []
    for c in conditions:
        p = pred[c].astype(str)
        agg, pooled = aggregate(fm[c]), fold_metrics(ds.y, p, S[c], classes, ds.hard_negative)
        ci = group_bootstrap_ci(ds.y, p, ds.groups, classes, n_boot, seed)
        recs.append({
            "tag": tag, "variant": label or variant, "kind": c[0] or "clean", "snr_db": c[1],
            "ba_fold_mean": agg["balanced_accuracy_mean"], "ba_fold_std": agg["balanced_accuracy_std"],
            "ba_folds": [m["balanced_accuracy"] for m in fm[c]], "ba_pooled": pooled["balanced_accuracy"],
            "ba_lo": ci["balanced_accuracy_lo"], "ba_hi": ci["balanced_accuracy_hi"],
            f"recall_{pos}": pooled[f"recall_{pos}"], f"recall_{pos}_lo": ci[f"recall_{pos}_lo"],
            f"recall_{pos}_hi": ci[f"recall_{pos}_hi"], f"precision_{pos}": pooled[f"precision_{pos}"],
            f"auc_{pos}": pooled[f"auc_{pos}"], f"ap_{pos}": pooled[f"ap_{pos}"],
            "specificity_plain": pooled["specificity_plain"], "specificity_hard_negative": pooled["specificity_hard_negative"],
            "noise_only_far": float(np.mean(npred[c].astype(str) != AMBIENT)) if len(npred[c]) else float("nan"),
            "n_noise_only": int(len(npred[c]))})
    return recs


def clean_train(ds: Dataset) -> Callable[[int, np.ndarray], np.ndarray]:
    return lambda i, tr: ds.X[tr]


def augmented_train(ds: Dataset, feats: NoisyFeatures, variant: str, held_out_kind: str, kinds: Sequence[str],
                    snrs: Sequence[float], seed: int) -> Callable[[int, np.ndarray], np.ndarray]:
    """Training matrix in which each training RECORDING is clean or corrupted by one noise condition drawn uniformly
    from the clean condition and every (kind, snr) of the other kinds. Same number of rows as the clean training set."""
    options: list[Condition] = [CLEAN] + [(k, float(s)) for k in kinds if k != held_out_kind for s in snrs]
    k_id = list(kinds).index(held_out_kind)

    def fn(i: int, tr: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 4242, i, k_id]))
        _, inv = np.unique(ds.groups[tr], return_inverse=True)
        choice = rng.integers(0, len(options), size=inv.max() + 1)[inv]
        X = ds.X[tr].copy()
        for j, c in enumerate(options):
            rows = np.flatnonzero(choice == j)
            if c != CLEAN and len(rows):
                X[rows] = feats.get(variant, *c)[ds.row_id[tr][rows]]
        return X

    return fn


# ---- summaries -----------------------------------------------------------------------------------------------------------

def paired_aug_vs_clean(recs: pd.DataFrame, aug: pd.DataFrame) -> pd.DataFrame:
    """Augmented training minus clean-only training on the held-out noise kind, paired over the same folds."""
    rows = []
    for r in aug.itertuples():
        base = recs[(recs["variant"] == r.variant) & (recs["kind"] == ("clean" if r.kind == "clean" else r.kind))
                    & ((recs["snr_db"].isna() & pd.isna(r.snr_db)) | (recs["snr_db"] == r.snr_db))]
        if base.empty:
            continue
        d = np.array(r.ba_folds) - np.array(base.iloc[0]["ba_folds"])
        rows.append({"variant": r.variant, "held_out_kind": r.held_out_kind, "kind": r.kind, "snr_db": r.snr_db,
                     "ba_pooled_clean_trained": float(base.iloc[0]["ba_pooled"]), "ba_pooled_augmented": float(r.ba_pooled),
                     "mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)),
                     "folds_better": int((d > 0).sum()), "n_folds": int(len(d))})
    return pd.DataFrame(rows)


def paired_against(recs: pd.DataFrame, baseline: str = "none") -> pd.DataFrame:
    """Per condition: variant minus baseline variant, paired over the same folds (mean, std, folds where it is better)."""
    rows = []
    for (kind, snr), g in recs.groupby(["kind", "snr_db"], dropna=False):
        base = g[g["variant"] == baseline]
        if base.empty:
            continue
        b = np.array(base.iloc[0]["ba_folds"])
        for r in g[g["variant"] != baseline].itertuples():
            d = np.array(r.ba_folds) - b
            rows.append({"kind": kind, "snr_db": snr, "variant": r.variant, "baseline": baseline,
                         "mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)),
                         "folds_better": int((d > 0).sum()), "n_folds": int(len(d))})
    return pd.DataFrame(rows)
