"""Ablation grid, nested selection and feature-group study (worker functions, resume logic, factor effects).

Workers are module-level so joblib can run them in separate processes. Every worker fits scalers, PCA and SVMs on the
training part of its fold only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cv import FoldFeatures, make_folds, run_fold_config

KEY_COLS = ["variant", "drop_level", "pca", "kernel", "C"]
METRIC_COLS = ["balanced_accuracy", "recall_aircraft", "specificity_plain", "specificity_hard_negative",
               "precision_aircraft", "f1_aircraft", "ap_aircraft", "auc_aircraft"]


# ---- grid definition ----------------------------------------------------------------------------------------------

def expand_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    g = cfg["m3"]["grid"]
    out = []
    for v in g["variants"]:
        for drop in g["drop_level_features"]:
            for k in g["pca"]:
                for svm in g["svm"]:
                    out.append({"variant": v, "drop_level": bool(drop), "pca": None if k is None else int(k),
                                "kernel": svm["kernel"], "C": float(svm["C"])})
    return out


def block_key(c: dict[str, Any]) -> str:
    return f"{c['variant']}|{int(c['drop_level'])}"


def cfg_id(c: dict[str, Any]) -> str:
    return f"{c['variant']}|drop={int(c['drop_level'])}|pca={'none' if c['pca'] is None else c['pca']}|{c['kernel']}|C={c['C']:g}"


def svm_of(c: dict[str, Any]) -> dict[str, Any]:
    return {"kernel": c["kernel"], "C": c["C"]}


def fingerprint(cfg: dict[str, Any], cache_keys: dict[str, str], folds: list[tuple[np.ndarray, np.ndarray]]) -> str:
    """Hash of everything a stored grid row depends on, except the grid ranges (rows are matched per configuration)."""
    m3 = {k: v for k, v in cfg["m3"].items() if k in ("silence_gate_rms", "class_weight", "level_feature_prefixes")}
    fold_digest = hashlib.sha256(b"".join(te.tobytes() for _, te in folds)).hexdigest()
    blob = json.dumps({"m3": m3, "seed": cfg["seed"], "cv_folds": cfg["splits"]["cv_folds"], "caches": cache_keys,
                       "folds": fold_digest, "holdout": cfg["splits"]["holdout_datasets"]}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---- workers ------------------------------------------------------------------------------------------------------

def fold_worker(X: np.ndarray, y: np.ndarray, hn: np.ndarray, tr: np.ndarray, te: np.ndarray,
                configs: list[dict[str, Any]], class_weight: str | None, seed: int) -> list[dict[str, Any]]:
    """All given configurations of one block on one fold. Returns one flat row per configuration."""
    ff = FoldFeatures(X[tr], X[te])
    rows = []
    for c in configs:
        m, _, _ = run_fold_config(ff, y[tr], y[te], hn[te], c["pca"], svm_of(c), class_weight, seed)
        rows.append({**c, **m})
    return rows


def nested_outer_worker(Xs: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray, hn: np.ndarray, tr: np.ndarray,
                        te: np.ndarray, configs: list[dict[str, Any]], inner_k: int, class_weight: str | None, seed: int,
                        outer_i: int) -> dict[str, Any]:
    """One outer fold: choose the configuration by inner grouped CV on the outer training part, then score it on the
    outer test part. The outer test part is touched only by the final scoring."""
    yt, gt, hnt = y[tr], groups[tr], hn[tr]
    inner = make_folds(yt, gt, inner_k, seed + 1000 + outer_i)
    inner_ba = np.zeros((len(configs), inner_k))
    by_block: dict[str, list[int]] = {}
    for i, c in enumerate(configs):
        by_block.setdefault(block_key(c), []).append(i)
    for bk, idxs in by_block.items():
        Xb = Xs[bk][tr]
        for j, (itr, ite) in enumerate(inner):
            ff = FoldFeatures(Xb[itr], Xb[ite])
            for ci in idxs:
                c = configs[ci]
                m, _, _ = run_fold_config(ff, yt[itr], yt[ite], hnt[ite], c["pca"], svm_of(c), class_weight, seed)
                inner_ba[ci, j] = m["balanced_accuracy"]
    mean = inner_ba.mean(axis=1)
    best = int(np.argmax(mean))  # first maximum in grid order: deterministic
    c = configs[best]
    Xb = Xs[block_key(c)]
    ff = FoldFeatures(Xb[tr], Xb[te])
    metrics, pred, S = run_fold_config(ff, y[tr], y[te], hn[te], c["pca"], svm_of(c), class_weight, seed)
    return {"outer_fold": outer_i, "chosen": c, "inner_ba": mean, "inner_ba_chosen": float(mean[best]), "metrics": metrics,
            "pred": pred, "scores": S, "te": te}


def subset_worker(X: np.ndarray, y: np.ndarray, hn: np.ndarray, tr: np.ndarray, te: np.ndarray,
                  subsets: list[tuple[str, np.ndarray]], svm: dict[str, Any], class_weight: str | None, seed: int
                  ) -> list[dict[str, Any]]:
    """Feature-group study on one fold: one pipeline (no PCA) per column subset."""
    rows = []
    for label, mask in subsets:
        ff = FoldFeatures(X[tr][:, mask], X[te][:, mask])
        m, _, _ = run_fold_config(ff, y[tr], y[te], hn[te], None, svm, class_weight, seed)
        rows.append({"subset": label, "n_features": int(mask.sum()), **m})
    return rows


# ---- stored rows and summaries --------------------------------------------------------------------------------------

def load_rows(path: Path, fp: str) -> pd.DataFrame:
    """Rows of the JSONL result file that match this fingerprint (others are ignored)."""
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("fingerprint") == fp]
    return pd.DataFrame(rows)


def append_rows(path: Path, rows: list[dict[str, Any]], fp: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**r, "fingerprint": fp}, default=lambda o: None if o != o else float(o)) + "\n")


def missing_by_block_fold(done: pd.DataFrame, configs: list[dict[str, Any]], n_folds: int) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Configurations still to run for each (block, fold)."""
    have: set[tuple] = set()
    if len(done):
        for r in done.itertuples():
            pca = None if (r.pca is None or r.pca != r.pca) else int(r.pca)
            have.add((r.variant, bool(r.drop_level), pca, r.kernel, float(r.C), int(r.fold)))
    todo: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for c in configs:
        for f in range(n_folds):
            if (c["variant"], c["drop_level"], c["pca"], c["kernel"], c["C"], f) not in have:
                todo.setdefault((block_key(c), f), []).append(c)
    return todo


def summarize_grid(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration: mean and std of each metric over folds."""
    d = rows.copy()
    d["pca"] = d["pca"].astype("Int64")
    g = d.groupby(KEY_COLS, dropna=False)
    out = g.agg(n_folds=("fold", "nunique"),
                **{f"{m}_mean": (m, "mean") for m in METRIC_COLS}, **{f"{m}_std": (m, "std") for m in METRIC_COLS}).reset_index()
    return out.sort_values("balanced_accuracy_mean", ascending=False).reset_index(drop=True)


def factor_effects(summary: pd.DataFrame, metric: str = "balanced_accuracy_mean") -> pd.DataFrame:
    """Paired effect of each factor level against a baseline level: the same configuration with only that factor changed.

    Baselines: variant `none`, level features kept, no PCA, rbf C=1. Reported: mean paired difference, the spread of the
    paired differences, and in how many matched configurations the level beats the baseline.
    """
    s = summary.copy()
    s["pca_key"] = s["pca"].map(lambda v: "none" if pd.isna(v) else str(int(v)))  # explicit: str(NA) differs across pandas versions
    s["svm_key"] = s["kernel"] + "|C=" + s["C"].map(lambda v: f"{v:g}")
    factors = {"variant": ("variant", "none"), "drop_level": ("drop_level", False), "pca": ("pca_key", "none"),
               "svm": ("svm_key", "rbf|C=1")}
    rows = []
    for fname, (col, base) in factors.items():
        others = [c for c in ("variant", "drop_level", "pca_key", "svm_key") if c != col]
        b = s[s[col] == base][others + [metric]].rename(columns={metric: "base"})
        for lev in s[col].unique():
            if lev == base:
                continue
            m = s[s[col] == lev][others + [metric]].merge(b, on=others)
            if m.empty:
                continue
            diff = m[metric] - m["base"]
            rows.append({"factor": fname, "level": str(lev), "baseline": str(base), "n_pairs": len(m),
                         "mean_diff": float(diff.mean()), "std_diff": float(diff.std(ddof=1)) if len(m) > 1 else np.nan,
                         "better_in": int((diff > 0).sum()), "worse_in": int((diff < 0).sum())})
    return pd.DataFrame(rows)
