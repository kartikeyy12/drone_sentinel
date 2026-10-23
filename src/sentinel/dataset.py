"""Load cached window features into the arrays the SVM sees, with the silence gate and the holdout rules applied.

Rules (all counted and returned in `Dataset.info`, nothing is dropped silently):
  holdout      windows from datasets listed in splits.holdout_datasets (own recordings) never enter training or CV.
  silence gate windows whose band-passed mean frame RMS (the `none` variant's rms_mean) is below m3.silence_gate_rms.
               The gate is defined on the `none` cache for every variant, so all variants see identical windows.
  label noise  a gated window labelled with a non-ambient class (silence labelled aircraft or drone) is dropped.
               A gated ambient window is not shown to the SVM and is not scored by it: the live system would not
               call the SVM on silence, and crediting it as a correct answer would inflate specificity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import PROJECT_ROOT

AMBIENT = "ambient"


@dataclass
class Dataset:
    variant: str
    X: np.ndarray                 # (n, d) float64, gated windows removed
    y: np.ndarray                 # str labels
    groups: np.ndarray            # str source-recording ids
    hard_negative: np.ndarray     # bool
    category: np.ndarray          # str source category
    names: list[str]
    row_id: np.ndarray            # row index in the cache file, for alignment checks
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def classes(self) -> list[str]:
        return sorted(set(self.y.tolist()))


def load_cache(cfg: dict[str, Any], variant: str) -> dict[str, np.ndarray]:
    p = PROJECT_ROOT / cfg["features"]["cache_dir"] / f"features_{variant}.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run: make features")
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def build_dataset(cfg: dict[str, Any], variant: str) -> Dataset:
    z = load_cache(cfg, variant)
    ref = z if variant == "none" else load_cache(cfg, "none")
    if not (np.array_equal(ref["file_index"], z["file_index"]) and np.array_equal(ref["t_start"], z["t_start"])):
        raise ValueError(f"cache for {variant!r} is not row-aligned with the `none` cache; rebuild with make features")
    names = [str(n) for n in z["feature_names"]]
    ref_names = [str(n) for n in ref["feature_names"]]
    gate_thr = float(cfg["m3"]["silence_gate_rms"])
    gated = ref["X"][:, ref_names.index("rms_mean")].astype(np.float64) < gate_thr
    label = z["label"].astype(str)
    holdout = np.isin(z["dataset"].astype(str), cfg["splits"]["holdout_datasets"])
    target = label != AMBIENT
    keep = ~holdout & ~gated
    info = {"n_cache_rows": int(len(label)), "n_holdout": int(holdout.sum()),
            "n_gated_ambient": int((~holdout & gated & ~target).sum()),
            "n_dropped_silent_target": int((~holdout & gated & target).sum()),
            "silence_gate_rms": gate_thr, "n_used": int(keep.sum())}
    idx = np.flatnonzero(keep)
    return Dataset(variant=variant, X=z["X"][idx].astype(np.float64), y=label[idx], groups=z["group_id"].astype(str)[idx],
                   hard_negative=z["hard_negative"].astype(bool)[idx], category=z["category"].astype(str)[idx],
                   names=names, row_id=idx, info=info)


@dataclass
class GatedWindows:
    """Near-silent ambient windows (the ones `build_dataset` removes). Under added noise they are the noise alone."""
    row_id: np.ndarray            # row index in every cache file
    groups: np.ndarray
    category: np.ndarray


def gated_ambient_windows(cfg: dict[str, Any]) -> GatedWindows:
    """Same gate and holdout rule as build_dataset, selecting the gated ambient windows instead of the kept ones."""
    z = load_cache(cfg, "none")
    names = [str(n) for n in z["feature_names"]]
    gated = z["X"][:, names.index("rms_mean")].astype(np.float64) < float(cfg["m3"]["silence_gate_rms"])
    holdout = np.isin(z["dataset"].astype(str), cfg["splits"]["holdout_datasets"])
    idx = np.flatnonzero(gated & ~holdout & (z["label"].astype(str) == AMBIENT))
    return GatedWindows(row_id=idx, groups=z["group_id"].astype(str)[idx], category=z["category"].astype(str)[idx])


def columns_with_prefix(names: list[str], prefixes: list[str]) -> np.ndarray:
    """Boolean mask of the feature names starting with any prefix."""
    return np.array([any(n.startswith(p) for p in prefixes) for n in names])


def feature_group_masks(names: list[str], groups: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """One boolean mask per feature group. Raises if a feature is in no group or in more than one."""
    masks = {g: columns_with_prefix(names, p) for g, p in groups.items()}
    count = np.sum(list(masks.values()), axis=0)
    if not np.all(count == 1):
        bad = [n for n, c in zip(names, count) if c != 1]
        raise ValueError(f"every feature must be in exactly one group; offenders: {bad[:5]}")
    return masks
