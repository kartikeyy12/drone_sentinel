"""Dataset loading: silence gate, holdout, row alignment and feature groups."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from sentinel.config import load_config
from sentinel.dataset import build_dataset, feature_group_masks
from sentinel.features import feature_names

CFG = load_config()
NAMES = feature_names(CFG["features"])
RMS = NAMES.index("rms_mean")


def _write_cache(path, X, label, dataset, group, rms_override=None):
    n = len(label)
    X = X.copy()
    if rms_override is not None:
        X[:, RMS] = rms_override
    np.savez_compressed(path, X=X.astype(np.float32), file_index=np.arange(n, dtype=np.int32), t_start=np.zeros(n),
                        label=np.array(label), group_id=np.array(group), dataset=np.array(dataset),
                        hard_negative=np.zeros(n, bool), category=np.array(["c"] * n), feature_names=np.array(NAMES))


def _setup(tmp_path):
    rng = np.random.default_rng(0)
    n = 10
    X = rng.standard_normal((n, len(NAMES)))
    label = ["aircraft", "aircraft", "ambient", "ambient", "ambient", "ambient", "ambient", "ambient", "aircraft", "ambient"]
    dataset = ["esc50"] * 8 + ["own", "own"]
    group = [f"g{i}" for i in range(n)]
    rms = np.array([1e-2, 1e-6, 1e-2, 1e-6, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])  # rows 1 (aircraft) and 3 (ambient) silent
    _write_cache(tmp_path / "features_none.npz", X, label, dataset, group, rms)
    cfg = copy.deepcopy(CFG)
    cfg["features"]["cache_dir"] = str(tmp_path)
    return cfg, X, rms


def test_gate_drops_silent_target_counts_silent_ambient_and_excludes_holdout(tmp_path):
    cfg, X, _ = _setup(tmp_path)
    ds = build_dataset(cfg, "none")
    assert ds.info["n_dropped_silent_target"] == 1 and ds.info["n_gated_ambient"] == 1 and ds.info["n_holdout"] == 2
    assert sorted(ds.row_id.tolist()) == [0, 2, 4, 5, 6, 7]
    assert "own" not in set(ds.groups.tolist()) and len(ds.X) == ds.info["n_used"] == 6


def test_gate_comes_from_the_none_cache_for_every_variant(tmp_path):
    cfg, X, _ = _setup(tmp_path)
    # the denoised variant has completely different rms values (denoising lowers them) but must use the same windows
    lab = ["aircraft", "aircraft", "ambient", "ambient", "ambient", "ambient", "ambient", "ambient", "aircraft", "ambient"]
    _write_cache(tmp_path / "features_specsub.npz", X, lab, ["esc50"] * 8 + ["own", "own"], [f"g{i}" for i in range(10)],
                 rms_override=np.full(10, 1e-9))
    a, b = build_dataset(cfg, "none"), build_dataset(cfg, "specsub")
    np.testing.assert_array_equal(a.row_id, b.row_id)


def test_misaligned_cache_is_rejected(tmp_path):
    cfg, X, rms = _setup(tmp_path)
    n = len(X)
    np.savez_compressed(tmp_path / "features_wiener.npz", X=X.astype(np.float32), file_index=np.arange(n)[::-1].astype(np.int32),
                        t_start=np.zeros(n), label=np.array(["ambient"] * n), group_id=np.array(["g"] * n),
                        dataset=np.array(["esc50"] * n), hard_negative=np.zeros(n, bool), category=np.array(["c"] * n),
                        feature_names=np.array(NAMES))
    with pytest.raises(ValueError, match="aligned"):
        build_dataset(cfg, "wiener")


def test_missing_cache_gives_a_helpful_error(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["features"]["cache_dir"] = str(tmp_path)
    with pytest.raises(FileNotFoundError, match="make features"):
        build_dataset(cfg, "none")


def test_feature_groups_partition_all_features_and_level_group_has_18():
    masks = feature_group_masks(NAMES, CFG["m3"]["feature_groups"])
    assert sum(int(m.sum()) for m in masks.values()) == len(NAMES) == 146
    assert masks["mfcc"].sum() == 40 and masks["delta_mfcc"].sum() == 40 and masks["delta2_mfcc"].sum() == 40
    from sentinel.dataset import columns_with_prefix

    assert columns_with_prefix(NAMES, CFG["m3"]["level_feature_prefixes"]).sum() == 18


def test_feature_groups_reject_overlap_and_gaps():
    with pytest.raises(ValueError):
        feature_group_masks(NAMES, {"a": ["mfcc"], "b": ["mfcc00_"]})  # overlap and gaps
    with pytest.raises(ValueError):
        feature_group_masks(NAMES, {"a": ["mfcc"]})
