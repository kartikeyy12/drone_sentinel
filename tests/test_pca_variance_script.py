"""Helpers of scripts/pca_variance.py (loaded by path, the script is not a package module)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("pca_variance", Path(__file__).resolve().parents[1] / "scripts" / "pca_variance.py")
pv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pv)


def test_zscore_has_zero_mean_unit_std_and_survives_constant_columns():
    X = np.random.default_rng(0).standard_normal((200, 5)) * [1, 10, 100, 0.1, 5] + 3
    X[:, 2] = 7.0
    Z = pv.zscore(X)
    assert np.isfinite(Z).all() and np.allclose(Z.mean(axis=0), 0, atol=1e-12)
    assert np.allclose(Z.std(axis=0)[[0, 1, 3, 4]], 1.0) and np.all(Z[:, 2] == 0)


def test_components_for_matches_a_hand_computed_case():
    ratio = np.array([0.5, 0.3, 0.15, 0.05])
    assert pv.components_for(ratio, 0.5) == 1
    assert pv.components_for(ratio, 0.8) == 2
    assert pv.components_for(ratio, 0.95) == 3
    assert pv.components_for(ratio, 1.0) == 4
