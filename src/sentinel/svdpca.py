"""PCA and uncentered truncated SVD, written from scratch with numpy.linalg.svd.

SVDPCA        centre the data, take the SVD, keep the top-k right singular vectors.
UncenteredSVD the same without centring (what sklearn.decomposition.TruncatedSVD computes).

Conventions follow scikit-learn so the two can be compared directly in tests:
  explained_variance_       s_i^2 / (n - 1)                     (SVDPCA)
  explained_variance_ratio_ explained_variance_ / total variance
  float n_components        smallest k whose cumulative ratio is strictly above the value
  signs                     each component is flipped so its largest-magnitude entry is positive
UncenteredSVD follows TruncatedSVD: explained variance is the variance of each transformed column,
and the ratio divides by the summed column variances of X. `energy_ratio_` (s_i^2 / ||X||_F^2) is
also provided because it is the quantity the Eckart-Young theorem talks about.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

NComponents = Optional[Union[int, float]]


def _sign_flip(vt: np.ndarray) -> np.ndarray:
    """Make the largest-magnitude entry of every row of vt positive (deterministic signs)."""
    idx = np.argmax(np.abs(vt), axis=1)
    signs = np.sign(vt[np.arange(vt.shape[0]), idx])
    signs[signs == 0] = 1.0
    return vt * signs[:, None]


def _as_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (samples, features), got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or inf")
    return X


def _resolve_k(n_components: NComponents, ratio: np.ndarray, max_k: int) -> int:
    """Number of components to keep. A float in (0, 1) is a cumulative variance target."""
    if n_components is None:
        return max_k
    if isinstance(n_components, (float, np.floating)) and not float(n_components).is_integer():
        if not 0.0 < n_components < 1.0:
            raise ValueError("a float n_components must lie in (0, 1)")
        k = int(np.searchsorted(np.cumsum(ratio), n_components, side="right") + 1)
        return min(k, max_k)
    k = int(n_components)
    if not 1 <= k <= max_k:
        raise ValueError(f"n_components must be in [1, {max_k}], got {k}")
    return k


class SVDPCA:
    """Principal component analysis through the SVD of the centred data matrix.

    Attributes after fit: mean_, components_ (k, d), singular_values_ (k,), explained_variance_ (k,),
    explained_variance_ratio_ (k,), plus the full spectrum in singular_values_all_ and
    explained_variance_ratio_all_ (used for scree plots and for choosing k afterwards).
    """

    def __init__(self, n_components: NComponents = None, whiten: bool = False) -> None:
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X: np.ndarray) -> "SVDPCA":
        X = _as_matrix(X)
        n, d = X.shape
        if n < 2:
            raise ValueError("need at least 2 samples")
        self.n_samples_, self.n_features_ = n, d
        self.mean_ = X.mean(axis=0)
        _, s, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        vt = _sign_flip(vt)
        ev = s ** 2 / (n - 1)
        total = ev.sum()
        ratio = ev / total if total > 0 else np.zeros_like(ev)
        k = _resolve_k(self.n_components, ratio, len(s))
        self.n_components_ = k
        self.singular_values_all_ = s
        self.explained_variance_ratio_all_ = ratio
        self.singular_values_ = s[:k]
        self.components_ = vt[:k]
        self.explained_variance_ = ev[:k]
        self.explained_variance_ratio_ = ratio[:k]
        return self

    def _check(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "components_"):
            raise RuntimeError("SVDPCA is not fitted yet")
        X = _as_matrix(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"expected {self.n_features_} features, got {X.shape[1]}")
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project onto the kept components (divided by the component std if whiten)."""
        Z = (self._check(X) - self.mean_) @ self.components_.T
        if self.whiten:
            Z = Z / np.sqrt(np.maximum(self.explained_variance_, np.finfo(float).tiny))
        return Z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Map component scores back to feature space (exact for k = rank, a projection otherwise)."""
        Z = np.asarray(Z, dtype=np.float64)
        if self.whiten:
            Z = Z * np.sqrt(self.explained_variance_)
        return Z @ self.components_ + self.mean_

    def reconstruction_mse(self, X: np.ndarray) -> float:
        """Mean squared error per entry of X after projecting to k components and back."""
        X = self._check(X)
        return float(np.mean((X - self.inverse_transform(self.transform(X))) ** 2))


class UncenteredSVD:
    """Truncated SVD of the raw (uncentred) data matrix, the same quantity as sklearn's TruncatedSVD.

    Without centring, the first component mostly points along the data mean, so it spends a
    component on the average level instead of on variation.
    """

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> "UncenteredSVD":
        X = _as_matrix(X)
        n, d = X.shape
        k = int(self.n_components)
        if not 1 <= k <= min(n, d):
            raise ValueError(f"n_components must be in [1, {min(n, d)}], got {k}")
        self.n_features_ = d
        _, s, vt = np.linalg.svd(X, full_matrices=False)
        vt = _sign_flip(vt)
        self.singular_values_all_ = s
        self.energy_ratio_all_ = s ** 2 / max((s ** 2).sum(), np.finfo(float).tiny)
        self.singular_values_ = s[:k]
        self.components_ = vt[:k]
        # sklearn TruncatedSVD definition: variance of each transformed column over the total column variance
        Z_all = X @ vt.T
        full_var = X.var(axis=0).sum()
        var_all = Z_all.var(axis=0)
        self.explained_variance_all_ = var_all
        self.explained_variance_ratio_all_ = var_all / full_var if full_var > 0 else np.zeros_like(var_all)
        self.explained_variance_ = var_all[:k]
        self.explained_variance_ratio_ = self.explained_variance_ratio_all_[:k]
        self.energy_ratio_ = self.energy_ratio_all_[:k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "components_"):
            raise RuntimeError("UncenteredSVD is not fitted yet")
        X = _as_matrix(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"expected {self.n_features_} features, got {X.shape[1]}")
        return X @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z, dtype=np.float64) @ self.components_
