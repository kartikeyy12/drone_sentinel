"""Feature selection, z-score, SVDPCA and SVM as one fit/predict object (the thing M4 serialises).

The ablation grid uses the faster FoldFeatures slicing in cv.py; tests check that both give the same decision values.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.svm import SVC

from .cv import class_scores, fit_svm
from .dataset import columns_with_prefix
from .svdpca import SVDPCA


class FeaturePipeline:
    """names -> optional level-feature drop -> z-score (train stats) -> optional SVDPCA(k) -> SVC."""

    def __init__(self, names: list[str], svm: dict[str, Any], n_components: int | None = None,
                 drop_level_features: bool = False, level_prefixes: list[str] | None = None,
                 class_weight: str | None = "balanced", seed: int = 0) -> None:
        self.names = list(names)
        self.svm, self.n_components = dict(svm), n_components
        self.drop_level_features = drop_level_features
        self.level_prefixes = list(level_prefixes or [])
        self.class_weight, self.seed = class_weight, seed
        keep = ~columns_with_prefix(self.names, self.level_prefixes) if drop_level_features else np.ones(len(names), bool)
        self.columns_ = np.flatnonzero(keep)

    def _z(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != len(self.names):
            raise ValueError(f"expected {len(self.names)} features, got {X.shape[1]}")
        return (X[:, self.columns_] - self.mu_) / self.sd_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeaturePipeline":
        X = np.asarray(X, dtype=np.float64)[:, self.columns_]
        self.mu_ = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd_ = np.where(sd > 0, sd, 1.0)
        Z = (X - self.mu_) / self.sd_
        self.pca_ = SVDPCA(self.n_components).fit(Z) if self.n_components is not None else None
        self.model_: SVC = fit_svm(Z if self.pca_ is None else self.pca_.transform(Z), y, self.svm, self.class_weight, self.seed)
        self.classes_ = [str(c) for c in self.model_.classes_]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self._z(X)
        return Z if self.pca_ is None else self.pca_.transform(Z)

    def decision(self, X: np.ndarray) -> np.ndarray:
        """Class scores aligned with `classes_`, shape (n, n_classes)."""
        return class_scores(self.model_, self.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(self.transform(X))
