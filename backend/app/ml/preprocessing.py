from __future__ import annotations

import numpy as np


class StandardScaler:
    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = np.mean(x, axis=0)
        self.scale_ = np.std(x, axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def pca_2d(x: np.ndarray) -> np.ndarray:
    """Project samples with PCA using only NumPy SVD."""

    centered = x - np.mean(x, axis=0)
    scale = np.std(centered, axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = centered / scale
    _, _, vt = np.linalg.svd(normalized, full_matrices=False)
    components = vt[: min(2, vt.shape[0])].T
    projected = normalized @ components
    if projected.shape[1] == 1:
        projected = np.column_stack((projected[:, 0], np.zeros(len(projected))))
    return projected
