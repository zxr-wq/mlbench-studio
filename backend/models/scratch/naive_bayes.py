"""Scratch Gaussian Naive Bayes classifier."""

import numpy as np

from backend.core.base_model import BaseModel
from backend.core.registry import register_model


@register_model("naive_bayes")
class GaussianNaiveBayes(BaseModel):
    task_type = "classification"

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = float(var_smoothing)

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Naive Bayes is a classifier and requires y")
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2D and contain one row per y value")
        self.classes_ = np.unique(y)
        self.means_ = np.vstack([X[y == label].mean(axis=0) for label in self.classes_])
        epsilon = self.var_smoothing * max(float(X.var(axis=0).max()), 1.0)
        self.variances_ = np.vstack([X[y == label].var(axis=0) for label in self.classes_]) + epsilon
        self.log_priors_ = np.log(np.array([(y == label).mean() for label in self.classes_]))
        return self

    def _log_likelihood(self, X):
        if not hasattr(self, "means_"):
            raise RuntimeError("fit must be called before predict")
        X = np.asarray(X, dtype=float)
        values = []
        for mean, variance, prior in zip(self.means_, self.variances_, self.log_priors_):
            values.append(prior - 0.5 * np.sum(np.log(2 * np.pi * variance) + (X - mean) ** 2 / variance, axis=1))
        return np.column_stack(values)

    def predict(self, X):
        return self.classes_[self._log_likelihood(X).argmax(axis=1)]

    def predict_proba(self, X):
        scores = self._log_likelihood(X)
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def get_params(self):
        return {"var_smoothing": self.var_smoothing}

    def get_visualization_data(self):
        if not hasattr(self, "means_"):
            return {}
        return {"type": "naive_bayes_stats", "classes": self.classes_.tolist(), "means": self.means_.tolist(), "variances": self.variances_.tolist()}
