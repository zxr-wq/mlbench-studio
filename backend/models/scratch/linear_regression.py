"""Scratch ordinary least squares linear regression."""

import numpy as np

from backend.core.base_model import BaseModel
from backend.core.registry import register_model


@register_model("linear_regression")
class LinearRegression(BaseModel):
    task_type = "regression"

    def __init__(self, fit_intercept=True):
        self.fit_intercept = bool(fit_intercept)

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("LinearRegression requires y")
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be 2D with one target value per row")
        design = np.column_stack([np.ones(len(X)), X]) if self.fit_intercept else X
        solution = np.linalg.pinv(design) @ y
        if self.fit_intercept:
            self.intercept_, self.coef_ = float(solution[0]), solution[1:]
        else:
            self.intercept_, self.coef_ = 0.0, solution
        residual = y - self.predict(X)
        self.loss_history_ = [{"epoch": 1, "loss": float(np.mean(residual ** 2))}]
        return self

    def predict(self, X):
        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before predict")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def get_params(self):
        return {"fit_intercept": self.fit_intercept}

    def get_visualization_data(self):
        return {"type": "linear_regression", "loss_history": getattr(self, "loss_history_", [])}
