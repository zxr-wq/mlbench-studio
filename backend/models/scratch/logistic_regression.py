"""Scratch multiclass logistic regression using softmax gradient descent."""

import numpy as np

from backend.core.base_model import BaseModel
from backend.core.registry import register_model


@register_model("logistic_regression")
class LogisticRegression(BaseModel):
    task_type = "classification"

    def __init__(self, learning_rate=0.1, max_iter=800, l2=0.01):
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.l2 = float(l2)
        # The shared interface is also consumed before a model is fitted.
        # Keep the visualization payload stable in that state.
        self.loss_history_ = []

    @staticmethod
    def _softmax(scores):
        values = np.exp(scores - scores.max(axis=1, keepdims=True))
        return values / values.sum(axis=1, keepdims=True)

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("LogisticRegression requires y")
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be 2D with one class label per row")
        self.classes_, encoded = np.unique(y, return_inverse=True)
        target = np.eye(len(self.classes_))[encoded]
        self.weights_ = np.zeros((X.shape[1], len(self.classes_)))
        self.bias_ = np.zeros(len(self.classes_))
        self.loss_history_ = []
        every = max(1, self.max_iter // 40)
        for iteration in range(self.max_iter):
            probabilities = self._softmax(X @ self.weights_ + self.bias_)
            error = probabilities - target
            self.weights_ -= self.learning_rate * ((X.T @ error) / len(X) + self.l2 * self.weights_)
            self.bias_ -= self.learning_rate * error.mean(axis=0)
            if iteration % every == 0 or iteration == self.max_iter - 1:
                loss = -np.mean(np.sum(target * np.log(probabilities + 1e-12), axis=1)) + .5 * self.l2 * float((self.weights_ ** 2).sum())
                self.loss_history_.append({"epoch": iteration + 1, "loss": float(loss)})
        return self

    def predict_proba(self, X):
        if not hasattr(self, "weights_"):
            raise RuntimeError("fit must be called before predict")
        return self._softmax(np.asarray(X, dtype=float) @ self.weights_ + self.bias_)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def get_params(self):
        return {"learning_rate": self.learning_rate, "max_iter": self.max_iter, "l2": self.l2}

    def get_visualization_data(self):
        return {"type": "logistic_regression_loss", "loss_history": self.loss_history_}
