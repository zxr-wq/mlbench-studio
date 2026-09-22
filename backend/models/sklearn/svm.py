"""SVM 的 sklearn 参考实现（统一规范 第 15 节）。

只用于验证 scratch 实现是否基本正确（Accuracy Difference / Prediction Agreement /
Training Time），项目主要展示 scratch 版本。
"""

import numpy as np
from sklearn.svm import SVC

from backend.core.base_model import BaseModel
from backend.core.registry import register_model


@register_model("svm", implementation="sklearn")
class SVM(BaseModel):
    task_type = "classification"

    def __init__(self, C=1.0, kernel="rbf", gamma="scale", degree=3, tol=1e-3, max_iter=-1):
        self.C = float(C)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self._model = None

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("SVM 是分类模型，fit 需要 y：model.fit(X, y)")
        self._model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        self._model.fit(np.asarray(X, dtype=float), np.asarray(y))
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit")
        return self._model.predict(np.asarray(X, dtype=float))

    def get_params(self):
        return {
            "C": self.C,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "degree": self.degree,
            "tol": self.tol,
            "max_iter": self.max_iter,
        }

    def get_visualization_data(self):
        return {
            "support_vector_count": int(self._model.n_support_.sum()) if self._model else 0,
            "classes": self._model.classes_.tolist() if self._model else [],
        }
