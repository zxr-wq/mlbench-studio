"""scikit-learn reference implementations for B's two linear models."""

from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from backend.core.base_model import BaseModel
from backend.core.registry import register_model


@register_model("linear_regression", implementation="sklearn")
class LinearRegression(BaseModel):
    task_type = "regression"

    def __init__(self, fit_intercept=True):
        self.fit_intercept = bool(fit_intercept)

    def fit(self, X, y=None):
        if y is None: raise ValueError("LinearRegression requires y")
        self.model_ = SklearnLinearRegression(fit_intercept=self.fit_intercept).fit(X, y)
        return self

    def predict(self, X):
        if not hasattr(self, "model_"): raise RuntimeError("fit must be called before predict")
        return self.model_.predict(X)

    def get_params(self): return {"fit_intercept": self.fit_intercept}
    def get_visualization_data(self): return {}


@register_model("logistic_regression", implementation="sklearn")
class LogisticRegression(BaseModel):
    task_type = "classification"

    def __init__(self, learning_rate=0.1, max_iter=800, l2=0.01):
        self.learning_rate, self.max_iter, self.l2 = float(learning_rate), int(max_iter), float(l2)

    def fit(self, X, y=None):
        if y is None: raise ValueError("LogisticRegression requires y")
        c_value = 1 / self.l2 if self.l2 > 0 else 1e9
        self.model_ = SklearnLogisticRegression(C=c_value, max_iter=self.max_iter, solver="lbfgs", random_state=42).fit(X, y)
        return self

    def predict(self, X):
        if not hasattr(self, "model_"): raise RuntimeError("fit must be called before predict")
        return self.model_.predict(X)

    def predict_proba(self, X): return self.model_.predict_proba(X)
    def get_params(self): return {"learning_rate": self.learning_rate, "max_iter": self.max_iter, "l2": self.l2}
    def get_visualization_data(self): return {}
