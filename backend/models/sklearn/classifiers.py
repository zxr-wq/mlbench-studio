"""Scikit-learn reference implementations for the A and D classifiers."""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from backend.core.base_model import BaseModel
from backend.core.registry import register_model


class _SklearnClassifier(BaseModel):
    task_type = "classification"
    estimator_class = None

    def fit(self, X, y=None):
        if y is None: raise ValueError("classification models require y")
        self.estimator_ = self.estimator_class(**self._estimator_params()).fit(X, y)
        return self

    def predict(self, X):
        if not hasattr(self, "estimator_"): raise RuntimeError("fit must be called before predict")
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def get_visualization_data(self):
        return {}


@register_model("knn", implementation="sklearn")
class SklearnKNN(_SklearnClassifier):
    estimator_class = KNeighborsClassifier
    def __init__(self, k=5, distance="euclidean", weights="uniform"):
        self.k, self.distance, self.weights = int(k), distance, weights
    def _estimator_params(self): return {"n_neighbors": self.k, "metric": self.distance, "weights": self.weights}
    def get_params(self): return {"k": self.k, "distance": self.distance, "weights": self.weights}


@register_model("naive_bayes", implementation="sklearn")
class SklearnGaussianNB(_SklearnClassifier):
    estimator_class = GaussianNB
    def __init__(self, var_smoothing=1e-9): self.var_smoothing = float(var_smoothing)
    def _estimator_params(self): return {"var_smoothing": self.var_smoothing}
    def get_params(self): return {"var_smoothing": self.var_smoothing}


@register_model("decision_tree", implementation="sklearn")
class SklearnDecisionTree(_SklearnClassifier):
    estimator_class = DecisionTreeClassifier
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1): self.max_depth, self.min_samples_split, self.min_samples_leaf = max_depth, int(min_samples_split), int(min_samples_leaf)
    def _estimator_params(self): return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split, "min_samples_leaf": self.min_samples_leaf, "random_state": 42}
    def get_params(self): return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split, "min_samples_leaf": self.min_samples_leaf}


@register_model("random_forest", implementation="sklearn")
class SklearnRandomForest(_SklearnClassifier):
    estimator_class = RandomForestClassifier
    def __init__(self, n_estimators=100, max_depth=None, max_features=None, min_samples_split=2, min_samples_leaf=1, random_state=42):
        self.n_estimators, self.max_depth, self.max_features, self.min_samples_split, self.min_samples_leaf, self.random_state = int(n_estimators), max_depth, max_features, int(min_samples_split), int(min_samples_leaf), int(random_state)
    def _estimator_params(self): return self.get_params() | {"n_estimators": self.n_estimators, "random_state": self.random_state}
    def get_params(self): return {"n_estimators": self.n_estimators, "max_depth": self.max_depth, "max_features": self.max_features, "min_samples_split": self.min_samples_split, "min_samples_leaf": self.min_samples_leaf, "random_state": self.random_state}


@register_model("gradient_boosting", implementation="sklearn")
class SklearnGradientBoosting(_SklearnClassifier):
    estimator_class = GradientBoostingClassifier
    def __init__(self, n_estimators=100, learning_rate=.1, max_depth=2, min_samples_leaf=1): self.n_estimators, self.learning_rate, self.max_depth, self.min_samples_leaf = int(n_estimators), float(learning_rate), int(max_depth), int(min_samples_leaf)
    def _estimator_params(self): return self.get_params() | {"random_state": 42}
    def get_params(self): return {"n_estimators": self.n_estimators, "learning_rate": self.learning_rate, "max_depth": self.max_depth, "min_samples_leaf": self.min_samples_leaf}
