from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.ml.base import Classifier


class KNNClassifier(Classifier):
    def __init__(self, k: int = 5, distance: str = "euclidean", weights: str = "distance") -> None:
        super().__init__()
        self.k = int(k)
        self.distance = str(distance)
        self.weights = str(weights)

    def _validate_params(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be a positive integer")
        if self.distance not in {"euclidean", "manhattan"}:
            raise ValueError("distance must be 'euclidean' or 'manhattan'")
        if self.weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'")

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self._validate_params()
        self.x_train_ = np.asarray(x, dtype=float)
        self.y_train_ = np.asarray(y, dtype=int)
        if self.x_train_.ndim != 2 or self.y_train_.ndim != 1:
            raise ValueError("x must be 2D and y must be 1D")
        if len(self.x_train_) != len(self.y_train_) or len(self.x_train_) == 0:
            raise ValueError("x and y must contain the same non-zero number of samples")
        if self.k > len(self.x_train_):
            raise ValueError("k cannot exceed the number of training samples")
        if not np.isfinite(self.x_train_).all():
            raise ValueError("x contains non-finite values")
        self.classes_ = np.unique(self.y_train_)
        return self

    def _neighbors(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "x_train_"):
            raise RuntimeError("fit must be called before predict")
        if x.ndim != 2 or x.shape[1] != self.x_train_.shape[1]:
            raise ValueError("x has an incompatible feature shape")
        differences = x[:, None, :] - self.x_train_[None, :, :]
        if self.distance == "manhattan":
            distances = np.sum(np.abs(differences), axis=2)
        else:
            distances = np.sqrt(np.sum(differences**2, axis=2))
        indices = np.argpartition(distances, kth=self.k - 1, axis=1)[:, : self.k]
        ordered = np.take_along_axis(distances, indices, axis=1).argsort(axis=1)
        indices = np.take_along_axis(indices, ordered, axis=1)
        nearest_distances = np.take_along_axis(distances, indices, axis=1)
        return indices, nearest_distances

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        predictions: list[int] = []
        all_indices = []
        all_distances = []
        for start in range(0, len(x), 128):
            batch = x[start : start + 128]
            neighbor_indices, neighbor_distances = self._neighbors(batch)
            all_indices.append(neighbor_indices)
            all_distances.append(neighbor_distances)
            for row, indices in enumerate(neighbor_indices):
                labels = self.y_train_[indices]
                if self.weights == "distance":
                    vote_weights = 1.0 / np.maximum(neighbor_distances[row], 1e-12)
                    scores = [float(np.sum(vote_weights[labels == label])) for label in self.classes_]
                else:
                    scores = [float(np.sum(labels == label)) for label in self.classes_]
                predictions.append(int(self.classes_[int(np.argmax(scores))]))
        self.last_queries_ = x.copy()
        self.last_neighbor_indices_ = np.vstack(all_indices) if all_indices else np.empty((0, self.k), dtype=int)
        self.last_neighbor_distances_ = np.vstack(all_distances) if all_distances else np.empty((0, self.k))
        return np.asarray(predictions, dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        neighbor_indices, neighbor_distances = self._neighbors(x)
        probabilities = np.zeros((len(x), len(self.classes_)), dtype=float)
        for row, indices in enumerate(neighbor_indices):
            labels = self.y_train_[indices]
            vote_weights = (
                1.0 / np.maximum(neighbor_distances[row], 1e-12)
                if self.weights == "distance"
                else np.ones(self.k)
            )
            for class_index, label in enumerate(self.classes_):
                probabilities[row, class_index] = float(np.sum(vote_weights[labels == label]))
        totals = np.sum(probabilities, axis=1, keepdims=True)
        return probabilities / np.maximum(totals, 1e-12)

    def get_visualization_data(self) -> dict[str, object]:
        if not hasattr(self, "last_neighbor_indices_") or len(self.last_neighbor_indices_) == 0:
            return {}
        query = self.last_queries_[0]
        indices = self.last_neighbor_indices_[0]
        neighborhood = np.vstack((query, self.x_train_[indices]))
        centered = neighborhood - np.mean(neighborhood, axis=0)
        _, _, components = np.linalg.svd(centered, full_matrices=False)
        if centered.shape[1] == 1:
            projection = np.column_stack((centered[:, 0], np.zeros(len(centered))))
        else:
            projection = centered @ components[:2].T
            if projection.shape[1] == 1:
                projection = np.column_stack((projection[:, 0], np.zeros(len(projection))))
        return {
            "type": "knn_neighbors",
            "query": {"x": float(projection[0, 0]), "y": float(projection[0, 1])},
            "neighbors": [
                {
                    "rank": rank + 1,
                    "x": float(projection[rank + 1, 0]),
                    "y": float(projection[rank + 1, 1]),
                    "label": int(self.y_train_[index]),
                    "distance": float(self.last_neighbor_distances_[0, rank]),
                }
                for rank, index in enumerate(indices)
            ],
        }


class GaussianNBClassifier(Classifier):
    def __init__(self, var_smoothing: float = 1e-9) -> None:
        super().__init__()
        self.var_smoothing = float(var_smoothing)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianNBClassifier":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int)
        if x.ndim != 2 or y.ndim != 1:
            raise ValueError("x must be 2D and y must be 1D")
        if len(x) != len(y) or len(x) == 0:
            raise ValueError("x and y must contain the same non-zero number of samples")
        if self.var_smoothing < 0:
            raise ValueError("var_smoothing must be non-negative")
        self.classes_ = np.unique(y)
        self.means_ = np.vstack([np.mean(x[y == label], axis=0) for label in self.classes_])
        variances = np.vstack([np.var(x[y == label], axis=0) for label in self.classes_])
        epsilon = self.var_smoothing * max(float(np.var(x, axis=0).max()), 1.0)
        self.variances_ = variances + epsilon
        self.log_priors_ = np.log([np.mean(y == label) for label in self.classes_])
        return self

    def _joint_log_likelihood(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "means_"):
            raise RuntimeError("fit must be called before predict")
        x = np.asarray(x, dtype=float)
        log_probabilities = []
        for class_index in range(len(self.classes_)):
            mean = self.means_[class_index]
            variance = self.variances_[class_index]
            log_likelihood = -0.5 * np.sum(np.log(2.0 * np.pi * variance) + ((x - mean) ** 2) / variance, axis=1)
            log_probabilities.append(log_likelihood + self.log_priors_[class_index])
        return np.column_stack(log_probabilities)

    def predict(self, x: np.ndarray) -> np.ndarray:
        log_probabilities = self._joint_log_likelihood(x)
        return self.classes_[np.argmax(log_probabilities, axis=1)]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        log_probabilities = self._joint_log_likelihood(x)
        shifted = log_probabilities - np.max(log_probabilities, axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities / np.sum(probabilities, axis=1, keepdims=True)

    def get_visualization_data(self) -> dict[str, object]:
        if not hasattr(self, "means_"):
            return {}
        return {
            "type": "naive_bayes_stats",
            "classes": [int(value) for value in self.classes_],
            "class_priors": np.exp(self.log_priors_).tolist(),
            "means": self.means_.tolist(),
            "variances": self.variances_.tolist(),
        }


class SoftmaxRegression(Classifier):
    def __init__(
        self,
        learning_rate: float = 0.08,
        epochs: int = 300,
        l2: float = 0.001,
    ) -> None:
        super().__init__()
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.l2 = float(l2)

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SoftmaxRegression":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        sample_count, feature_count = x.shape
        class_count = len(self.classes_)
        self.weights_ = np.zeros((feature_count, class_count), dtype=float)
        self.bias_ = np.zeros(class_count, dtype=float)
        targets = np.eye(class_count)[encoded]
        report_every = max(1, self.epochs // 30)

        for epoch in range(max(1, self.epochs)):
            probabilities = self._softmax(x @ self.weights_ + self.bias_)
            error = probabilities - targets
            weight_gradient = (x.T @ error) / sample_count + self.l2 * self.weights_
            bias_gradient = np.mean(error, axis=0)
            self.weights_ -= self.learning_rate * weight_gradient
            self.bias_ -= self.learning_rate * bias_gradient
            if epoch % report_every == 0 or epoch == self.epochs - 1:
                loss = -np.mean(np.sum(targets * np.log(probabilities + 1e-12), axis=1))
                loss += 0.5 * self.l2 * float(np.sum(self.weights_**2))
                self.history_.append({"epoch": float(epoch + 1), "loss": float(loss)})
        self.feature_importances_ = np.mean(np.abs(self.weights_), axis=1)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        probabilities = self._softmax(np.asarray(x, dtype=float) @ self.weights_ + self.bias_)
        return self.classes_[np.argmax(probabilities, axis=1)]


class LinearSVMClassifier(Classifier):
    def __init__(self, learning_rate: float = 0.025, epochs: int = 260, regularization: float = 0.01) -> None:
        super().__init__()
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.regularization = float(regularization)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearSVMClassifier":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        sample_count, feature_count = x.shape
        class_count = len(self.classes_)
        self.weights_ = np.zeros((feature_count, class_count), dtype=float)
        self.bias_ = np.zeros(class_count, dtype=float)
        targets = -np.ones((sample_count, class_count), dtype=float)
        targets[np.arange(sample_count), encoded] = 1.0
        report_every = max(1, self.epochs // 30)

        for epoch in range(max(1, self.epochs)):
            scores = x @ self.weights_ + self.bias_
            margins = 1.0 - targets * scores
            active = margins > 0
            signed_active = targets * active
            weight_gradient = -(x.T @ signed_active) / sample_count + self.regularization * self.weights_
            bias_gradient = -np.mean(signed_active, axis=0)
            schedule = self.learning_rate / (1.0 + 0.006 * epoch)
            self.weights_ -= schedule * weight_gradient
            self.bias_ -= schedule * bias_gradient
            if epoch % report_every == 0 or epoch == self.epochs - 1:
                loss = float(np.mean(np.maximum(0.0, margins)))
                loss += 0.5 * self.regularization * float(np.sum(self.weights_**2))
                self.history_.append({"epoch": float(epoch + 1), "loss": loss})
        self.feature_importances_ = np.mean(np.abs(self.weights_), axis=1)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = np.asarray(x, dtype=float) @ self.weights_ + self.bias_
        return self.classes_[np.argmax(scores, axis=1)]


@dataclass
class _TreeNode:
    prediction: int
    feature: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None


class DecisionTreeClassifier(Classifier):
    def __init__(self, max_depth: int = 6, min_samples_split: int = 4, min_samples_leaf: int = 2) -> None:
        super().__init__()
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return float(1.0 - np.sum(probabilities**2))

    @staticmethod
    def _majority(y: np.ndarray) -> int:
        values, counts = np.unique(y, return_counts=True)
        return int(values[int(np.argmax(counts))])

    def _candidate_thresholds(self, values: np.ndarray) -> np.ndarray:
        unique = np.unique(values)
        if len(unique) <= 1:
            return np.empty(0)
        if len(unique) > 64:
            unique = np.unique(np.quantile(unique, np.linspace(0.02, 0.98, 48)))
        return (unique[:-1] + unique[1:]) / 2.0

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        node = _TreeNode(prediction=self._majority(y))
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return node

        parent_impurity = self._gini(y)
        best_gain = 0.0
        best_feature: int | None = None
        best_threshold: float | None = None
        best_mask: np.ndarray | None = None

        for feature in range(x.shape[1]):
            for threshold in self._candidate_thresholds(x[:, feature]):
                mask = x[:, feature] <= threshold
                left_count = int(np.sum(mask))
                right_count = len(y) - left_count
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue
                impurity = (left_count * self._gini(y[mask]) + right_count * self._gini(y[~mask])) / len(y)
                gain = parent_impurity - impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = float(threshold)
                    best_mask = mask

        if best_feature is None or best_mask is None:
            return node
        self.feature_importances_[best_feature] += best_gain * len(y)
        node.feature = best_feature
        node.threshold = best_threshold
        node.left = self._build(x[best_mask], y[best_mask], depth + 1)
        node.right = self._build(x[~best_mask], y[~best_mask], depth + 1)
        return node

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        self.feature_importances_ = np.zeros(x.shape[1], dtype=float)
        self.root_ = self._build(x, y, 0)
        total = float(np.sum(self.feature_importances_))
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _predict_one(self, row: np.ndarray) -> int:
        node = self.root_
        while node.feature is not None and node.left is not None and node.right is not None:
            node = node.left if row[node.feature] <= float(node.threshold) else node.right
        return node.prediction

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray([self._predict_one(row) for row in np.asarray(x, dtype=float)], dtype=int)
