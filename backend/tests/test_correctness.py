"""正确性测试：scratch 实现与 sklearn 参考实现对照（统一规范 第 15 节）。

- SVM：Accuracy Difference < 0.05，Prediction Agreement > 0.9
- K-Means：两种实现的轮廓系数接近、簇结构一致
- PCA：解释方差比与 numpy.eigh 对齐（< 1e-8），投影按符号对齐后一致
- 指标：accuracy / precision / recall / f1 / mse / r2 与 sklearn.metrics 一致
"""

import unittest

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.svm import SVC

import backend.models  # noqa: F401  触发注册
from backend.core.factory import ModelFactory
from backend.core.runner import prepare_data
from backend.datasets.loader import load_dataset
from backend.metrics import classification, regression


def _prepared(dataset="breast_cancer"):
    bundle, X_train, X_test, y_train, y_test = prepare_data(
        {
            "dataset": dataset,
            "split": {"test_size": 0.2, "random_state": 42},
            "preprocessing": ["standard_scaler"],
        }
    )
    return bundle, X_train, X_test, y_train, y_test


class SvmCorrectnessTest(unittest.TestCase):
    def test_matches_sklearn_on_breast_cancer(self):
        _, X_train, X_test, y_train, y_test = _prepared()
        params = {"C": 1.0, "kernel": "rbf"}

        scratch = ModelFactory.create("svm", params)
        scratch.fit(X_train, y_train)
        y_scratch = scratch.predict(X_test)

        reference = SVC(C=1.0, kernel="rbf", gamma="scale")
        reference.fit(X_train, y_train)
        y_ref = reference.predict(X_test)

        acc_s, acc_r = accuracy_score(y_test, y_scratch), accuracy_score(y_test, y_ref)
        self.assertLess(abs(acc_s - acc_r), 0.05, f"scratch={acc_s:.4f} sklearn={acc_r:.4f}")
        agreement = (y_scratch == y_ref).mean()
        self.assertGreater(agreement, 0.9, f"预测一致率 {agreement:.4f}")

    def test_linear_kernel_perfectly_separable_subset(self):
        bundle = load_dataset("iris")
        X = bundle["X"][:100]  # setosa / versicolor 线性可分
        y = bundle["y"][:100]
        svm = ModelFactory.create("svm", {"C": 1.0, "kernel": "linear"})
        svm.fit(X, y)
        self.assertGreaterEqual(accuracy_score(y, svm.predict(X)), 0.99)


class KMeansCorrectnessTest(unittest.TestCase):
    def test_silhouette_close_to_sklearn(self):
        bundle, X_train, _, _, _ = _prepared("iris")
        from backend.metrics import clustering

        scratch = ModelFactory.create("kmeans", {"k": 3, "random_state": 42, "n_init": 8})
        scratch.fit(X_train)
        reference = ModelFactory.create(
            "kmeans", {"k": 3, "random_state": 42, "n_init": 10}, implementation="sklearn"
        )
        reference.fit(X_train)

        sil_s = clustering.silhouette_score(X_train, scratch.labels_)
        sil_r = clustering.silhouette_score(X_train, reference.labels_)
        self.assertGreater(sil_s, 0.3)
        self.assertLess(abs(sil_s - sil_r), 0.1, f"scratch={sil_s:.4f} sklearn={sil_r:.4f}")

    def test_centroid_history_recorded(self):
        bundle, X_train, _, _, _ = _prepared("iris")
        kmeans = ModelFactory.create("kmeans", {"k": 3, "n_init": 2})
        kmeans.fit(X_train)
        history = kmeans.get_visualization_data()["centroid_history"]
        self.assertGreaterEqual(len(history), 2)
        self.assertEqual(len(history[0]), 3)


class PcaCorrectnessTest(unittest.TestCase):
    def test_explained_variance_matches_sklearn(self):
        bundle, X_train, _, _, _ = _prepared("wine")
        scratch = ModelFactory.create("pca", {"n_components": None})
        reference = ModelFactory.create("pca", {"n_components": None}, implementation="sklearn")
        scratch.fit(X_train)
        reference.fit(X_train)
        np.testing.assert_allclose(
            scratch.explained_variance_ratio_,
            reference.explained_variance_ratio_,
            atol=1e-8,
        )

    def test_projection_sign_agreement(self):
        bundle, X_train, _, _, _ = _prepared("wine")
        proj_s = ModelFactory.create("pca", {"n_components": 2}).fit_transform(X_train)
        proj_r = ModelFactory.create(
            "pca", {"n_components": 2}, implementation="sklearn"
        ).fit_transform(X_train)
        signs = np.sign((proj_s * proj_r).sum(axis=0))
        signs[signs == 0] = 1.0
        np.testing.assert_allclose(proj_s * signs, proj_r, atol=1e-7)


class MetricsConsistencyTest(unittest.TestCase):
    def test_classification_metrics_match_sklearn(self):
        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 3, 200)
        y_pred = rng.randint(0, 3, 200)
        self.assertAlmostEqual(classification.accuracy(y_true, y_pred), accuracy_score(y_true, y_pred))
        self.assertAlmostEqual(
            classification.precision(y_true, y_pred), precision_score(y_true, y_pred, average="macro")
        )
        self.assertAlmostEqual(
            classification.recall(y_true, y_pred), recall_score(y_true, y_pred, average="macro")
        )
        self.assertAlmostEqual(
            classification.f1(y_true, y_pred), f1_score(y_true, y_pred, average="macro")
        )

    def test_regression_metrics_match_sklearn(self):
        rng = np.random.RandomState(0)
        y_true = rng.randn(100)
        y_pred = y_true + rng.randn(100) * 0.5
        self.assertAlmostEqual(regression.mse(y_true, y_pred), mean_squared_error(y_true, y_pred))
        self.assertAlmostEqual(regression.r2(y_true, y_pred), r2_score(y_true, y_pred))

    def test_explained_variance_sums_to_one(self):
        from backend.metrics import dimensionality

        ratios = dimensionality.explained_variance_ratio([4.0, 2.0, 1.0, 1.0])
        self.assertAlmostEqual(sum(ratios), 1.0)
        cumulative = dimensionality.cumulative_explained_variance(ratios)
        self.assertAlmostEqual(cumulative[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
