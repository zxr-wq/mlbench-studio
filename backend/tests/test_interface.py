"""接口合规测试（统一规范 第 1/2/3/5/12 节）。

验证所有已注册模型满足 BaseModel 契约：
- task_type 只能是四种取值之一
- fit(self, X, y=None) 签名统一
- 统一 get_params() / get_visualization_data()
- 分类/回归 predict()；K-Means predict + fit_predict；PCA transform + fit_transform（无 predict）
"""

import inspect
import unittest

import numpy as np

import backend.models  # noqa: F401  触发注册
from backend.core.base_model import TASK_TYPES, BaseModel
from backend.core.factory import ModelFactory
from backend.core.registry import registered_names, register_model
from backend.core.runner import run_experiment
from backend.datasets.loader import load_dataset


class InterfaceContractTest(unittest.TestCase):
    def test_task_type_in_allowed_values(self):
        for implementation in ("scratch", "sklearn"):
            for name in registered_names(implementation):
                model = ModelFactory.create(name, {}, implementation)
                self.assertIn(model.task_type, TASK_TYPES, f"{name}/{implementation} task_type 非法")

    def test_fit_signature_accepts_y_none(self):
        for implementation in ("scratch", "sklearn"):
            for name in registered_names(implementation):
                model = ModelFactory.create(name, {}, implementation)
                params = list(inspect.signature(model.fit).parameters)
                # 绑定方法的 signature 不含 self，统一签名 fit(self, X, y=None)
                self.assertEqual(
                    params, ["X", "y"], f"{name}/{implementation}.fit 必须是 fit(self, X, y=None)"
                )
                self.assertEqual(
                    inspect.signature(model.fit).parameters["y"].default,
                    None,
                    f"{name}/{implementation}.fit 的 y 必须默认为 None",
                )

    def test_get_params_and_visualization(self):
        for implementation in ("scratch", "sklearn"):
            for name in registered_names(implementation):
                model = ModelFactory.create(name, {}, implementation)
                self.assertIsInstance(model.get_params(), dict)
                self.assertIsInstance(model.get_visualization_data(), dict)

    def test_task_specific_methods(self):
        bundle = load_dataset("iris")
        X, y = bundle["X"][:60], bundle["y"][:60]

        # 分类：predict(X)
        svm = ModelFactory.create("svm", {"C": 1.0, "kernel": "linear"})
        svm.fit(X, y)
        self.assertEqual(len(svm.predict(X)), len(X))
        with self.assertRaises(ValueError):
            ModelFactory.create("svm", {"kernel": "linear"}).fit(X)  # 分类必须给 y

        # 聚类：predict + fit_predict，且不使用 y
        kmeans = ModelFactory.create("kmeans", {"k": 3, "n_init": 2})
        labels = kmeans.fit_predict(X)
        self.assertEqual(len(labels), len(X))
        again = ModelFactory.create("kmeans", {"k": 3, "n_init": 2})
        again.fit(X, y)  # y 被忽略，不能影响结果
        np.testing.assert_array_equal(labels, again.labels_)

        # 降维：transform + fit_transform，不实现 predict
        pca = ModelFactory.create("pca", {"n_components": 2})
        projection = pca.fit_transform(X)
        self.assertEqual(projection.shape, (len(X), 2))
        np.testing.assert_allclose(pca.transform(X), projection)
        self.assertFalse(hasattr(pca, "predict"), "PCA 不应实现 predict（统一规范 第 2 节）")

    def test_registry_prevents_duplicate(self):
        with self.assertRaises(ValueError):

            @register_model("svm")
            class Duplicate(BaseModel):
                task_type = "classification"

                def fit(self, X, y=None):
                    return self


class UnifiedResultTest(unittest.TestCase):
    """统一 Result 外层结构（第 12 节）：具体指标从 metrics 动态读取。"""

    RESULT_KEYS = {
        "model",
        "task_type",
        "implementation",
        "dataset",
        "params",
        "metrics",
        "training_time",
        "inference_time",
        "visualization",
    }

    def _run(self, name, params, metrics):
        return run_experiment(
            {
                "dataset": "iris",
                "split": {"test_size": 0.2, "random_state": 42},
                "preprocessing": ["standard_scaler"],
                "model": {"name": name, "implementation": "scratch", "params": params},
                "metrics": metrics,
            }
        )

    def test_result_shape(self):
        result = self._run("svm", {"C": 1.0, "kernel": "linear"}, ["accuracy", "f1"])
        self.assertEqual(set(result), self.RESULT_KEYS)
        self.assertEqual(result["task_type"], "classification")
        self.assertEqual(set(result["metrics"]), {"accuracy", "f1"})
        self.assertGreaterEqual(result["training_time"], 0)
        self.assertGreaterEqual(result["inference_time"], 0)

    def test_pca_metrics_not_accuracy(self):
        result = self._run("pca", {"n_components": 0.95}, [])
        self.assertEqual(result["task_type"], "dimensionality_reduction")
        self.assertNotIn("accuracy", result["metrics"])
        self.assertIn("explained_variance_ratio", result["metrics"])
        self.assertIn("cumulative_explained_variance", result["metrics"])

    def test_clustering_metrics(self):
        result = self._run("kmeans", {"k": 3, "n_init": 4}, [])
        self.assertEqual(set(result["metrics"]), {"silhouette_score", "inertia"})

    def test_unsupported_metric_rejected(self):
        with self.assertRaises(ValueError):
            self._run("pca", {"n_components": 2}, ["accuracy"])  # PCA 不算 Accuracy

    def test_visualization_payload(self):
        result = self._run("kmeans", {"k": 3, "n_init": 2}, [])
        self.assertIn("centroid_history", result["visualization"])
        result = self._run("pca", {"n_components": 2}, [])
        self.assertIn("explained_variance_ratio", result["visualization"])
        self.assertIn("projection", result["visualization"])


if __name__ == "__main__":
    unittest.main()
