"""ModelFactory：统一创建模型（统一规范 第 5 节）。

统一创建方式::

    model = ModelFactory.create("knn", {"k": 5})
    model = ModelFactory.create("svm", {"C": 1.0, "kernel": "linear"}, implementation="sklearn")

杜绝主程序里出现大量 if model == "knn": ... elif model == "svm": ...
"""

from backend.core.registry import get_model_class


class ModelFactory:
    @staticmethod
    def create(name, params=None, implementation="scratch"):
        cls = get_model_class(name, implementation)
        return cls(**(params or {}))
