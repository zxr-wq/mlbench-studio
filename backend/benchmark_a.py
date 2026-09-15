from __future__ import annotations

import json
import time
from collections.abc import Callable

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.ml.models import GaussianNBClassifier, KNNClassifier
from app.ml.sklearn_models import SklearnGaussianNBClassifier, SklearnKNNClassifier


DATASETS: dict[str, Callable] = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}
MODELS = {
    "knn": {
        "scratch": lambda: KNNClassifier(k=5, distance="euclidean", weights="distance"),
        "sklearn": lambda: SklearnKNNClassifier(k=5, distance="euclidean", weights="distance"),
    },
    "naive_bayes": {
        "scratch": GaussianNBClassifier,
        "sklearn": SklearnGaussianNBClassifier,
    },
}


def evaluate(factory: Callable, x_train, y_train, x_test, y_test, repeats: int = 5) -> dict[str, object]:
    training_times = []
    inference_times = []
    prediction = None
    for _ in range(repeats):
        model = factory()
        start = time.perf_counter()
        model.fit(x_train, y_train)
        training_times.append((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        prediction = model.predict(x_test)
        inference_times.append((time.perf_counter() - start) * 1000.0)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, prediction, average="macro", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_test, prediction)), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "training_ms_mean": round(float(np.mean(training_times)), 4),
        "inference_ms_mean": round(float(np.mean(inference_times)), 4),
        "prediction": prediction,
    }


def run() -> dict[str, object]:
    rows = []
    parameter_experiments = []
    for dataset_name, loader in DATASETS.items():
        dataset = loader()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.data,
            dataset.target,
            test_size=0.2,
            random_state=42,
            stratify=dataset.target,
        )
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        for model_name, implementations in MODELS.items():
            evaluated = {
                name: evaluate(factory, x_train, y_train, x_test, y_test)
                for name, factory in implementations.items()
            }
            agreement = float(np.mean(evaluated["scratch"]["prediction"] == evaluated["sklearn"]["prediction"]))
            for result in evaluated.values():
                result.pop("prediction")
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "accuracy_difference": round(
                        abs(evaluated["scratch"]["accuracy"] - evaluated["sklearn"]["accuracy"]), 6
                    ),
                    "prediction_agreement": round(agreement, 6),
                    "results": evaluated,
                }
            )
        parameter_experiments.append(
            {
                "dataset": dataset_name,
                "knn_k_accuracy": {
                    str(k): evaluate(
                        lambda k=k: KNNClassifier(k=k, distance="euclidean", weights="distance"),
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        repeats=1,
                    )["accuracy"]
                    for k in (1, 3, 5, 7, 9, 11, 15)
                },
                "naive_bayes_smoothing_accuracy": {
                    str(value): evaluate(
                        lambda value=value: GaussianNBClassifier(var_smoothing=value),
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        repeats=1,
                    )["accuracy"]
                    for value in (1e-12, 1e-9, 1e-6, 1e-3)
                },
            }
        )
    return {
        "protocol": {
            "test_size": 0.2,
            "random_state": 42,
            "stratified": True,
            "preprocessing": "StandardScaler fitted on training data only",
            "timing_repeats": 5,
        },
        "rows": rows,
        "parameter_experiments": parameter_experiments,
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
