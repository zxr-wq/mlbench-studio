from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import WebSocket

from app.api.schemas import ExperimentRequest
from app.core.registry import algorithm_registry
from app.services.datasets import DatasetBundle, dataset_service


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale[scale < 1e-12] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def _split_indices(y: np.ndarray, method: str, test_size: float, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    if method == "random":
        rng.shuffle(indices)
        test_count = max(1, int(round(len(y) * test_size)))
        return [(indices[test_count:], indices[:test_count])]

    by_class: list[np.ndarray] = []
    for label in np.unique(y):
        label_indices = indices[y == label].copy()
        rng.shuffle(label_indices)
        by_class.append(label_indices)

    if method == "stratified":
        test_parts = []
        train_parts = []
        for label_indices in by_class:
            test_count = min(len(label_indices) - 1, max(1, int(round(len(label_indices) * test_size))))
            test_parts.append(label_indices[:test_count])
            train_parts.append(label_indices[test_count:])
        train_indices = np.concatenate(train_parts)
        test_indices = np.concatenate(test_parts)
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)
        return [(train_indices, test_indices)]

    fold_parts: list[list[np.ndarray]] = [[] for _ in range(folds)]
    for label_indices in by_class:
        for fold_index, part in enumerate(np.array_split(label_indices, folds)):
            fold_parts[fold_index].append(part)
    output = []
    for fold_index in range(folds):
        test_indices = np.concatenate(fold_parts[fold_index])
        train_indices = np.setdiff1d(indices, test_indices, assume_unique=False)
        output.append((train_indices, test_indices))
    return output


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> tuple[dict[str, float], list[list[int]]]:
    matrix = np.zeros((class_count, class_count), dtype=int)
    for truth, prediction in zip(y_true, y_pred, strict=True):
        matrix[int(truth), int(prediction)] += 1
    precision_values = []
    recall_values = []
    f1_values = []
    for class_index in range(class_count):
        true_positive = matrix[class_index, class_index]
        false_positive = np.sum(matrix[:, class_index]) - true_positive
        false_negative = np.sum(matrix[class_index, :]) - true_positive
        precision = float(true_positive / max(1, true_positive + false_positive))
        recall = float(true_positive / max(1, true_positive + false_negative))
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    values = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": float(np.mean(precision_values)),
        "recall": float(np.mean(recall_values)),
        "f1": float(np.mean(f1_values)),
    }
    return values, matrix.tolist()


def _pca_projection(x: np.ndarray) -> np.ndarray:
    centered = x - np.mean(x, axis=0)
    scale = np.std(centered, axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = centered / scale
    _, _, components = np.linalg.svd(standardized, full_matrices=False)
    if len(components) == 1:
        return np.column_stack((standardized @ components[0], np.zeros(len(x))))
    return standardized @ components[:2].T


def _train_model(
    model_id: str,
    implementation: str,
    parameters: dict[str, Any],
    bundle: DatasetBundle,
    splits: list[tuple[np.ndarray, np.ndarray]],
    standardize: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    all_true: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_test_indices: list[np.ndarray] = []
    fold_scores: list[float] = []
    training_ms = 0.0
    inference_ms = 0.0
    last_model = None

    for train_indices, test_indices in splits:
        x_train = bundle.x[train_indices]
        x_test = bundle.x[test_indices]
        if standardize:
            x_train, x_test = _standardize(x_train, x_test)
        model = algorithm_registry.create(model_id, parameters, implementation)
        start = time.perf_counter()
        model.fit(x_train, bundle.y[train_indices])
        training_ms += (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        prediction = model.predict(x_test)
        inference_ms += (time.perf_counter() - start) * 1000.0
        truth = bundle.y[test_indices]
        fold_scores.append(float(np.mean(truth == prediction)))
        all_true.append(truth)
        all_predictions.append(prediction)
        all_test_indices.append(test_indices)
        last_model = model

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_predictions)
    test_indices = np.concatenate(all_test_indices)
    metric_values, confusion = _metrics(y_true, y_pred, len(bundle.target_names))
    projection = _pca_projection(bundle.x)
    point_limit = min(len(test_indices), 700)
    chosen = np.linspace(0, len(test_indices) - 1, point_limit, dtype=int)
    points = [
        {
            "x": round(float(projection[test_indices[position], 0]), 5),
            "y": round(float(projection[test_indices[position], 1]), 5),
            "label": int(y_true[position]),
            "prediction": int(y_pred[position]),
            "correct": bool(y_true[position] == y_pred[position]),
        }
        for position in chosen
    ]

    feature_importance = []
    if last_model is not None and hasattr(last_model, "feature_importances_"):
        importance = np.asarray(last_model.feature_importances_, dtype=float)
        order = np.argsort(importance)[::-1][:8]
        feature_importance = [
            {"feature": bundle.feature_names[int(index)], "value": round(float(importance[index]), 6)}
            for index in order
            if importance[index] > 0
        ]
    spec = algorithm_registry.get(model_id)
    visualization = last_model.get_visualization_data() if last_model is not None else {}
    if visualization:
        visualization["feature_names"] = list(bundle.feature_names)
        visualization["class_names"] = list(bundle.target_names)
    result = {
        "id": f"{model_id}:{implementation}",
        "model": model_id,
        "task_type": spec.task_type,
        "implementation": implementation,
        "dataset": bundle.id,
        "name": spec.name,
        "short_name": spec.short_name,
        "family": spec.family,
        "metrics": {name: round(value, 6) for name, value in metric_values.items()},
        "confusion_matrix": confusion,
        "class_names": list(bundle.target_names),
        "training_ms": round(training_ms, 3),
        "inference_ms": round(inference_ms, 3),
        "training_time": round(training_ms / 1000.0, 6),
        "inference_time": round(inference_ms / 1000.0, 6),
        "fold_scores": [round(value, 6) for value in fold_scores],
        "projection": points,
        "training_curve": list(getattr(last_model, "history_", [])) if last_model is not None else [],
        "feature_importance": feature_importance,
        "parameters": last_model.get_params() if last_model is not None else parameters,
        "params": last_model.get_params() if last_model is not None else parameters,
        "visualization": visualization,
    }
    return result, y_pred


class ConnectionManager:
    def __init__(self) -> None:
        self.connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, experiment_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections[experiment_id].add(websocket)

    def disconnect(self, experiment_id: str, websocket: WebSocket) -> None:
        self.connections[experiment_id].discard(websocket)

    async def broadcast(self, experiment_id: str, payload: dict[str, Any]) -> None:
        stale = []
        for websocket in tuple(self.connections.get(experiment_id, set())):
            try:
                await websocket.send_json(payload)
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            self.disconnect(experiment_id, websocket)


class ExperimentService:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.manager = ConnectionManager()

    def create(self, request: ExperimentRequest) -> dict[str, Any]:
        for model_id in request.model_ids:
            algorithm_registry.get(model_id)
        dataset_service.load(request.dataset_id)
        experiment_id = uuid.uuid4().hex[:12]
        run_plan = [
            {"model": model_id, "implementation": implementation}
            for model_id in request.model_ids
            for implementation in request.implementations
            if implementation in algorithm_registry.get(model_id).public_dict()["implementations"]
        ]
        if not run_plan:
            raise ValueError("the selected models do not support the requested implementations")
        record = {
            "id": experiment_id,
            "status": "queued",
            "progress": 0,
            "stage": "实验已进入队列",
            "created_at": _iso_now(),
            "finished_at": None,
            "dataset_id": request.dataset_id,
            "model_ids": request.model_ids,
            "run_plan": run_plan,
            "config": request.model_dump(),
            "results": [],
            "best_model": None,
            "best_accuracy": None,
            "error": None,
        }
        self.records[experiment_id] = record
        return record

    def get(self, experiment_id: str) -> dict[str, Any]:
        if experiment_id not in self.records:
            raise KeyError(f"unknown experiment: {experiment_id}")
        return self.records[experiment_id]

    def list(self) -> list[dict[str, Any]]:
        return sorted(self.records.values(), key=lambda item: item["created_at"], reverse=True)[:30]

    async def _update(self, experiment_id: str, **changes: Any) -> None:
        self.records[experiment_id].update(changes)
        await self.manager.broadcast(experiment_id, {"type": "experiment", "experiment": self.records[experiment_id]})

    async def run(self, experiment_id: str, request: ExperimentRequest) -> None:
        try:
            await self._update(experiment_id, status="running", progress=4, stage="载入数据集")
            bundle = await asyncio.to_thread(dataset_service.load, request.dataset_id)
            splits = _split_indices(bundle.y, request.split_method, request.test_size, request.folds, request.seed)
            results: list[dict[str, Any]] = []
            predictions: dict[tuple[str, str], np.ndarray] = {}
            run_plan = self.records[experiment_id]["run_plan"]
            total = len(run_plan)
            for index, run_item in enumerate(run_plan):
                model_id = run_item["model"]
                implementation = run_item["implementation"]
                spec = algorithm_registry.get(model_id)
                start_progress = 10 + int(index / total * 82)
                await self._update(
                    experiment_id,
                    progress=start_progress,
                    stage=f"训练 {spec.short_name} {implementation} ({index + 1}/{total})",
                )
                result, prediction = await asyncio.to_thread(
                    _train_model,
                    model_id,
                    implementation,
                    request.parameters.get(model_id, {}),
                    bundle,
                    splits,
                    request.standardize,
                )
                results.append(result)
                predictions[(model_id, implementation)] = prediction
                scratch_key = (model_id, "scratch")
                sklearn_key = (model_id, "sklearn")
                if scratch_key in predictions and sklearn_key in predictions:
                    scratch_result = next(item for item in results if item["id"] == f"{model_id}:scratch")
                    sklearn_result = next(item for item in results if item["id"] == f"{model_id}:sklearn")
                    comparison = {
                        "accuracy_difference": round(
                            abs(scratch_result["metrics"]["accuracy"] - sklearn_result["metrics"]["accuracy"]), 6
                        ),
                        "prediction_agreement": round(
                            float(np.mean(predictions[scratch_key] == predictions[sklearn_key])), 6
                        ),
                    }
                    scratch_result["comparison"] = comparison
                    sklearn_result["comparison"] = comparison
                await self._update(
                    experiment_id,
                    results=results.copy(),
                    progress=10 + int((index + 1) / total * 82),
                    stage=f"已完成 {spec.short_name} {implementation}",
                )
            best = max(results, key=lambda item: item["metrics"]["accuracy"])
            await self._update(
                experiment_id,
                status="completed",
                progress=100,
                stage="评估完成",
                finished_at=_iso_now(),
                results=results,
                best_model=best["id"],
                best_accuracy=best["metrics"]["accuracy"],
                dataset={
                    "id": bundle.id,
                    "name": bundle.name,
                    "samples": int(bundle.x.shape[0]),
                    "features": int(bundle.x.shape[1]),
                    "classes": len(bundle.target_names),
                },
            )
        except Exception as exc:
            await self._update(
                experiment_id,
                status="failed",
                stage="实验失败",
                finished_at=_iso_now(),
                error=str(exc),
            )


experiment_service = ExperimentService()
