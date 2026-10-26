"""HTTP and WebSocket server that delegates every experiment to core.runner."""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import backend.models  # noqa: F401 - model modules register themselves
from backend.core.registry import get_model_class, registered_names
from backend.core.runner import run_experiment
from backend.core.runner import METRIC_KEYS, SUPPORTED_PREPROCESSING
from backend.core.factory import ModelFactory
from backend.datasets.loader import load_dataset
from backend.server.history import load_records, save_record


app = FastAPI(title="MLBench Studio API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    records: dict[str, dict] = load_records()
except Exception:
    logging.exception("Unable to read local experiment history")
    records = {}
connections: dict[str, set[WebSocket]] = {}


def now():
    return datetime.now(timezone.utc).isoformat()


async def publish(experiment_id: str):
    record = records[experiment_id]
    stale = []
    for socket in tuple(connections.get(experiment_id, set())):
        try:
            await socket.send_json({"type": "experiment", "experiment": record})
        except Exception:
            stale.append(socket)
    for socket in stale:
        connections.get(experiment_id, set()).discard(socket)


def normalize_config(payload: dict) -> dict:
    """Validate the one public Config shape before it reaches the Runner."""
    try:
        model = payload["model"]
        name = model["name"]
        implementation = model.get("implementation", "scratch")
        dataset = payload["dataset"]
    except (TypeError, KeyError) as error:
        raise ValueError("config requires dataset and model.name") from error
    model_class = get_model_class(name, implementation)
    bundle = load_dataset(dataset)
    split = payload.get("split") or {}
    if not isinstance(split, dict):
        raise ValueError("split must be an object")
    test_size = float(split.get("test_size", 0.2))
    if not 0 < test_size < 1:
        raise ValueError("split.test_size must be between 0 and 1")
    evaluation = payload.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be an object")
    strategy = evaluation.get("strategy", "holdout")
    repetitions = int(evaluation.get("repetitions", 1))
    n_splits = int(evaluation.get("n_splits", 5))
    if strategy not in {"holdout", "kfold"}:
        raise ValueError("evaluation.strategy must be holdout or kfold")
    if repetitions not in {1, 3, 5}:
        raise ValueError("evaluation.repetitions must be 1, 3, or 5")
    if not 2 <= n_splits <= 10:
        raise ValueError("evaluation.n_splits must be between 2 and 10")
    preprocessing = payload.get("preprocessing") or []
    metrics = payload.get("metrics") or []
    params = model.get("params") or {}
    if not isinstance(preprocessing, list) or any(step not in SUPPORTED_PREPROCESSING for step in preprocessing):
        raise ValueError("preprocessing must be a list of supported scaler names")
    if not isinstance(metrics, list) or any(metric not in METRIC_KEYS[model_class.task_type] for metric in metrics):
        raise ValueError("metrics do not match the model task type")
    if not isinstance(params, dict):
        raise ValueError("model.params must be an object")
    for key in ("C", "learning_rate", "tol", "var_smoothing", "l2"):
        if key in params:
            number = float(params[key])
            if not math.isfinite(number) or number < 0 or (key in {"C", "learning_rate", "tol"} and number == 0):
                raise ValueError(f"model.params.{key} 应为有效的正数（平滑和正则项允许为零）")
    for key in ("k", "n_estimators", "max_depth", "n_init", "min_samples_leaf", "min_samples_split"):
        if key in params and params[key] is not None:
            number = float(params[key])
            if not math.isfinite(number) or number < 1 or not number.is_integer():
                raise ValueError(f"model.params.{key} 应为正整数")
    if "k" in params and int(params["k"]) >= int(len(bundle["X"]) * (1 - test_size)):
        raise ValueError("K 应小于训练集样本数")
    if name == "svm" and params.get("kernel", "rbf") not in {"linear", "rbf", "poly"}:
        raise ValueError("SVM 核函数应为 linear / rbf / poly")
    if "roc_auc" in metrics and (len(bundle["target_names"]) != 2 or not hasattr(model_class, "predict_proba")):
        raise ValueError("ROC-AUC 仅支持能输出概率的二分类模型")
    if model_class.task_type in {"classification", "regression"} and bundle["task_type"] != model_class.task_type:
        raise ValueError("模型与数据集的任务类型不兼容")
    if strategy == "kfold" and model_class.task_type not in {"classification", "regression"}:
        raise ValueError("聚类和降维请使用固定划分")
    seed = int(split.get("random_state", 42))
    if not 0 <= seed <= 2**32 - 1 - repetitions:
        raise ValueError("随机种子应为 0 到 4294967290 之间的整数")
    if name == "pca" and isinstance(params.get("n_components"), int) and not 1 <= params["n_components"] <= len(bundle["feature_names"]):
        raise ValueError("PCA 主成分数不能超过数据集特征数")
    try:
        ModelFactory.create(name, params, implementation)
    except (TypeError, ValueError) as error:
        raise ValueError(f"模型参数无效：{error}") from error
    return {
        "dataset": dataset,
        "split": {"test_size": test_size, "random_state": seed},
        "preprocessing": preprocessing,
        "model": {"name": name, "implementation": implementation, "params": params},
        "metrics": metrics,
        "evaluation": {"strategy": strategy, "repetitions": repetitions, "n_splits": n_splits},
    }


async def execute(experiment_id: str, config: dict):
    records[experiment_id].update(status="running", progress=15, stage="loading dataset")
    await publish(experiment_id)
    try:
        records[experiment_id].update(progress=45, stage="training model")
        await publish(experiment_id)
        loop = asyncio.get_running_loop()
        def report_progress(done, total):
            def update():
                if records[experiment_id]["status"] != "running":
                    return
                records[experiment_id].update(progress=15 + int(80 * done / total), stage=f"已完成 {done} / {total} 次真实训练与评价")
                asyncio.create_task(publish(experiment_id))
            loop.call_soon_threadsafe(update)
        result = await asyncio.to_thread(run_experiment, config, report_progress)
        records[experiment_id].update(status="completed", progress=100, stage="completed", result=result, finished_at=now())
    except Exception as error:
        records[experiment_id].update(status="failed", progress=100, stage="failed", error=str(error), finished_at=now())
    # A storage problem must not turn a successful training run into a failed run.
    try:
        await asyncio.to_thread(save_record, records[experiment_id])
    except Exception:
        logging.exception("Unable to persist experiment %s", experiment_id)
        records[experiment_id]["storage_warning"] = "本地后端记录保存失败；请下载 JSON 备份。"
    await publish(experiment_id)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "mlbench-studio"}


@app.get("/api/datasets")
async def datasets():
    entries = []
    for dataset in ("iris", "wine", "breast_cancer", "digits", "moons", "circles", "diabetes", "friedman1"):
        bundle = load_dataset(dataset)
        entries.append({"id": dataset, "task_type": bundle["task_type"], "samples": len(bundle["X"]), "features": len(bundle["feature_names"]), "target_names": bundle["target_names"]})
    return entries


@app.get("/api/algorithms")
async def algorithms():
    entries = []
    for name in registered_names("scratch"):
        scratch = get_model_class(name, "scratch")
        entries.append({"name": name, "task_type": scratch.task_type, "implementations": [item for item in ("scratch", "sklearn") if name in registered_names(item)]})
    return entries


@app.post("/api/experiments", status_code=202)
async def create_experiment(payload: dict):
    try:
        config = normalize_config(payload)
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    experiment_id = uuid4().hex[:12]
    records[experiment_id] = {"id": experiment_id, "status": "queued", "progress": 0, "stage": "queued", "config": config, "result": None, "error": None, "created_at": now()}
    try:
        await asyncio.to_thread(save_record, records[experiment_id])
    except Exception:
        logging.exception("Unable to persist queued experiment")
    asyncio.create_task(execute(experiment_id, config))
    return {"id": experiment_id, "status": "queued"}


@app.get("/api/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    if experiment_id not in records:
        raise HTTPException(status_code=404, detail="unknown experiment")
    return records[experiment_id]


@app.websocket("/api/ws/experiments/{experiment_id}")
async def experiment_socket(websocket: WebSocket, experiment_id: str):
    if experiment_id not in records:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    connections.setdefault(experiment_id, set()).add(websocket)
    await websocket.send_json({"type": "experiment", "experiment": records[experiment_id]})
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        connections.get(experiment_id, set()).discard(websocket)
        if not connections.get(experiment_id):
            connections.pop(experiment_id, None)
