"""HTTP and WebSocket server that delegates every experiment to core.runner."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import backend.models  # noqa: F401 - model modules register themselves
from backend.core.registry import get_model_class, registered_names
from backend.core.runner import run_experiment
from backend.datasets.loader import load_dataset


app = FastAPI(title="MLBench Studio API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

records: dict[str, dict] = {}
connections: dict[str, set[WebSocket]] = {}


def now():
    return datetime.now(timezone.utc).isoformat()


async def publish(experiment_id: str):
    record = records[experiment_id]
    stale = []
    for socket in connections.get(experiment_id, set()):
        try:
            await socket.send_json({"type": "experiment", "experiment": record})
        except Exception:
            stale.append(socket)
    for socket in stale:
        connections[experiment_id].discard(socket)


def normalize_config(payload: dict) -> dict:
    """Validate the one public Config shape before it reaches the Runner."""
    try:
        model = payload["model"]
        name = model["name"]
        implementation = model.get("implementation", "scratch")
        dataset = payload["dataset"]
    except (TypeError, KeyError) as error:
        raise ValueError("config requires dataset and model.name") from error
    get_model_class(name, implementation)
    load_dataset(dataset)
    split = payload.get("split") or {}
    return {
        "dataset": dataset,
        "split": {"test_size": float(split.get("test_size", 0.2)), "random_state": int(split.get("random_state", 42))},
        "preprocessing": list(payload.get("preprocessing") or []),
        "model": {"name": name, "implementation": implementation, "params": dict(model.get("params") or {})},
        "metrics": list(payload.get("metrics") or []),
    }


async def execute(experiment_id: str, config: dict):
    records[experiment_id].update(status="running", progress=15, stage="loading dataset")
    await publish(experiment_id)
    try:
        records[experiment_id].update(progress=45, stage="training model")
        await publish(experiment_id)
        result = await asyncio.to_thread(run_experiment, config)
        records[experiment_id].update(status="completed", progress=100, stage="completed", result=result, finished_at=now())
    except Exception as error:
        records[experiment_id].update(status="failed", progress=100, stage="failed", error=str(error), finished_at=now())
    await publish(experiment_id)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "mlbench-studio"}


@app.get("/api/datasets")
async def datasets():
    entries = []
    for dataset in ("iris", "wine", "breast_cancer", "digits"):
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
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    experiment_id = uuid4().hex[:12]
    records[experiment_id] = {"id": experiment_id, "status": "queued", "progress": 0, "stage": "queued", "config": config, "result": None, "error": None, "created_at": now()}
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
        connections[experiment_id].discard(websocket)
