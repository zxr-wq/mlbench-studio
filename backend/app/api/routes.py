from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.api.schemas import ExperimentCreated, ExperimentRequest
from app.core.registry import algorithm_registry
from app.services.datasets import dataset_service
from app.services.experiments import experiment_service


router = APIRouter(prefix="/api")


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "ml-studio"}


@router.get("/datasets")
async def list_datasets() -> list[dict[str, object]]:
    return await asyncio.to_thread(dataset_service.list)


@router.get("/datasets/{dataset_id}")
async def dataset_preview(dataset_id: str) -> dict[str, object]:
    try:
        return await asyncio.to_thread(dataset_service.preview, dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/algorithms")
async def list_algorithms() -> list[dict[str, object]]:
    return [spec.public_dict() for spec in algorithm_registry.list()]


@router.post("/experiments", status_code=202, response_model=ExperimentCreated)
async def create_experiment(request: ExperimentRequest) -> ExperimentCreated:
    try:
        record = experiment_service.create(request)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    asyncio.create_task(experiment_service.run(record["id"], request))
    return ExperimentCreated(id=record["id"], status=record["status"])


@router.get("/experiments")
async def list_experiments() -> list[dict[str, object]]:
    return experiment_service.list()


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str) -> dict[str, object]:
    try:
        return experiment_service.get(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.websocket("/ws/experiments/{experiment_id}")
async def experiment_socket(websocket: WebSocket, experiment_id: str) -> None:
    try:
        record = experiment_service.get(experiment_id)
    except KeyError:
        await websocket.close(code=4404)
        return
    await experiment_service.manager.connect(experiment_id, websocket)
    await websocket.send_json({"type": "experiment", "experiment": record})
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        experiment_service.manager.disconnect(experiment_id, websocket)
