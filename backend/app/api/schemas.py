from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ExperimentRequest(BaseModel):
    dataset_id: str = "iris"
    model_ids: list[str] = Field(default_factory=lambda: ["logistic_regression"])
    split_method: Literal["stratified", "random", "kfold"] = "stratified"
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    folds: int = Field(default=5, ge=3, le=10)
    standardize: bool = True
    seed: int = Field(default=42, ge=0, le=1_000_000)
    parameters: dict[str, dict[str, Any]] = Field(default_factory=dict)
    implementations: list[Literal["scratch", "sklearn"]] = Field(default_factory=lambda: ["scratch"])
    metrics: list[Literal["accuracy", "precision", "recall", "f1"]] = Field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1"]
    )

    @field_validator("model_ids")
    @classmethod
    def require_models(cls, value: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(value))
        if not cleaned:
            raise ValueError("至少选择一个算法")
        if len(cleaned) > 5:
            raise ValueError("一次最多比较五个算法")
        return cleaned

    @field_validator("implementations")
    @classmethod
    def require_implementations(cls, value: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(value))
        if not cleaned:
            raise ValueError("至少选择一种实现")
        return cleaned

    @field_validator("metrics")
    @classmethod
    def require_metrics(cls, value: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(value))
        if not cleaned:
            raise ValueError("至少选择一个评价指标")
        return cleaned


class ExperimentCreated(BaseModel):
    id: str
    status: str


class ExperimentSummary(BaseModel):
    id: str
    status: str
    created_at: str
    dataset_id: str
    model_ids: list[str]
    progress: int
    best_model: str | None = None
    best_accuracy: float | None = None
