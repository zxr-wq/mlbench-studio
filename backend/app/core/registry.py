from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.ml.base import Classifier


@dataclass(frozen=True)
class AlgorithmSpec:
    id: str
    name: str
    short_name: str
    family: str
    description: str
    principle: str
    formula: str
    strengths: tuple[str, ...]
    defaults: dict[str, Any]
    parameter_schema: tuple[dict[str, Any], ...]
    factory: Callable[..., Classifier]
    sklearn_factory: Callable[..., Classifier] | None = None
    task_type: str = "classification"

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "short_name": self.short_name,
            "family": self.family,
            "description": self.description,
            "principle": self.principle,
            "formula": self.formula,
            "strengths": list(self.strengths),
            "defaults": self.defaults,
            "parameter_schema": list(self.parameter_schema),
            "implementations": ["scratch", "sklearn"] if self.sklearn_factory is not None else ["scratch"],
            "task_type": self.task_type,
        }


class AlgorithmRegistry:
    def __init__(self) -> None:
        self._items: dict[str, AlgorithmSpec] = {}

    def register(self, spec: AlgorithmSpec) -> None:
        if spec.id in self._items:
            raise ValueError(f"duplicate algorithm id: {spec.id}")
        self._items[spec.id] = spec

    def get(self, algorithm_id: str) -> AlgorithmSpec:
        try:
            return self._items[algorithm_id]
        except KeyError as exc:
            raise KeyError(f"unknown algorithm: {algorithm_id}") from exc

    def create(
        self,
        algorithm_id: str,
        parameters: dict[str, Any] | None = None,
        implementation: str = "scratch",
    ) -> Classifier:
        spec = self.get(algorithm_id)
        config = {**spec.defaults, **(parameters or {})}
        allowed = {item["key"] for item in spec.parameter_schema}
        config = {key: value for key, value in config.items() if key in allowed}
        if implementation == "scratch":
            factory = spec.factory
        elif implementation == "sklearn" and spec.sklearn_factory is not None:
            factory = spec.sklearn_factory
        else:
            raise ValueError(f"implementation '{implementation}' is not available for {algorithm_id}")
        return factory(**config)

    def list(self) -> list[AlgorithmSpec]:
        return list(self._items.values())


algorithm_registry = AlgorithmRegistry()
