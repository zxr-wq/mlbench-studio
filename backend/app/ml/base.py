from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Classifier(ABC):
    """Minimal estimator protocol used by the experiment engine."""

    task_type = "classification"
    history_: list[dict[str, float]]

    def __init__(self) -> None:
        self.history_ = []

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> "Classifier":
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        return {
            name: value
            for name, value in vars(self).items()
            if not name.endswith("_") and not name.startswith("_")
        }

    def get_visualization_data(self) -> dict[str, Any]:
        """Return JSON-serializable model-specific visualization data."""
        return {}
