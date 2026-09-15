from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine


@dataclass(frozen=True)
class DatasetBundle:
    id: str
    name: str
    description: str
    x: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    name: str
    domain: str
    description: str
    accent: str
    loader: Callable[[], object]


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        id="iris",
        name="Iris 鸢尾花",
        domain="植物分类",
        description="以花萼和花瓣的 4 个形态特征识别 3 种鸢尾花。",
        accent="#7157d9",
        loader=load_iris,
    ),
    DatasetSpec(
        id="wine",
        name="Wine 葡萄酒",
        domain="化学鉴别",
        description="根据 13 项化学检测指标判断葡萄酒所属的 3 个产区。",
        accent="#c45a47",
        loader=load_wine,
    ),
    DatasetSpec(
        id="breast_cancer",
        name="Breast Cancer",
        domain="医学诊断",
        description="以细胞核统计特征辅助区分良性与恶性肿瘤。",
        accent="#1c8b76",
        loader=load_breast_cancer,
    ),
    DatasetSpec(
        id="digits",
        name="Digits 手写数字",
        domain="图像识别",
        description="由 8x8 灰度像素构成的十分类手写数字数据集。",
        accent="#ce8533",
        loader=load_digits,
    ),
)


class DatasetService:
    def __init__(self) -> None:
        self._specs = {item.id: item for item in DATASETS}

    def list(self) -> list[dict[str, object]]:
        output = []
        for spec in DATASETS:
            raw = spec.loader()
            output.append(
                {
                    "id": spec.id,
                    "name": spec.name,
                    "domain": spec.domain,
                    "description": spec.description,
                    "accent": spec.accent,
                    "samples": int(raw.data.shape[0]),
                    "features": int(raw.data.shape[1]),
                    "classes": int(len(raw.target_names)),
                    "task_type": "classification",
                }
            )
        return output

    def load(self, dataset_id: str) -> DatasetBundle:
        if dataset_id not in self._specs:
            raise KeyError(f"unknown dataset: {dataset_id}")
        spec = self._specs[dataset_id]
        raw = spec.loader()
        feature_names = getattr(raw, "feature_names", [f"feature_{i + 1}" for i in range(raw.data.shape[1])])
        return DatasetBundle(
            id=spec.id,
            name=spec.name,
            description=spec.description,
            x=np.asarray(raw.data, dtype=float),
            y=np.asarray(raw.target, dtype=int),
            feature_names=tuple(str(name) for name in feature_names),
            target_names=tuple(str(name) for name in raw.target_names),
        )

    def preview(self, dataset_id: str) -> dict[str, object]:
        bundle = self.load(dataset_id)
        class_values, counts = np.unique(bundle.y, return_counts=True)
        stats = []
        for index, name in enumerate(bundle.feature_names[:10]):
            column = bundle.x[:, index]
            stats.append(
                {
                    "name": name,
                    "mean": round(float(np.mean(column)), 4),
                    "std": round(float(np.std(column)), 4),
                    "min": round(float(np.min(column)), 4),
                    "max": round(float(np.max(column)), 4),
                }
            )
        return {
            "id": bundle.id,
            "name": bundle.name,
            "shape": [int(value) for value in bundle.x.shape],
            "features": list(bundle.feature_names),
            "classes": [
                {"id": int(value), "name": bundle.target_names[int(value)], "count": int(count)}
                for value, count in zip(class_values, counts, strict=True)
            ],
            "feature_stats": stats,
        }


dataset_service = DatasetService()
