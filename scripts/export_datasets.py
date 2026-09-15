"""Export scikit-learn toy datasets to JSON files consumed by the browser ML core.

Usage:
    python scripts/export_datasets.py

Generates:
    public/data/iris.json
    public/data/wine.json
    public/data/breast_cancer.json
    public/data/digits.json
"""

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits

OUT_DIR = Path(__file__).resolve().parent.parent / "public" / "data"


def export(loader, dataset_id: str, name: str, rounding: int = 4) -> None:
    bundle = loader()
    payload = {
        "id": dataset_id,
        "name": name,
        "featureNames": [str(item) for item in bundle.feature_names],
        "targetNames": [str(item) for item in bundle.target_names],
        "data": np.round(bundle.data, rounding).tolist(),
        "target": bundle.target.tolist(),
    }
    path = OUT_DIR / f"{dataset_id}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    samples, features = bundle.data.shape
    print(f"wrote {path.relative_to(OUT_DIR.parent)} ({samples} samples x {features} features)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export(load_iris, "iris", "Iris")
    export(load_wine, "wine", "Wine")
    export(load_breast_cancer, "breast_cancer", "Breast Cancer Wisconsin")
    export(load_digits, "digits", "Optical Digits", rounding=1)


if __name__ == "__main__":
    main()
