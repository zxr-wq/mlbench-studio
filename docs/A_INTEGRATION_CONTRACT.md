# A 组件对接约定

## 稳定标识

- KNN：`knn`
- Gaussian Naive Bayes：`naive_bayes`
- 实现版本：`scratch` 或 `sklearn`
- 任务类型：`classification`

前端不再发送 `gaussian_nb`、`n_neighbors` 或 `distance_weighted`。KNN 参数固定为 `k`、`distance`、`weights`。

## 当前批量实验请求

现有工作台支持一次选择多个模型，因此在单模型 Config 外使用了一层批量封装：

```json
{
  "dataset_id": "iris",
  "model_ids": ["knn", "naive_bayes"],
  "implementations": ["scratch", "sklearn"],
  "split_method": "stratified",
  "test_size": 0.2,
  "folds": 5,
  "standardize": true,
  "seed": 42,
  "metrics": ["accuracy", "precision", "recall", "f1"],
  "parameters": {
    "knn": {"k": 5, "distance": "euclidean", "weights": "distance"},
    "naive_bayes": {"var_smoothing": 1e-9}
  }
}
```

这一批量封装只属于 Experiment Service；其中每一个 `model_id + implementation` 仍对应一次独立模型运行。

## 单次结果

```json
{
  "id": "knn:scratch",
  "model": "knn",
  "task_type": "classification",
  "implementation": "scratch",
  "dataset": "iris",
  "params": {"k": 5, "distance": "euclidean", "weights": "distance"},
  "metrics": {"accuracy": 0.966667, "precision": 0.969697, "recall": 0.966667, "f1": 0.966583},
  "training_time": 0.00002,
  "inference_time": 0.00046,
  "visualization": {}
}
```

同一模型的两种实现都完成后，结果增加：

```json
{
  "comparison": {
    "accuracy_difference": 0.0,
    "prediction_agreement": 1.0
  }
}
```

## 可视化数据

KNN 的 `visualization.type` 为 `knn_neighbors`，包含 `query`、`neighbors`、`feature_names`和 `class_names`。Naive Bayes 的类型为 `naive_bayes_stats`，包含 `class_priors`、`means`、`variances`、`feature_names`和 `class_names`。其他模型没有专用数据时返回空字典。

## 其他成员向前端返回的可视化 Schema

Decision Tree：

```json
{
  "type": "decision_tree",
  "tree": {
    "feature": "petal length (cm)", "threshold": 2.45, "samples": 120, "value": [40, 40, 40],
    "left": {"feature": null, "threshold": null, "samples": 40, "value": [40, 0, 0], "left": null, "right": null},
    "right": {}
  }
}
```

SVM：

```json
{
  "type": "svm_boundary",
  "points": [{"x": -1.2, "y": 0.4, "label": 0}],
  "support_vectors": [{"x": -0.3, "y": 0.1, "label": 0}],
  "boundary_path": "M 20 100 L 480 150"
}
```

K-Means：

```json
{
  "type": "kmeans_clusters",
  "centroid_history": [
    [[-1.0, 0.5], [1.2, -0.4]],
    [[-0.8, 0.6], [1.0, -0.2]]
  ]
}
```

PCA：

```json
{
  "type": "pca_projection",
  "projection": [{"x": -1.2, "y": 0.4, "label": 0, "prediction": 0, "correct": true}],
  "explained_variance_ratio": [0.73, 0.18],
  "class_names": ["class 0", "class 1"]
}
```

Gradient Boosting：

```json
{
  "type": "gradient_boosting_loss",
  "loss_history": [0.68, 0.51, 0.39, 0.31]
}
```

前端 `VisualizationPanel` 会按 `type` 自动选择图表，无需在页面中增加模型名判断。
