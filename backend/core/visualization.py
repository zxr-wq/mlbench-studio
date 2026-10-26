"""Read-only diagnostics for fitted models; no estimator is fitted here."""

import numpy as np


def projection_basis(training):
    training = np.asarray(training, dtype=float)
    center = training.mean(axis=0)
    _, _, vectors = np.linalg.svd(training - center, full_matrices=False)
    return center, vectors[:min(2, training.shape[1])]


def project(values, center, basis):
    result = (np.asarray(values, dtype=float) - center) @ basis.T
    if result.shape[1] == 1:
        result = np.column_stack([result[:, 0], np.zeros(len(result))])
    return result


def _points(values, labels):
    return [{"x": round(float(x), 6), "y": round(float(y), 6), "label": int(label)}
            for (x, y), label in zip(values, labels)]


def classification_diagnostics(model, bundle, X_train, X_test, y_train, y_test):
    """Build bounded payloads from the actual fitted estimator and native feature distances."""
    payload = {}
    estimator = getattr(model, "estimator_", None)
    classes = np.unique(y_train)
    if estimator is not None and hasattr(estimator, "tree_"):
        tree = estimator.tree_
        def export_tree(index):
            counts = np.asarray(tree.value[index]).reshape(-1)
            if counts.sum() <= 1.000001:
                counts = counts * tree.n_node_samples[index]
            leaf = tree.children_left[index] == tree.children_right[index]
            return {"feature": None if leaf else str(bundle["feature_names"][tree.feature[index]]),
                    "threshold": None if leaf else float(tree.threshold[index]),
                    "samples": int(tree.n_node_samples[index]), "value": np.rint(counts).astype(int).tolist(),
                    "left": None if leaf else export_tree(tree.children_left[index]),
                    "right": None if leaf else export_tree(tree.children_right[index])}
        payload["tree_structure"] = export_tree(0)
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X_test[:40]), dtype=float)
        payload["probabilities"] = {
            "classes": classes.astype(int).tolist(),
            "rows": [{"sample": index, "actual": int(y_test[index]),
                      "values": np.round(row, 6).tolist()}
                     for index, row in enumerate(probabilities)],
        }

    if hasattr(model, "k") and hasattr(model, "distance"):
        center, basis = projection_basis(X_train)
        train_projection = project(X_train, center, basis)
        test_projection = project(X_test[:40], center, basis)
        examples = []
        for index, point in enumerate(X_test[:40]):
            difference = X_train - point
            distances = (np.abs(difference).sum(axis=1) if model.distance == "manhattan"
                         else np.sqrt((difference ** 2).sum(axis=1)))
            neighbors = np.argsort(distances, kind="stable")[:min(15, len(X_train))]
            examples.append({
                "sample": index, "actual": int(y_test[index]),
                "x": float(test_projection[index, 0]), "y": float(test_projection[index, 1]),
                "neighbors": [{"index": int(i), "label": int(y_train[i]),
                               "distance": float(distances[i])} for i in neighbors],
            })
        payload["knn"] = {"k": model.k, "distance": model.distance,
                          "weights": model.weights, "training": _points(train_projection, y_train),
                          "examples": examples, "projection": "PCA fitted on training data; distances use all preprocessed features"}

    if hasattr(model, "kernel"):
        if X_train.shape[1] != 2:
            payload["svm_note"] = "真实决策边界仅在原始特征数为 2 时绘制；可使用 Two Moons / Circles。高维数据的二维投影不是模型决策边界。"
        else:
            lo, hi = X_train.min(axis=0), X_train.max(axis=0)
            margin = np.maximum(hi - lo, .1) * .12
            xs, ys = np.linspace(lo[0] - margin[0], hi[0] + margin[0], 55), np.linspace(lo[1] - margin[1], hi[1] + margin[1], 55)
            xx, yy = np.meshgrid(xs, ys)
            grid = np.column_stack([xx.ravel(), yy.ravel()])
            reference = getattr(model, "_model", None)
            decision = reference.decision_function(grid) if reference is not None else model.decision_function(grid)
            decision = np.asarray(decision)
            # Scratch is one-vs-rest, with scores shaped (classes, samples).
            # Its prediction boundary is score(class 1) - score(class 0) = 0.
            if reference is None:
                decision = decision[1] - decision[0] if len(classes) == 2 else decision.max(axis=0)
            elif decision.ndim > 1:
                decision = decision.max(axis=1)
            supports = reference.support_vectors_ if reference is not None else np.unique(np.vstack(model.support_vectors_), axis=0)
            payload["svm_boundary"] = {
                "xs": xs.tolist(), "ys": ys.tolist(),
                "labels": model.predict(grid).astype(int).reshape(len(ys), len(xs)).tolist(),
                "scores": decision.reshape(len(ys), len(xs)).tolist(),
                "supports": [{"x": float(x), "y": float(y)} for x, y in supports],
                "training": _points(X_train, y_train), "test": _points(X_test, y_test),
                "feature_names": list(bundle["feature_names"]), "kernel": model.kernel,
            }

    means = getattr(model, "means_", getattr(estimator, "theta_", None))
    variances = getattr(model, "variances_", getattr(estimator, "var_", None))
    if means is not None and variances is not None:
        features = []
        for feature in range(min(4, X_train.shape[1])):
            mu, variance = np.asarray(means)[:, feature], np.asarray(variances)[:, feature]
            sd = np.sqrt(np.maximum(variance, 1e-12))
            xs = np.linspace(float(np.min(mu - 3 * sd)), float(np.max(mu + 3 * sd)), 65)
            curves = [np.exp(-.5 * ((xs - m) / s) ** 2) / (s * np.sqrt(2 * np.pi)) for m, s in zip(mu, sd)]
            features.append({"name": str(bundle["feature_names"][feature]), "x": xs.tolist(),
                             "curves": [curve.tolist() for curve in curves]})
        payload["gaussian_features"] = features

    if hasattr(model, "trees_") and model.__class__.__name__ == "RandomForest":
        votes = np.zeros((len(X_test), len(classes)), dtype=int)
        rows = []
        for count, tree in enumerate(model.trees_, start=1):
            prediction = tree.predict(X_test)
            votes += prediction[:, None] == classes[None, :]
            rows.append({"iteration": count, "accuracy": float(np.mean(classes[votes.argmax(axis=1)] == y_test))})
        payload["ensemble_curve"] = rows
    elif hasattr(model, "trees_") and model.__class__.__name__ == "GradientBoosting":
        scores = np.zeros((len(X_test), len(classes)))
        rows = []
        for count, stage in enumerate(model.trees_, start=1):
            for column, tree in enumerate(stage):
                scores[:, column] += model.learning_rate * tree.predict(X_test)
            rows.append({"iteration": count, "accuracy": float(np.mean(classes[scores.argmax(axis=1)] == y_test))})
        payload["ensemble_curve"] = rows
    elif estimator is not None and hasattr(estimator, "staged_predict"):
        payload["ensemble_curve"] = [{"iteration": index, "accuracy": float(np.mean(prediction == y_test))}
                                     for index, prediction in enumerate(estimator.staged_predict(X_test), start=1)]
    elif estimator is not None and hasattr(estimator, "estimators_") and model.__class__.__name__ == "SklearnRandomForest":
        probabilities = np.zeros((len(X_test), len(classes)))
        rows = []
        for count, tree in enumerate(estimator.estimators_, start=1):
            probabilities += tree.predict_proba(X_test)
            rows.append({"iteration": count, "accuracy": float(np.mean(classes[probabilities.argmax(axis=1)] == y_test))})
        payload["ensemble_curve"] = rows
    return payload
