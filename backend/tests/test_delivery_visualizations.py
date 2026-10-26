"""Diagnostics must describe actual models without changing fitted predictions."""

import json
import numpy as np
import pytest

import backend.models  # noqa: F401
from backend.core.factory import ModelFactory
from backend.core.runner import _project_2d, prepare_data, run_experiment
from backend.core.visualization import classification_diagnostics, projection_basis, project
from backend.server.app import normalize_config
from backend.server.history import load_records, save_record


def config(name, dataset="iris", implementation="scratch", params=None):
    return {"dataset": dataset, "split": {"test_size": .2, "random_state": 42},
            "preprocessing": ["standard_scaler"],
            "model": {"name": name, "implementation": implementation, "params": params or {}},
            "metrics": []}


def test_test_projection_is_fitted_exclusively_on_training_data():
    training = np.array([[0., 0.], [5., .1], [10., -.1], [15., .2]])
    test = np.array([[2., 10.], [4., 30.]])
    center, basis = projection_basis(training)
    expected = project(test, center, basis)
    np.testing.assert_allclose(_project_2d(test, training), expected)
    # Adding a radically different test observation cannot change an existing point.
    changed = _project_2d(np.vstack([test, [1000., -5000.]]), training)
    np.testing.assert_allclose(changed[:2], expected)


def test_repeated_run_visualization_names_its_actual_seed_and_fold():
    value = config("knn", params={"k": 5})
    value["evaluation"] = {"strategy": "kfold", "n_splits": 3, "repetitions": 3}
    result = run_experiment(value)
    source = result["visualization"]["provenance"]
    assert source["seed"] == 42 and source["fold"] == 1 and source["total_runs"] == 9
    assert source["metrics"] == result["evaluation"]["per_run"][0]["metrics"]


@pytest.mark.parametrize("implementation", ["scratch", "sklearn"])
def test_svm_boundary_reuses_real_scores_and_predictions(implementation):
    value = config("svm", "moons", implementation, {"kernel": "linear", "max_iter": 1000})
    bundle, train, test, y_train, y_test = prepare_data(value, "classification")
    model = ModelFactory.create("svm", value["model"]["params"], implementation).fit(train, y_train)
    before = model.predict(test)
    charts = classification_diagnostics(model, bundle, train, test, y_train, y_test)
    boundary = charts["svm_boundary"]
    grid = np.array([[x, y] for y in boundary["ys"] for x in boundary["xs"]])
    np.testing.assert_array_equal(model.predict(grid), np.asarray(boundary["labels"]).ravel())
    if implementation == "scratch":
        scores = model.decision_function(grid)
        np.testing.assert_allclose(np.asarray(boundary["scores"]).ravel(), scores[1] - scores[0])
    else:
        np.testing.assert_allclose(np.asarray(boundary["scores"]).ravel(), model._model.decision_function(grid))
    assert boundary["supports"]
    np.testing.assert_array_equal(before, model.predict(test))
    json.dumps(charts, allow_nan=False)


def test_knn_neighbors_are_nearest_in_full_feature_space():
    value = config("knn", "digits", params={"k": 5})
    bundle, train, test, y_train, y_test = prepare_data(value, "classification")
    model = ModelFactory.create("knn", {"k": 5}).fit(train, y_train)
    chart = classification_diagnostics(model, bundle, train, test, y_train, y_test)["knn"]
    example = chart["examples"][0]
    distance = np.linalg.norm(train - test[0], axis=1)
    assert [row["index"] for row in example["neighbors"]] == np.argsort(distance, kind="stable")[:15].tolist()
    np.testing.assert_allclose([row["distance"] for row in example["neighbors"]], np.sort(distance)[:15])


def test_cluster_playback_uses_native_distance_assignments():
    result = run_experiment(config("kmeans", params={"k": 3, "random_state": 42, "n_init": 2}))
    chart = result["visualization"]["clustering"]
    assert len(chart["iterations"]) == len(chart["centroid_history"])
    assert len(chart["iterations"][0]["labels"]) == 120
    np.testing.assert_array_equal(chart["iterations"][-1]["labels"], [p["cluster"] for p in chart["points"]])
    assert chart["iterations"][-1]["inertia"] == pytest.approx(result["metrics"]["inertia"])


@pytest.mark.parametrize("model", ["random_forest", "gradient_boosting"])
def test_final_ensemble_checkpoint_matches_reported_accuracy(model):
    result = run_experiment(config(model, params={"n_estimators": 5, "max_depth": 2}))
    curve = result["visualization"]["ensemble_curve"]
    assert len(curve) == 5
    assert curve[-1]["accuracy"] == pytest.approx(result["metrics"]["accuracy"])


def test_softmax_feature_contributions_reduce_the_class_axis():
    value = config("logistic_regression", "wine")
    bundle, train, _, labels, _ = prepare_data(value, "classification")
    model = ModelFactory.create("logistic_regression").fit(train, labels)
    from backend.core.runner import _feature_contributions
    contribution = _feature_contributions(model, bundle["feature_names"])
    expected = np.abs(model.weights_).mean(axis=1)
    for item in contribution["items"]:
        index = bundle["feature_names"].index(item["name"])
        assert item["value"] == pytest.approx(expected[index], abs=1e-6)
    assert len(contribution["items"]) == 10


def test_sklearn_tree_has_real_structure_and_rf_feature_defaults_match():
    result = run_experiment(config("decision_tree", implementation="sklearn", params={"max_depth": 3}))
    assert result["visualization"]["tree_structure"]["samples"] == 120
    forest = ModelFactory.create("random_forest", {}, "sklearn")
    assert forest._estimator_params()["max_features"] == "sqrt"


def test_pca_repeated_variance_threshold_is_json_safe():
    value = config("pca", "wine", params={"n_components": .95})
    value["evaluation"] = {"strategy": "holdout", "repetitions": 3}
    result = run_experiment(value)
    assert result["metrics"]["cumulative_explained_variance"][-1] >= .95
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("change", [
    {"preprocessing": "standard_scaler"}, {"metrics": ["r2"]},
    {"split": {"random_state": -1}}, {"model": {"name": "knn", "params": {"unknown": 1}}},
    {"model": {"name": "knn", "params": {"k": 10000}}},
    {"model": {"name": "random_forest", "params": {"n_estimators": 0}}},
    {"model": {"name": "logistic_regression", "params": {"learning_rate": float("nan")}}},
])
def test_invalid_configs_are_rejected_before_training(change):
    with pytest.raises(ValueError):
        normalize_config(config("knn") | change)


def test_backend_history_survives_restart_and_marks_interrupted_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("MLBENCH_HISTORY_DB", str(tmp_path / "history.sqlite3"))
    save_record({"id": "done", "status": "completed", "result": {"metrics": {"accuracy": .9}}})
    save_record({"id": "interrupted", "status": "running"})
    records = load_records()
    assert records["done"]["result"]["metrics"]["accuracy"] == .9
    assert records["interrupted"]["status"] == "failed"
