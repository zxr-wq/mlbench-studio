# B Algorithms and Integration

This integration keeps one machine-learning core. B's contribution extends that core rather than introducing another model service.

## Linear Regression

`backend/models/scratch/linear_regression.py` implements ordinary least squares with a Moore-Penrose pseudo-inverse. It uses the regression contract: `fit(X, y)`, `predict(X)`, `get_params()`, and `get_visualization_data()`. The reference adapter is `backend/models/sklearn/linear_models.py`.

The Diabetes dataset is the regression benchmark. Metrics are MSE, RMSE, MAE, and R2. The runner rejects attempts to evaluate a regression model on a classification dataset, preventing invalid cross-task comparisons.

## Logistic Regression

`backend/models/scratch/logistic_regression.py` implements multiclass softmax regression with batch gradient descent and L2 regularization. It records cross-entropy loss history for the frontend. The sklearn adapter uses the same public parameters (`max_iter`, `l2`) for a reference result.

## Integration and Server Review

Both models register through `@register_model`, are created through `ModelFactory`, and execute through `run_experiment`. The FastAPI server validates model existence, dataset existence, `test_size`, and returns a clear 400 response for invalid configurations. WebSocket updates expose queued, running, completed, and failed records without duplicating model logic in the transport layer.

## Verification

`backend/tests/test_b_models.py` checks least-squares parity, Logistic Regression accuracy, and the Diabetes Runner result shape. `backend/tests/test_server.py` checks invalid-config handling and the Iris KNN HTTP plus WebSocket path.
