import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


def eval_metrics(actual: np.ndarray, pred: np.ndarray):
    """Computes standard regression evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def run_training(alpha: float, l1_ratio: float, tracking_uri: str = "sqlite:///mlflow.db"):
    """
    Executes an end-to-end training cycle and records it in MLflow.
    """
    # 1. Direct MLflow to store metadata in a local SQLite database
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("california-housing-starter")

    # 2. Ingest and split dataset
    print("Fetching California Housing dataset...")
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Context manager opens an isolated Run
    with mlflow.start_run() as run:
        print(f"Active Run ID: {run.info.run_id}")

        # Train model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(X_train, y_train)

        # Generate predictions and evaluate
        predictions = model.predict(X_test)
        rmse, mae, r2 = eval_metrics(y_test, predictions)

        # 4. Log parameters and metrics to backend store
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # 5. Infer and enforce input-output schema signature
        signature = infer_signature(X_train, predictions)

        # 6. Package model as an MLflow standard artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",  # updated from artifact_path="model"
            registered_model_name="HousingElasticNet",
            signature=signature,
            input_example=X_train.iloc[:1]
        )
        # mlflow.sklearn.log_model(
        #     sk_model=model,
        #     artifact_path="model",
        #     registered_model_name="HousingElasticNet",
        #     signature=signature,
        #     input_example=X_train.iloc[:1]
        # )

        print(f"Completed run with alpha={alpha}, l1_ratio={l1_ratio}")
        print(f"Evaluation Metrics -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        return run.info.run_id, r2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an ElasticNet model with MLflow.")
    parser.add_argument("--alpha", type=float, default=0.5, help="ElasticNet alpha (regularization strength)")
    parser.add_argument("--l1_ratio", type=float, default=0.5, help="ElasticNet l1_ratio (mixing parameter)")
    args = parser.parse_args()

    run_training(alpha=args.alpha, l1_ratio=args.l1_ratio)