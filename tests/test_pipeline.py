import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.train import eval_metrics, run_training
from src.serve import app, model_registry
import mlflow.pyfunc

# Create an in-memory HTTP test client for the FastAPI app
client = TestClient(app)


# =====================================================================
# 1. Unit Logic Tests
# =====================================================================

def test_eval_metrics_computation():
    """Verify standard metric calculations with deterministic arrays."""
    actual = np.array([3.0, -0.5, 2.0, 7.0])
    pred = np.array([2.5, 0.0, 2.0, 8.0])

    rmse, mae, r2 = eval_metrics(actual, pred)

    # Hand-calculated reference:
    # errors: [0.5, -0.5, 0.0, -1.0] -> abs: [0.5, 0.5, 0, 1] -> mean: 0.5
    # squared: [0.25, 0.25, 0, 1] -> mean: 0.375 -> sqrt: ~0.61237
    assert np.isclose(mae, 0.5)
    assert np.isclose(rmse, np.sqrt(0.375))
    assert r2 <= 1.0


# =====================================================================
# 2. Model Quality Gate Test (Automated Training & Evaluation)
# =====================================================================

def test_model_training_quality_gate(tmp_path):
    """
    Train a candidate model into a temporary isolated SQLite database
    and enforce a strict performance threshold.
    """
    # tmp_path is a built-in pytest fixture providing a clean temporary directory
    isolated_db = f"sqlite:///{tmp_path}/test_gate.db"

    run_id, r2 = run_training(alpha=0.1, l1_ratio=0.1, tracking_uri=isolated_db)

    # Verification 1: Run ID was generated and logged
    assert run_id is not None
    assert isinstance(run_id, str)

    # Verification 2: Performance Gate (Minimum R^2 threshold)
    MINIMUM_R2_THRESHOLD = 0.40
    assert r2 >= MINIMUM_R2_THRESHOLD, (
        f"Candidate model failed the Quality Gate! "
        f"Expected R2 >= {MINIMUM_R2_THRESHOLD}, but got {r2:.4f}"
    )


# =====================================================================
# 3. API Integration Tests
# =====================================================================

def test_api_health_endpoint():
    """Ensure the health check endpoint returns 200."""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert "status" in json_data
    assert "model_loaded" in json_data


def test_api_predict_validation_error():
    """Test that submitting invalid feature data returns HTTP 422 Unprocessable Entity."""
    # Send missing required fields
    bad_payload = [
        {
            "MedInc": 3.87,
            # Missing HouseAge, AveRooms, etc.
            "Latitude": 37.88
        }
    ]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # Pydantic schema validation error


def test_api_predict_with_mock_model():
    """Verify the /predict payload conversion logic by mocking the model call."""
    class DummyModel:
        def predict(self, df):
            return np.array([2.5] * len(df))

    # Temporarily inject a dummy model into the registry
    original_model = model_registry.get("model")
    model_registry["model"] = DummyModel()

    valid_payload = [
        {
            "MedInc": 3.87,
            "HouseAge": 28.0,
            "AveRooms": 5.4,
            "AveBedrms": 1.0,
            "Population": 1048.0,
            "AveOccup": 3.1,
            "Latitude": 37.88,
            "Longitude": -122.23
        }
    ]

    try:
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200
        assert response.json() == {"predictions": [2.5]}
    finally:
        # Restore original state
        model_registry["model"] = original_model