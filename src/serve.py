import os
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import mlflow.pyfunc

# Global container for the in-memory loaded model
model_registry = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that runs on server startup and shutdown.
    Loads the MLflow model into memory before taking requests.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    # Default to version 4 or fallback to models:/HousingElasticNet/latest
    model_uri = os.getenv("MODEL_URI", "models:/HousingElasticNet/latest")
    
    print(f"Connecting to MLflow Tracking at: {tracking_uri}")
    print(f"Attempting to load model from: {model_uri}")

    mlflow.set_tracking_uri(tracking_uri)
    try:
        model_registry["model"] = mlflow.pyfunc.load_model(model_uri)
        print("Model successfully loaded into memory.")
    except Exception as exc:
        print(f"Error loading model from registry: {exc}")
        # Fallback helper for local dev if registry fails to resolve alias
        model_registry["model"] = None

    yield
    # Cleanup logic on server shutdown
    model_registry.clear()
    print("Inference service shut down cleanly.")


app = FastAPI(
    title="California Housing Price Inference API",
    description="Production-grade inference endpoint backed by MLflow model registry.",
    version="1.0.0",
    lifespan=lifespan,
)


# Define input data contract matching dataset features
class HousingRecord(BaseModel):
    MedInc: float = Field(..., description="Median income in block group", examples=[3.87])
    HouseAge: float = Field(..., description="Median house age in block group", examples=[28.0])
    AveRooms: float = Field(..., description="Average number of rooms per household", examples=[5.4])
    AveBedrms: float = Field(..., description="Average number of bedrooms per household", examples=[1.0])
    Population: float = Field(..., description="Block group population", examples=[1048.0])
    AveOccup: float = Field(..., description="Average number of household members", examples=[3.1])
    Latitude: float = Field(..., description="Block group latitude", examples=[37.88])
    Longitude: float = Field(..., description="Block group longitude", examples=[-122.23])


class PredictionResponse(BaseModel):
    predictions: List[float]


@app.get("/health", tags=["Monitoring"])
def health():
    """Liveness probe used by orchestrators (Docker/Kubernetes)."""
    is_loaded = model_registry.get("model") is not None
    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(payload: List[HousingRecord]):
    """
    Accepts a list of housing records, formats into a DataFrame,
    and returns array of continuous value predictions.
    """
    model = model_registry.get("model")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Check server startup logs.",
        )

    # Convert incoming Pydantic payload directly to pandas DataFrame
    df_features = pd.DataFrame([record.model_dump() for record in payload])

    try:
        preds = model.predict(df_features)
        return {"predictions": preds.tolist()}
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Inference execution failed: {err}")