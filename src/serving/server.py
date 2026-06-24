"""Model Serving Module.

FastAPI-based model serving with health checks,
prediction endpoints, and model versioning.
"""

import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any
import numpy as np


app = FastAPI(title="ML Model Serving API", version="1.0.0")

model = None
model_version = None


class PredictionRequest(BaseModel):
    """Request schema for predictions."""
    features: list[float] = Field(..., description="Input features for prediction")
    model_version: str | None = Field(None, description="Specific model version to use")


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    prediction: int | list[int]
    probability: float | list[float]
    model_version: str
    api_version: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: str | None


def load_model(version: str | None = None) -> Any:
    """Load model from MLflow model registry.

    Args:
        version: Specific model version to load.

    Returns:
        Loaded model.
    """
    global model, model_version

    if version:
        model_uri = f"models:/ProductionModel/{version}"
        model_version = version
    else:
        model_uri = "models:/ProductionModel/latest"
        model_version = "latest"

    model = mlflow.sklearn.load_model(model_uri)
    return model


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health status.

    Returns:
        Health status with model information.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_version=model_version
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Make predictions using the loaded model.

    Args:
        request: Prediction request with features.

    Returns:
        Prediction results with probabilities.
    """
    global model, model_version

    if request.model_version and request.model_version != model_version:
        load_model(request.model_version)

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = np.array(request.features).reshape(1, -1)

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0].max())

    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        model_version=model_version or "unknown",
        api_version="1.0.0"
    )


@app.post("/batch_predict")
async def batch_predict(features: list[list[float]]) -> dict[str, Any]:
    """Make batch predictions.

    Args:
        features: List of feature vectors.

    Returns:
        Batch prediction results.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features_array = np.array(features)

    predictions = model.predict(features_array).tolist()
    probabilities = model.predict_proba(features_array).max(axis=1).tolist()

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "count": len(predictions),
        "model_version": model_version
    }


@app.get("/model_info")
async def model_info() -> dict[str, Any]:
    """Get information about the loaded model.

    Returns:
        Model metadata.
    """
    return {
        "model_type": "RandomForestClassifier",
        "model_version": model_version,
        "api_version": "1.0.0",
        "input_features": "numeric_array"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
