from functools import lru_cache
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

class PredictRequest(BaseModel):
    day_of_year: int = Field(..., ge=1, le=366)
    month: int = Field(..., ge=1, le=12)
    humidity: float = Field(..., ge=0, le=100)
    wind_speed: float
    pressure: float
    cloud_cover: float = Field(..., ge=0, le=100)
    previous_temp: float

class PredictResponse(BaseModel):
    predicted_temperature: float

app = FastAPI(title="Weather Prediction API", version="0.1")

@lru_cache(maxsize=1)
def load_model():
    # Loads the model once and caches it
    return joblib.load("weather_model.pkl")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    model = load_model()
    X = np.array([[
        payload.day_of_year,
        payload.month,
        payload.humidity,
        payload.wind_speed,
        payload.pressure,
        payload.cloud_cover,
        payload.previous_temp
    ]])
    pred = model.predict(X)
    return PredictResponse(predicted_temperature=float(pred[0]))
