from datetime import datetime
import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.features import haversine  # já existe no teu ficheiro features


app = FastAPI(title="NYC Taxi Trip Duration API")

_MODEL = None


def get_model():
    """Lazy load do modelo."""
    global _MODEL
    if _MODEL is None:
        model_path = os.getenv("MODEL_PATH", "artifacts/model.pkl")
        _MODEL = joblib.load(model_path)
    return _MODEL


class TripInput(BaseModel):
    pickup_datetime: datetime
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    passenger_count: int
    trip_distance: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(trip: TripInput):
    dt = trip.pickup_datetime

    hav_km = haversine(
        trip.pickup_longitude,
        trip.pickup_latitude,
        trip.dropoff_longitude,
        trip.dropoff_latitude,
    )

    features = pd.DataFrame(
        {
            "trip_distance": [trip.trip_distance],
            "hav_km": [hav_km],
            "passenger_count": [trip.passenger_count],
            "pickup_hour": [dt.hour],
            "pickup_dow": [dt.weekday()],
            "pickup_mon": [dt.month],
        }
    )

    model = get_model()
    pred = float(model.predict(features)[0])

    return {
        "predicted_duration_min": pred,
        "model_version": os.getenv("MODEL_VERSION", "local"),
    }
