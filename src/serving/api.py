from datetime import datetime
import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

# from src.features import haversine  # not needed for leaderboard input right now

app = FastAPI(title="NYC Taxi Trip Duration API (Leaderboard-Compatible)")

_MODEL = None

def get_model():
    """Lazy load do modelo."""
    global _MODEL
    if _MODEL is None:
        model_path = os.getenv("MODEL_PATH", "artifacts/model.pkl")
        _MODEL = joblib.load(model_path)
    return _MODEL


# === SCHEMAS THAT MATCH THE LEADERBOARD EXAMPLE ENDPOINT ===
# Example expected input (from example_endpoint.py):
# {
#   "data": [
#     {
#       "VendorID": 2,
#       "tpep_pickup_datetime": "2011-01-01 00:10:00",
#       "passenger_count": 4,
#       "trip_distance": 1,
#       "PULocationID": 145,
#       "DOLocationID": 145
#     },
#     ...
#   ]
# }

class LeaderboardRecord(BaseModel):
    VendorID: int
    tpep_pickup_datetime: datetime
    passenger_count: int
    trip_distance: float
    PULocationID: int
    DOLocationID: int


class LeaderboardRequest(BaseModel):
    data: List[LeaderboardRecord]


@app.get("/health")
def health():
    # Match the example endpoint's response as closely as possible
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: LeaderboardRequest):
    """
    Leaderboard-compatible prediction endpoint.

    Input:  { "data": [ { VendorID, tpep_pickup_datetime, passenger_count,
                          trip_distance, PULocationID, DOLocationID }, ... ] }
    Output: { "predictions": [ ... ] }
    """

    records = []
    for rec in payload.data:
        dt = rec.tpep_pickup_datetime

        hav_km_approx = rec.trip_distance * 1.60934

        records.append(
            {
                "trip_distance": rec.trip_distance,
                "hav_km": hav_km_approx,
                "passenger_count": rec.passenger_count,
                "pickup_hour": dt.hour,
                "pickup_dow": dt.weekday(),  # same convention as pandas .dt.dayofweek
                "pickup_mon": dt.month,
            }
        )

    features = pd.DataFrame.from_records(records)

    model = get_model()
    preds = model.predict(features)

    # Leaderboard expects {"predictions": [ ... ]}
    return {"predictions": [float(p) for p in preds]}


if __name__ == "__main__":

    port = int(os.getenv("PORT", 9001))
    uvicorn.run(app, host="0.0.0.0", port=port)