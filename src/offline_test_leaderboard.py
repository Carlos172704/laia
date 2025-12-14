from datetime import datetime
import joblib
import pandas as pd

def build_features_from_leaderboard_payload(payload: dict) -> pd.DataFrame:
    """
    Convert a leaderboard-style payload into the feature dataframe
    expected by your trained model.

    Payload format (like example_endpoint.py):
    {
        "data": [
            {
                "VendorID": 2,
                "tpep_pickup_datetime": "2011-01-01 00:10:00",
                "passenger_count": 4,
                "trip_distance": 1.0,
                "PULocationID": 145,
                "DOLocationID": 145
            },
            ...
        ]
    }
    """
    records = []

    for rec in payload["data"]:
        # Parse pickup datetime
        pickup_str = rec["tpep_pickup_datetime"]
        # Format is "YYYY-MM-DD HH:MM:SS"
        pickup_dt = datetime.strptime(pickup_str, "%Y-%m-%d %H:%M:%S")

        # Features used in training:
        # trip_distance, hav_km, passenger_count, pickup_hour, pickup_dow, pickup_mon
        trip_distance = float(rec["trip_distance"])
        passenger_count = int(rec["passenger_count"])

        features = {
            "trip_distance": trip_distance,
            # same approximation as your API: miles -> km
            "hav_km": trip_distance * 1.60934,
            "passenger_count": passenger_count,
            "pickup_hour": pickup_dt.hour,
            "pickup_dow": pickup_dt.weekday(),
            "pickup_mon": pickup_dt.month,
        }
        records.append(features)

    return pd.DataFrame.from_records(records)


def main():
    # 1) Example "question" in leaderboard format
    payload = {
        "data": [
            {
                "VendorID": 2,
                "tpep_pickup_datetime": "2013-01-01 00:10:00",
                "passenger_count": 1,
                "trip_distance": 3.5,
                "PULocationID": 145,
                "DOLocationID": 236,
            },
            {
                "VendorID": 1,
                "tpep_pickup_datetime": "2013-01-02 08:30:00",
                "passenger_count": 2,
                "trip_distance": 1.2,
                "PULocationID": 132,
                "DOLocationID": 170,
            },
        ]
    }

    # 2) Build feature DataFrame
    X = build_features_from_leaderboard_payload(payload)
    print("Feature dataframe:")
    print(X)

    # 3) Load your trained model
    model_path = "artifacts/model.pkl"  # or use MODEL_PATH env var if you prefer
    model = joblib.load(model_path)

    # 4) Predict
    preds = model.predict(X)

    # 5) Print "answers" like the leaderboard would see
    print("\nPredicted trip durations (in minutes):")
    for rec, p in zip(payload["data"], preds):
        print(
            f"Pickup {rec['tpep_pickup_datetime']}, "
            f"distance {rec['trip_distance']} miles -> {p:.2f} min"
        )


if __name__ == "__main__":
    main()
