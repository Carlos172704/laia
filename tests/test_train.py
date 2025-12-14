import pandas as pd

from src.training.train import build_pipeline


def test_build_pipeline_trains_on_dummy_data():
    pipe = build_pipeline()

    X = pd.DataFrame(
        {
            "trip_distance": [1.2, 3.4],
            "hav_km": [1.0, 2.0],
            "passenger_count": [1, 2],
            "pickup_hour": [8, 17],
            "pickup_dow": [0, 4],
            "pickup_mon": [1, 12],
        }
    )
    y = pd.Series([10.0, 20.0])

    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert len(preds) == len(y)
