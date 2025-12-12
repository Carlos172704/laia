import joblib
import pandas as pd
from training.utils import prepare_dataframe

MODEL_PATH = "artifacts/model.pkl"
TEST_FILE = "data/testing_data/yellow_tripdata_2013-01.parquet"

def main():
    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print(f"Loading test data from {TEST_FILE} ...")
    df_raw = pd.read_parquet(TEST_FILE)

    # Prepare features/target exactly like in training
    df = prepare_dataframe(df_raw)
    print(f"Rows after prepare_dataframe: {len(df)}")

    X = df.drop(columns=["duration_min"])
    y_true = df["duration_min"]

    # Take a small sample to inspect
    sample = df.sample(n=5, random_state=42)
    X_sample = sample.drop(columns=["duration_min"])
    y_sample = sample["duration_min"]

    y_pred = model.predict(X_sample)

    # Show side-by-side
    out = sample.copy()
    out["pred_duration_min"] = y_pred

    cols_to_show = [
        "trip_distance",
        "pickup_hour",
        "pickup_dow",
        "pickup_mon",
        "duration_min",
        "pred_duration_min",
    ]
    print("\nSample of real trips (true vs predicted):")
    print(out[cols_to_show])

if __name__ == "__main__":
    main()