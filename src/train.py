import os
from math import sqrt
import glob

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import prepare_dataframe

# === CONFIG ===
# Pattern for training parquet files (2011–2012)
TRAIN_GLOB = os.getenv(
    "TRAIN_GLOB",
    "data/training_data/yellow_tripdata_201[1-2]-*.parquet",
)

FALLBACK_TRAIN_GLOB = os.getenv(
    "FALLBACK_TRAIN_GLOB",
    "data/training_data/yellow_tripdata_2011-*.parquet",
)

# Pattern for explicit test parquet files (2013)
TEST_GLOB = os.getenv(
    "TEST_GLOB",
    "data/testing_data/yellow_tripdata_2013-*.parquet",
)

# Max rows to load for training / testing (to avoid killing the VM)
MAX_TRAIN_ROWS = int(os.getenv("MAX_TRAIN_ROWS", "20000000")) #acresecentar 0 para aumentar treino
MAX_TEST_ROWS = int(os.getenv("MAX_TEST_ROWS", "10000000")) #acresecentar 0 para aumentar teste

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "baseline_trip_duration")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:mlruns")


def load_raw_data(path_pattern: str, max_rows: int | None = None) -> pd.DataFrame:
    """
    Load one or more parquet files matching `path_pattern`, optionally
    sampling a maximum total number of rows.
    This avoids reading the full dataset into memory on a small VM.
    """
    files = sorted(glob.glob(path_pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files match pattern: {path_pattern}")

    dfs = []
    total = 0

    for fp in files:
        df = pd.read_parquet(fp)

        if max_rows is not None:
            remaining = max_rows - total
            if remaining <= 0:
                break

            if len(df) > remaining:
                # Random sample for this chunk
                df = df.sample(n=remaining, random_state=42)

        dfs.append(df)
        total += len(df)

        if max_rows is not None and total >= max_rows:
            break

    return pd.concat(dfs, ignore_index=True)


def load_data() -> pd.DataFrame:
    """
    Fallback para CI / ambientes sem dados reais:
    cria um dataset sintético pequenino só para conseguir treinar.
    """
    print("[WARN] A usar dados sintéticos para treino CI (sem parquet reais).")

    base_pickup = pd.to_datetime("2013-01-01 08:00:00")
    n = 100

    df = pd.DataFrame(
        {
            "pickup_datetime": base_pickup + pd.to_timedelta(range(n), unit="min"),
            "dropoff_datetime": base_pickup
            + pd.to_timedelta([i + 10 for i in range(n)], unit="min"),
            "pickup_longitude": [-73.95] * n,
            "pickup_latitude": [40.75] * n,
            "dropoff_longitude": [-73.99] * n,
            "dropoff_latitude": [40.76] * n,
            "trip_distance": [1.0 + 0.05 * i for i in range(n)],
            "passenger_count": [1] * n,
        }
    )

    return prepare_dataframe(df)


def build_pipeline() -> Pipeline:
    num_features = [
        "trip_distance",
        "hav_km",
        "passenger_count",
        "pickup_hour",
        "pickup_dow",
        "pickup_mon",
    ]

    pre = ColumnTransformer(
        [("num", StandardScaler(), num_features)],
        remainder="drop",
    )

    model = Ridge(alpha=1.0)

    pipe = Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", model),
        ]
    )
    return pipe


def main():
    # 1) MLflow config
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ===== TRAIN DATA (2011–2012, with fallback) =====
    try:
        print(f"Loading training data from: {TRAIN_GLOB}")
        print(f"MAX_TRAIN_ROWS = {MAX_TRAIN_ROWS}")
        df_raw = load_raw_data(TRAIN_GLOB, max_rows=MAX_TRAIN_ROWS)
        df = prepare_dataframe(df_raw)
        print(f"After prepare_dataframe (2011–2012): n_rows = {len(df)}")
        if df.empty:
            raise ValueError("Preprocessed training data from 2011–2012 is empty.")
    except (FileNotFoundError, ValueError) as e:
        # Fallback: try 2013 training data (same schema as you used before)
        print(f"[WARN] {e} — falling back to training data from: {FALLBACK_TRAIN_GLOB}")
        try:
            df_raw = load_raw_data(FALLBACK_TRAIN_GLOB, max_rows=MAX_TRAIN_ROWS)
            df = prepare_dataframe(df_raw)
            print(f"After prepare_dataframe (fallback 2013): n_rows = {len(df)}")
            if df.empty:
                raise ValueError("Preprocessed fallback training data (2013) is empty.")
        except (FileNotFoundError, ValueError) as e2:
            # CI / GitHub Actions case: no real parquet files, use synthetic data
            print(f"[WARN] {e2} — falling back to synthetic CI dataset.")
            df = load_data()  # already prepared
            print(f"Synthetic dataset rows: {len(df)}")

    # Features / target
    X = df.drop(columns=["duration_min"])
    y = df["duration_min"]

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        pipe = build_pipeline()
        pipe.fit(X_train, y_train)

        # Validation metrics (train split)
        y_pred = pipe.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = sqrt(mean_squared_error(y_val, y_pred))

        mlflow.log_param("model", "Ridge")
        mlflow.log_param("alpha", pipe.named_steps["model"].alpha)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)

        print(f"VAL_MAE={mae:.2f} | VAL_RMSE={rmse:.2f}")

        # ===== EXPLICIT TEST ON 2013 (TEST_GLOB) =====
        try:
            print(f"Loading test data from: {TEST_GLOB}")
            print(f"MAX_TEST_ROWS = {MAX_TEST_ROWS}")
            df_test_raw = load_raw_data(TEST_GLOB, max_rows=MAX_TEST_ROWS)
            df_test = prepare_dataframe(df_test_raw)
            print(f"After prepare_dataframe (test 2013): n_rows = {len(df_test)}")

            if not df_test.empty:
                X_test = df_test.drop(columns=["duration_min"])
                y_test = df_test["duration_min"]

                y_test_pred = pipe.predict(X_test)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))

                mlflow.log_metric("TEST_MAE", test_mae)
                mlflow.log_metric("TEST_RMSE", test_rmse)

                print(f"TEST_MAE={test_mae:.2f} | TEST_RMSE={test_rmse:.2f}")
            else:
                print("[WARN] Test dataframe is empty after preprocessing; skipping TEST metrics.")
        except FileNotFoundError as e:
            print(f"[WARN] {e} — no explicit 2013 test data found, skipping TEST evaluation.")

        # 4) Log model and export artifact for API
        mlflow.sklearn.log_model(pipe, "model")

        os.makedirs("artifacts", exist_ok=True)
        import joblib

        joblib.dump(pipe, "artifacts/model.pkl")
        print("Saved model to artifacts/model.pkl")


if __name__ == "__main__":
    os.makedirs("mlruns", exist_ok=True)
    main()
