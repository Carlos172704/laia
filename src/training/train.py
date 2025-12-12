import os
from math import sqrt

import mlflow
import mlflow.sklearn

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from utils import *
from config import configs


def main():
    # 1) MLflow config
    mlflow.set_tracking_uri(configs.MLFLOW_TRACKING_URI) # Local server setup
    mlflow.set_experiment(configs.EXPERIMENT_NAME) # Experiment setup

    # ===== TRAIN DATA (2011–2012, with fallback) =====
    try:
        print(f"Loading training data from: {configs.TRAIN_GLOB}")
        print(f"MAX_TRAIN_ROWS = {configs.MAX_TRAIN_ROWS}")
        df_raw = load_raw_data(configs.TRAIN_GLOB, max_rows=configs.MAX_TRAIN_ROWS)
        df = prepare_dataframe(df_raw)
        print(f"After prepare_dataframe (2011–2012): n_rows = {len(df)}")
        if df.empty:
            raise ValueError("Preprocessed training data from 2011–2012 is empty.")
    except (FileNotFoundError, ValueError) as e:
        # Fallback: try 2013 training data (same schema as you used before)
        print(f"[WARN] {e} — falling back to training data from: {configs.FALLBACK_TRAIN_GLOB}")
        try:
            df_raw = load_raw_data(configs.FALLBACK_TRAIN_GLOB, max_rows=configs.MAX_TRAIN_ROWS)
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

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=configs.SEED
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
            print(f"Loading test data from: {configs.TEST_GLOB}")
            print(f"MAX_TEST_ROWS = {configs.MAX_TEST_ROWS}")
            df_test_raw = load_raw_data(configs.TEST_GLOB, max_rows=configs.MAX_TEST_ROWS)
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
