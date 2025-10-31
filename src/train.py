import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from features import prepare_dataframe

RAW_PATH = "data/raw/yellow_tripdata_2011-01.parquet"
EXPT_NAME = "baseline_trip_duration"

def load_data():
    df = pd.read_parquet(RAW_PATH)
    return prepare_dataframe(df)

def build_pipeline():
    num_features = ["trip_distance","hav_km","passenger_count","pickup_hour","pickup_dow","pickup_mon"]
    pre = ColumnTransformer([("num", StandardScaler(), num_features)], remainder="drop")
    model = Ridge(alpha=1.0, random_state=42)
    return Pipeline([("pre", pre), ("model", model)])

def main():
    mlflow.set_tracking_uri("http://localhost:5050")
    mlflow.set_experiment("baseline_trip_duration")
    with mlflow.start_run():
        df = load_data()
        X = df.drop(columns=["duration_min"])
        y = df["duration_min"]

        # simple split: first 80% train, last 20% val (time-agnostic for CP1 demo)
        split = int(0.8 * len(df))
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        pipe = build_pipeline()
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = sqrt(mean_squared_error(y_val, preds))

        mlflow.log_param("model", "Ridge")
        mlflow.log_param("alpha", 1.0)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)

        mlflow.sklearn.log_model(pipe, "model")

        print(f"MAE={mae:.2f} | RMSE={rmse:.2f}")

if __name__ == "__main__":
    os.makedirs("mlruns", exist_ok=True)
    main()
