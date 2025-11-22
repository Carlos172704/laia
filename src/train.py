import os
from math import sqrt

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import prepare_dataframe


RAW_PATH = "data/training_data/training/yellow_tripdata_2013-01.parquet"
EXPT_NAME = "baseline_trip_duration"


import os
import pandas as pd
from src.features import prepare_dataframe   # garante que tens este import no topo

RAW_PATH = "data/training_data/training/yellow_tripdata_2013-01.parquet"


def load_data():
    """
    Carrega os dados reais se o ficheiro existir.
    Se não existir (ex.: GitHub Actions), usa um dataset sintético pequenino
    só para o CI conseguir treinar um modelo.
    """
    if os.path.exists(RAW_PATH):
        df = pd.read_parquet(RAW_PATH)
        return prepare_dataframe(df)

    # Fallback para CI / ambiente sem dados reais
    print(f"[WARN] RAW_PATH '{RAW_PATH}' não encontrado — a usar dados sintéticos para treino CI.")

    base_pickup = pd.to_datetime("2013-01-01 08:00:00")
    n = 100

    df = pd.DataFrame({
        "pickup_datetime": base_pickup + pd.to_timedelta(range(n), unit="min"),
        "dropoff_datetime": base_pickup + pd.to_timedelta([i + 10 for i in range(n)], unit="min"),
        "pickup_longitude": [-73.95] * n,
        "pickup_latitude": [40.75] * n,
        "dropoff_longitude": [-73.99] * n,
        "dropoff_latitude": [40.76] * n,
        "trip_distance": [1.0 + 0.05 * i for i in range(n)],
        "passenger_count": [1] * n,
    })

    return prepare_dataframe(df)


def build_pipeline():
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
    model = Ridge(alpha=1.0, random_state=42)
    return Pipeline([("pre", pre), ("model", model)])


def main():
    # Ficheiro local por defeito (funciona no CI sem servidor MLflow)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPT_NAME)

    from sklearn import set_config
    set_config(transform_output="default")

    with mlflow.start_run():
        df = load_data()
        X = df.drop(columns=["duration_min"])
        y = df["duration_min"]

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

        # Exportar modelo para servir na API
        os.makedirs("artifacts", exist_ok=True)
        import joblib

        joblib.dump(pipe, "artifacts/model.pkl")

        print(f"MAE={mae:.2f} | RMSE={rmse:.2f}")


if __name__ == "__main__":
    os.makedirs("mlruns", exist_ok=True)
    main()
