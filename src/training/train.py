import os
import joblib
import numpy as np

import mlflow

from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from .utils import *
from .config import configs

def train_model():
    # Setup MLFlow to use local server
    mlflow_uri = configs.MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(mlflow_uri)

    # Set experiment
    mlflow.set_experiment(configs.EXPERIMENT_NAME)

    # Load data
    df_train, df_val = load_data()
    
    df_train = build_dataframe(df_train)
    df_val = build_dataframe(df_val)

    X = df_train.drop(columns=["duration_min"])
    y = df_train["duration_min"]

    X_val = df_val.drop(columns=["duration_min"])
    y_val = df_val["duration_min"]

    # Run
    with mlflow.start_run() as run:
        print_and_log("Training started", "info")
        pipe = build_pipeline()

        cv = KFold(n_splits=5, shuffle=True, random_state=configs.SEED)
        cv_results = cross_validate(pipe, X, y, cv=cv, 
                                    scoring={
                                        "mae": "neg_mean_absolute_error", 
                                        "rmse": "neg_root_mean_squared_error"},
                                    return_train_score=False
                                    )
        
        cv_mae = -cv_results["test_mae"]
        cv_rmse = -cv_results["test_rmse"]

        mlflow.log_param("model", "Ridge")
        mlflow.log_metric("CV_MAE_MEAN", float(np.mean(cv_mae)))
        mlflow.log_metric("CV_MAE_STD", float(np.std(cv_mae)))
        mlflow.log_metric("CV_RMSE_MEAN", float(np.mean(cv_rmse)))
        mlflow.log_metric("CV_RMSE_STD", float(np.std(cv_rmse)))
        
        pipe.fit(X, y)

        y_pred = pipe.predict(X_val)
        val_mae = mean_absolute_error(y_val, y_pred)
        val_rmse = sqrt(mean_squared_error(y_val, y_pred))

        mlflow.log_metric("VAL_MAE", val_mae)
        mlflow.log_metric("VAL_RMSE", val_rmse)

        mlflow.log_param("alpha", pipe.named_steps["model"].alpha)
        mlflow.sklearn.log_model(pipe, "model")

        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(pipe, configs.MODEL_PATH)
        print_and_log(f"Model saved on {configs.MODEL_PATH}.", "info")

    model_uri = f"runs:/{run.info.run_id}/model"

    try:
        registered_model = mlflow.register_model(model_uri, configs.MODEL_NAME)
        print(f"Registered model version: {registered_model.version}")
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        client.set_registered_model_alias(
            name=configs.MODEL_NAME,
            alias="staging",
            version=registered_model.version,
        )

        client.set_registered_model_alias(
            name=configs.MODEL_NAME,
            alias=configs.COMMIT_SHA,
            version=registered_model.version,
        )

        print(f"Model version {registered_model.version} promoted to Production")
    except Exception as e:
        print(f"ERROR: Failed to register/promote model: {e}")
        print(f"Model URI: {model_uri}")
        print(f"MLflow tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}")
        raise 


if __name__ == "__main__":
    train_model()
