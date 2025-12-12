import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv(override=False)

@dataclass
class Configs:
    """
    Class for environment configuration for training.
    """
    # Runtime configs
    COMMIT_SHA: str
    MODEL_NAME: str
    EXPERIMENT_NAME: str
    MLFLOW_TRACKING_URI: str

    # Data
    TRAIN_PATH: str
    TEST_PATH: str

    # Training configs
    MAX_TRAIN_ROWS: int
    MAX_TEST_ROWS: int
    SEED: int
    TEST_SIZE: float

def load_configs() -> Configs:
    """
    Read environment variables and loads them into a 'Configs' class instance.

    :return: Configs class
    """
    commit_sha = os.getenv("COMMIT_SHA")
    if not commit_sha:
        raise EnvironmentError("Missing required env var: COMMIT_SHA")
    
    model_name = os.getenv("MLFLOW_MODEL_NAME")
    if not model_name:
        raise EnvironmentError("Missing required env var: MLFLOW_MODEL_NAME")
    
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    if not experiment_name:
        raise EnvironmentError("Missing required env var: MLFLOW_EXPERIMENT_NAME")
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", 
                             "http://localhost:5050"
                             )

    data_path = os.getenv("TAXI_DATA_PATH")
    if not data_path:
        raise EnvironmentError(
            "Missing required env var: TAXI_DATA_PATH (path to the CSV file)"
        )
    return Configs(
        COMMIT_SHA=commit_sha,
        MODEL_NAME=model_name,
        EXPERIMENT_NAME=experiment_name,
        MLFLOW_TRACKING_URI=tracking_uri,
        data_path=data_path,
    )

configs = load_configs()