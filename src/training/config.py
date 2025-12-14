import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv(override=False)

@dataclass
class Configs:
    """
    Datalass for configuration.
    """
    # Runtime configs
    MODEL_NAME: str
    EXPERIMENT_NAME: str
    MLFLOW_TRACKING_URI: str

    # Data
    BASE_URL: str
    TRAIN_DATA: list
    VALIDATION_DATA: list

    # Training configs
    MAX_TRAIN_ROWS: int
    MAX_TEST_ROWS: int
    SEED: int
    TEST_SIZE: float

    MODEL_PATH: str
    COMMIT_SHA: str

def load_configs() -> Configs:
    """
    Read environment variables and loads them into a 'Configs' class instance.

    :return: Configs class
    """

    # Runtime
    model_name = os.getenv("MLFLOW_MODEL_NAME")
    if not model_name:
        raise EnvironmentError("Missing required env var: MLFLOW_MODEL_NAME")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    if not experiment_name:
        raise EnvironmentError("Missing required env var: MLFLOW_EXPERIMENT_NAME")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", 
                             "http://localhost:5000"
                             )

    # Data
    base_url = os.getenv("BASE_URL")
    if not base_url:
        raise EnvironmentError("Missing required env var: BASE_URL")
    train_data = os.getenv("TRAIN_DATA").split(",")
    if not train_data:
        raise EnvironmentError("Missing required env var: TRAIN_DATA")
    validation_data = os.getenv("VALIDATION_DATA").split(",")
    if not validation_data:
        raise EnvironmentError("Missing required env var: VALIDATION_DATA")
    
    # Training
    max_train_rows=int(os.getenv("MAX_TRAIN_ROWS", 20000000))
    max_test_rows=int(os.getenv("MAX_TEST_ROWS", 10000000))
    seed=int(os.getenv("SEED", 42))
    test_size=float(os.getenv("TEST_SIZE", 0.2))
    
    model_path = os.getenv("MODEL_PATH", "artifacts/model.pkl")
    commit_sha = os.getenv("COMMIT_SHA", "local-dev")
    
    return Configs(
        MODEL_NAME=model_name,
        EXPERIMENT_NAME=experiment_name,
        MLFLOW_TRACKING_URI=tracking_uri,
        BASE_URL=base_url,
        TRAIN_DATA=train_data,
        VALIDATION_DATA=validation_data,
        MAX_TRAIN_ROWS=max_train_rows,
        MAX_TEST_ROWS=max_test_rows,
        SEED=seed,
        TEST_SIZE=test_size,
        MODEL_PATH=model_path,
        COMMIT_SHA=commit_sha
    )

configs = load_configs()
