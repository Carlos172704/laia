import pandas as pd
import logging
import glob
import os
from typing import Tuple

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import configs

logging.basicConfig(filename="logs.log",
                    level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    )
logger = logging.getLogger(__name__)

"""def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train_list = []
    df_validation_list = []

    # Training data
    for year in configs.TRAIN_DATA:
        for month in range(1,5):
            url = f"{configs.BASE_URL}/yellow_tripdata_{year}-{month:02d}.parquet"
            print_and_log(f"Loading {url}...", "info")
            try:
                df = pd.read_parquet(url)
                df_train_list.append(df)
            except Exception as e:
                print_and_log(f"Could not load {url}: {e}", "error")
    
    for year in configs.VALIDATION_DATA:
        for month in range(1,5):
            url = f"{configs.BASE_URL}/yellow_tripdata_{year}-{month:02d}.parquet"
            print_and_log(f"Loading {url}...", "info")
            try:
                df = pd.read_parquet(url)
                df_validation_list.append(df)
            except Exception as e:
                print_and_log(f"Could not load {url}: {e}", "error")
    
    return pd.concat(df_train_list, ignore_index=True), pd.concat(df_validation_list, ignore_index=True)"""
    
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and validation data from LOCAL parquet files instead of URLs.

      - training/*.parquet   -> 2011–2012 data (train)
      - testing/*.parquet    -> 2013 data (validation/test)

    If you use different folder names, set them in config.py as:
      configs.TRAIN_DIR and configs.VALIDATION_DIR
    """
    train_dir = getattr(configs, "TRAIN_DIR", "training")
    val_dir = getattr(configs, "VALIDATION_DIR", "testing")

    train_pattern = os.path.join(train_dir, "yellow_tripdata_*.parquet")
    val_pattern = os.path.join(val_dir, "yellow_tripdata_*.parquet")

    train_files = sorted(glob.glob(train_pattern))
    val_files = sorted(glob.glob(val_pattern))

    if not train_files:
        raise FileNotFoundError(
            f"No training parquet files found matching pattern: {train_pattern}"
        )
    if not val_files:
        raise FileNotFoundError(
            f"No validation parquet files found matching pattern: {val_pattern}"
        )

    df_train_list = []
    df_val_list = []

    # Load training files
    for fp in train_files:
        print_and_log(f"Loading local training file {fp}...", "info")
        df = pd.read_parquet(fp)
        df_train_list.append(df)

    # Load validation files
    for fp in val_files:
        print_and_log(f"Loading local validation file {fp}...", "info")
        df = pd.read_parquet(fp)
        df_val_list.append(df)

    df_train = pd.concat(df_train_list, ignore_index=True)
    df_val = pd.concat(df_val_list, ignore_index=True)

    print_and_log(
        f"Loaded {len(df_train)} training rows from {len(train_files)} files.", "info"
    )
    print_and_log(
        f"Loaded {len(df_val)} validation rows from {len(val_files)} files.", "info"
    )

    return df_train, df_val

def build_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "tpep_pickup_datetime": "pickup_datetime",
            "tpep_dropoff_datetime": "dropoff_datetime",
        }
    )

    if "pickup_datetime" not in df.columns or "dropoff_datetime" not in df.columns:
        print_and_log("Missing pickup_datetime or dropoff_datetime columns in DataFrame", "error")
        raise Exception(
            "Missing pickup_datetime or dropoff_datetime"
        )

    if "trip_distance" not in df.columns:
        print_and_log("Missing trip_distance column in DataFrame", "error")
        raise Exception(
            "Dataframe não tem trip_distance; não é possível construir hav_km."
        )

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce")

    df["duration_min"] = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds() / 60.0
    df = df.dropna(subset=["pickup_datetime", "dropoff_datetime", "duration_min", "trip_distance"])
    df = df[(df["duration_min"] > 0) & (df["duration_min"] <= 180)]

    if df.empty:
        return df

    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_dow"]  = df["pickup_datetime"].dt.dayofweek
    df["pickup_mon"]  = df["pickup_datetime"].dt.month

    df["hav_km"] = df["trip_distance"] * 1.60934

    if "passenger_count" not in df.columns:
        df["passenger_count"] = 1

    cols = [
        "trip_distance",
        "hav_km",
        "passenger_count",
        "pickup_hour",
        "pickup_dow",
        "pickup_mon",
        "duration_min",
    ]
    existing_cols = [c for c in cols if c in df.columns]

    return df[existing_cols].dropna()

def print_and_log(content: str, level):
    valid_levels = ["info", "warning", "error"]
    match level:
        case "info":
            print(f"{str(level).upper()}: {content}")
            logger.info(content)
        case "warning":
            print(f"{str(level).upper()}: {content}")
            logger.warning(content)
        case "error":
            print(f"{str(level).upper()}: {content}")
            logger.error(content) 
        case _:
            raise Exception("Invalid log level")
        
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
