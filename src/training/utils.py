import pandas as pd
import glob
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara o dataframe TLC para treino:

    - Normaliza nomes de colunas de datetime.
    - Calcula duration_min (minutos) e remove outliers extremos.
    - Cria features temporais (hora, dia da semana, mês).
    - Cria hav_km a partir de trip_distance (milhas -> km),
      tal como é feito na API durante o /predict.
    - Garante que temos trip_distance e passenger_count.
    """

    # --- 1) Normalizar nomes de datetime ---
    # Alguns ficheiros usam tpep_*, outros pickup_datetime diretamente
    df = df.rename(
        columns={
            "tpep_pickup_datetime": "pickup_datetime",
            "tpep_dropoff_datetime": "dropoff_datetime",
        }
    )

    if "pickup_datetime" not in df.columns or "dropoff_datetime" not in df.columns:
        raise ValueError(
            "Faltam colunas pickup_datetime/dropoff_datetime no dataframe."
        )

    # --- 2) Alvo: duração em minutos (0 < dur <= 180) ---
    pickup_ts = pd.to_datetime(df["pickup_datetime"])
    dropoff_ts = pd.to_datetime(df["dropoff_datetime"])

    df["duration_min"] = (dropoff_ts - pickup_ts).dt.total_seconds() / 60.0
    df = df[(df["duration_min"] > 0) & (df["duration_min"] <= 180)]

    if df.empty:
        return df

    # --- 3) Features temporais ---
    df["pickup_hour"] = pickup_ts.dt.hour
    df["pickup_dow"] = pickup_ts.dt.dayofweek
    df["pickup_mon"] = pickup_ts.dt.month

    # --- 4) Distância em km (hav_km) ---
    # Não temos coordenadas, por isso usamos trip_distance (milhas) -> km,
    # alinhado com a API (api.py)
    if "trip_distance" not in df.columns:
        raise ValueError(
            "Dataframe não tem trip_distance; não é possível construir hav_km."
        )

    df["hav_km"] = df["trip_distance"] * 1.60934  # 1 milha ≈ 1.60934 km

    # --- 5) Garantir passenger_count ---
    if "passenger_count" not in df.columns:
        df["passenger_count"] = 1

    # --- 6) Selecionar colunas finais ---
    cols = [
        "trip_distance",
        "hav_km",
        "passenger_count",
        "pickup_hour",
        "pickup_dow",
        "pickup_mon",
        "duration_min",
    ]

    # Algumas colunas podem não existir em certos ficheiros muito antigos,
    # mas com os ficheiros que mostraste devem existir todas.
    existing_cols = [c for c in cols if c in df.columns]

    return df[existing_cols].dropna()
