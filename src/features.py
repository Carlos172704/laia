import numpy as np
import pandas as pd


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
