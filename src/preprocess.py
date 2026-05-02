from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessConfig:
    city: str = "paris"

    raw_data_path: str = "data/raw/weather_Paris_hourly.csv"
    processed_dir: str = "data/processed"

    # Fenêtre d'entrée : 24 dernières heures
    input_window: int = 24

    # Horizon de prédiction : température dans 6 heures
    forecast_horizon: int = 6

    target: str = "temperature_2m"

    base_features: Tuple[str, ...] = (
        "temperature_2m",
        "relative_humidity_2m",
        "pressure_msl",
        "cloud_cover",
        "wind_speed_10m",
        "precipitation",
    )
    train_ratio: float = 0.70
    val_ratio: float = 0.15


CFG = PreprocessConfig()


def load_raw_data(path: str) -> pd.DataFrame:
    """Charge le CSV brut et vérifie les colonnes principales."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\nLance d'abord : python -m src.download_data"
        )

    df = pd.read_csv(path)

    if "time" not in df.columns:
        raise ValueError("La colonne 'time' est absente du fichier brut.")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    missing_cols = [col for col in CFG.base_features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes météo manquantes : {missing_cols}")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    hour = df["time"].dt.hour
    day_of_year = df["time"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # On utilise 365.25 pour tenir compte approximativement des années bissextiles.
    df["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    return df


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="linear", limit_direction="both"
    )

    df = df.dropna().reset_index(drop=True)

    return df


def create_supervised_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    input_window: int,
    forecast_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = df[feature_columns].to_numpy(dtype=np.float32)
    target = df[target_column].to_numpy(dtype=np.float32)
    times = df["time"].to_numpy()

    X_list = []
    y_list = []
    target_times = []

    max_start = len(df) - input_window - forecast_horizon + 1

    for start in range(max_start):
        end = start + input_window
        target_index = end + forecast_horizon - 1

        X_list.append(features[start:end])
        y_list.append(target[target_index])
        target_times.append(times[target_index])

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)
    target_times = np.array(target_times)

    return X, y, target_times


def temporal_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> tuple:
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_train = X[:n_train]
    y_train = y[:n_train]
    times_train = times[:n_train]

    X_val = X[n_train : n_train + n_val]
    y_val = y[n_train : n_train + n_val]
    times_val = times[n_train : n_train + n_val]

    X_test = X[n_train + n_val :]
    y_test = y[n_train + n_val :]
    times_test = times[n_train + n_val :]

    return (
        X_train,
        y_train,
        times_train,
        X_val,
        y_val,
        times_val,
        X_test,
        y_test,
        times_test,
    )


def scale_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple:
    n_train, window, n_features = X_train.shape

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, n_features)
    feature_scaler.fit(X_train_2d)

    target_scaler.fit(y_train.reshape(-1, 1))

    def transform_X(X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        X_2d = X.reshape(-1, n_features)
        X_scaled = feature_scaler.transform(X_2d)
        return X_scaled.reshape(n_samples, window, n_features).astype(np.float32)

    def transform_y(y: np.ndarray) -> np.ndarray:
        return target_scaler.transform(y.reshape(-1, 1)).astype(np.float32).ravel()

    X_train_scaled = transform_X(X_train)
    X_val_scaled = transform_X(X_val)
    X_test_scaled = transform_X(X_test)

    y_train_scaled = transform_y(y_train)
    y_val_scaled = transform_y(y_val)
    y_test_scaled = transform_y(y_test)

    return (
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        X_test_scaled,
        y_test_scaled,
        feature_scaler,
        target_scaler,
    )


def save_processed_arrays(
    processed_dir: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    times_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    times_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    times_test: np.ndarray,
    feature_columns: list[str],
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
) -> None:
    os.makedirs(processed_dir, exist_ok=True)

    np.save(os.path.join(processed_dir, "X_train.npy"), X_train)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "times_train.npy"), times_train)

    np.save(os.path.join(processed_dir, "X_val.npy"), X_val)
    np.save(os.path.join(processed_dir, "y_val.npy"), y_val)
    np.save(os.path.join(processed_dir, "times_val.npy"), times_val)

    np.save(os.path.join(processed_dir, "X_test.npy"), X_test)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)
    np.save(os.path.join(processed_dir, "times_test.npy"), times_test)

    joblib.dump(
        {
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
        },
        os.path.join(processed_dir, "scalers.joblib"),
    )

    info = {
        "config": asdict(CFG),
        "feature_columns": feature_columns,
        "target": CFG.target,
        "input_window": CFG.input_window,
        "forecast_horizon": CFG.forecast_horizon,
        "shapes": {
            "X_train": list(X_train.shape),
            "y_train": list(y_train.shape),
            "X_val": list(X_val.shape),
            "y_val": list(y_val.shape),
            "X_test": list(X_test.shape),
            "y_test": list(y_test.shape),
        },
        "date_ranges": {
            "train": [str(times_train[0]), str(times_train[-1])],
            "val": [str(times_val[0]), str(times_val[-1])],
            "test": [str(times_test[0]), str(times_test[-1])],
        },
    }

    with open(
        os.path.join(processed_dir, "dataset_info.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(info, f, indent=4, ensure_ascii=False)


def main() -> None:
    print("Chargement des données brutes...")
    df = load_raw_data(CFG.raw_data_path)

    print("Ajout des variables temporelles cycliques...")
    df = add_time_features(df)

    print("Nettoyage des valeurs manquantes...")
    df = clean_missing_values(df)

    feature_columns = list(CFG.base_features) + [
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    ]

    print("Création des fenêtres supervisées...")
    X, y, times = create_supervised_windows(
        df=df,
        feature_columns=feature_columns,
        target_column=CFG.target,
        input_window=CFG.input_window,
        forecast_horizon=CFG.forecast_horizon,
    )

    print("Découpage temporel train/validation/test...")
    (
        X_train,
        y_train,
        times_train,
        X_val,
        y_val,
        times_val,
        X_test,
        y_test,
        times_test,
    ) = temporal_train_val_test_split(
        X=X,
        y=y,
        times=times,
        train_ratio=CFG.train_ratio,
        val_ratio=CFG.val_ratio,
    )

    print("Normalisation des données...")
    (
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        X_test_scaled,
        y_test_scaled,
        feature_scaler,
        target_scaler,
    ) = scale_data(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    print("Sauvegarde des données prétraitées...")
    save_processed_arrays(
        processed_dir=CFG.processed_dir,
        X_train=X_train_scaled,
        y_train=y_train_scaled,
        times_train=times_train,
        X_val=X_val_scaled,
        y_val=y_val_scaled,
        times_val=times_val,
        X_test=X_test_scaled,
        y_test=y_test_scaled,
        times_test=times_test,
        feature_columns=feature_columns,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )

    print("\nPrétraitement terminé.")
    print(f"Features utilisées : {feature_columns}")
    print(f"X_train : {X_train_scaled.shape}, y_train : {y_train_scaled.shape}")
    print(f"X_val   : {X_val_scaled.shape}, y_val   : {y_val_scaled.shape}")
    print(f"X_test  : {X_test_scaled.shape}, y_test  : {y_test_scaled.shape}")
    print(f"Fichiers sauvegardés dans : {CFG.processed_dir}")


if __name__ == "__main__":
    main()
