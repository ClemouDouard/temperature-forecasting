from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models import persistent_baseline


@dataclass
class EvalConfig:
    processed_dir: str = "data/processed"
    models_dir: str = "models"
    reports_dir: str = "reports"

    temperature_feature_index: int = 0


CFG = EvalConfig()


def ensure_reports_dir() -> None:
    os.makedirs(CFG.reports_dir, exist_ok=True)


def load_test_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Charge X_test, y_test, times_test et les scalers."""
    required_files = [
        "X_test.npy",
        "y_test.npy",
        "times_test.npy",
        "scalers.joblib",
    ]

    missing = [
        f
        for f in required_files
        if not os.path.exists(os.path.join(CFG.processed_dir, f))
    ]

    if missing:
        raise FileNotFoundError(
            f"Fichiers manquants dans {CFG.processed_dir} : {missing}\n"
            "Lance d'abord : python -m src.preprocess"
        )

    X_test = np.load(os.path.join(CFG.processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(CFG.processed_dir, "y_test.npy"))
    times_test = np.load(
        os.path.join(CFG.processed_dir, "times_test.npy"), allow_pickle=True
    )
    scalers = joblib.load(os.path.join(CFG.processed_dir, "scalers.joblib"))

    return X_test, y_test, times_test, scalers


def inverse_target_scaling(y_scaled: np.ndarray, target_scaler) -> np.ndarray:
    """Repasse une cible normalisée en degrés Celsius."""
    y_scaled = np.asarray(y_scaled).reshape(-1, 1)
    y = target_scaler.inverse_transform(y_scaled).ravel()
    return y


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """Aplati les fenêtres temporelles pour Ridge."""
    return X.reshape(X.shape[0], -1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcule les métriques principales en degrés Celsius."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE_degC": float(mae),
        "RMSE_degC": float(rmse),
        "R2": float(r2),
    }


def load_keras_model(model_name: str) -> tf.keras.Model:
    path = os.path.join(CFG.models_dir, f"{model_name}.keras")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle Keras introuvable : {path}")
    return tf.keras.models.load_model(path)


def load_ridge_model():
    path = os.path.join(CFG.models_dir, "ridge.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle Ridge introuvable : {path}")
    return joblib.load(path)


def main() -> None:
    ensure_reports_dir()

    print("Chargement des données de test...")
    X_test, y_test_scaled, times_test, scalers = load_test_data()
    target_scaler = scalers["target_scaler"]

    y_test = inverse_target_scaling(y_test_scaled, target_scaler)

    print(f"X_test : {X_test.shape}")
    print(f"y_test : {y_test.shape}")

    predictions = {
        "time": pd.to_datetime(times_test),
        "y_true": y_test,
    }

    metrics = {}

    print("Évaluation : baseline persistante")
    y_pred_baseline_scaled = persistent_baseline(
        X_test,
        temperature_feature_index=CFG.temperature_feature_index,
    )
    y_pred_baseline = inverse_target_scaling(y_pred_baseline_scaled, target_scaler)
    predictions["baseline"] = y_pred_baseline
    metrics["Baseline persistante"] = compute_metrics(y_test, y_pred_baseline)

    print("Évaluation : Ridge")
    ridge = load_ridge_model()
    y_pred_ridge_scaled = ridge.predict(flatten_windows(X_test))
    y_pred_ridge = inverse_target_scaling(y_pred_ridge_scaled, target_scaler)
    predictions["ridge"] = y_pred_ridge
    metrics["Ridge"] = compute_metrics(y_test, y_pred_ridge)

    keras_models = {
        "MLP": "mlp",
        "CNN 1D": "cnn1d",
        "GRU": "gru",
    }

    for display_name, file_name in keras_models.items():
        print(f"Évaluation : {display_name}")
        model = load_keras_model(file_name)
        y_pred_scaled = model.predict(X_test, verbose=0).ravel()
        y_pred = inverse_target_scaling(y_pred_scaled, target_scaler)

        predictions[file_name] = y_pred
        metrics[display_name] = compute_metrics(y_test, y_pred)

    metrics_df = pd.DataFrame(metrics).T
    metrics_df = metrics_df.sort_values("MAE_degC")
    metrics_path = os.path.join(CFG.reports_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path)

    predictions_df = pd.DataFrame(predictions)
    predictions_path = os.path.join(CFG.reports_dir, "predictions_test.csv")
    predictions_df.to_csv(predictions_path, index=False)

    print("\nRésultats sur l'ensemble de test :")
    print(metrics_df.round(4))

    print(f"\nMétriques sauvegardées dans : {metrics_path}")
    print(f"Prédictions sauvegardées dans : {predictions_path}")


if __name__ == "__main__":
    main()
