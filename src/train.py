from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict

import joblib
import numpy as np
import tensorflow as tf

from src.models import (
    build_cnn1d,
    build_gru,
    build_mlp,
    build_ridge_model,
    compile_keras_model,
)


@dataclass
class TrainConfig:
    processed_dir: str = "data/processed"
    models_dir: str = "models"
    reports_dir: str = "reports"

    seed: int = 42

    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    patience: int = 6

    ridge_alpha: float = 1.0


CFG = TrainConfig()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dirs() -> None:
    os.makedirs(CFG.models_dir, exist_ok=True)
    os.makedirs(CFG.reports_dir, exist_ok=True)
    os.makedirs(os.path.join(CFG.reports_dir, "histories"), exist_ok=True)


def load_processed_data(
    processed_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required_files = [
        "X_train.npy",
        "y_train.npy",
        "X_val.npy",
        "y_val.npy",
    ]

    missing = [
        f for f in required_files if not os.path.exists(os.path.join(processed_dir, f))
    ]
    if missing:
        raise FileNotFoundError(
            f"Fichiers prétraités manquants : {missing}\n"
            "Lance d'abord : python -m src.preprocess"
        )

    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))

    return X_train, y_train, X_val, y_val


def flatten_windows(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def save_history(history: tf.keras.callbacks.History, model_name: str) -> None:
    history_path = os.path.join(
        CFG.reports_dir, "histories", f"history_{model_name}.json"
    )
    serializable_history = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(serializable_history, f, indent=4)


def get_callbacks(model_name: str) -> list[tf.keras.callbacks.Callback]:
    checkpoint_path = os.path.join(CFG.models_dir, f"{model_name}.keras")

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=CFG.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]


def train_keras_model(
    model: tf.keras.Model,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tf.keras.callbacks.History:
    print("\n" + "=" * 70)
    print(f"Entraînement du modèle : {model_name}")
    print("=" * 70)

    model = compile_keras_model(model, learning_rate=CFG.learning_rate)
    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=CFG.epochs,
        batch_size=CFG.batch_size,
        callbacks=get_callbacks(model_name),
        verbose=1,
    )

    model.save(os.path.join(CFG.models_dir, f"{model_name}.keras"))
    save_history(history, model_name)

    return history


def main() -> None:
    set_seed(CFG.seed)
    ensure_dirs()

    print("Chargement des données prétraitées...")
    X_train, y_train, X_val, y_val = load_processed_data(CFG.processed_dir)

    input_shape = X_train.shape[1:]
    print(f"X_train : {X_train.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"X_val   : {X_val.shape}")
    print(f"y_val   : {y_val.shape}")
    print(f"input_shape Keras : {input_shape}")

    print("\n" + "=" * 70)
    print("Entraînement du modèle : Ridge")
    print("=" * 70)

    ridge = build_ridge_model(alpha=CFG.ridge_alpha)
    ridge.fit(flatten_windows(X_train), y_train)
    joblib.dump(ridge, os.path.join(CFG.models_dir, "ridge.joblib"))
    print("Modèle Ridge sauvegardé dans models/ridge.joblib")

    mlp = build_mlp(input_shape=input_shape)
    train_keras_model(
        model=mlp,
        model_name="mlp",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    cnn1d = build_cnn1d(input_shape=input_shape)
    train_keras_model(
        model=cnn1d,
        model_name="cnn1d",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    gru = build_gru(input_shape=input_shape)
    train_keras_model(
        model=gru,
        model_name="gru",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    with open(
        os.path.join(CFG.reports_dir, "train_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(asdict(CFG), f, indent=4, ensure_ascii=False)

    print("\nEntraînement terminé.")
    print(f"Modèles sauvegardés dans : {CFG.models_dir}")
    print(
        f"Historiques sauvegardés dans : {os.path.join(CFG.reports_dir, 'histories')}"
    )


if __name__ == "__main__":
    main()
