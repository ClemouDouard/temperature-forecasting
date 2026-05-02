from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.linear_model import Ridge


def persistent_baseline(
    X: np.ndarray, temperature_feature_index: int = 0
) -> np.ndarray:
    return X[:, -1, temperature_feature_index]


def build_ridge_model(alpha: float = 1.0) -> Ridge:
    return Ridge(alpha=alpha)


def build_mlp(input_shape: Tuple[int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.10)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mlp_forecaster")
    return model


def build_cnn1d(input_shape: Tuple[int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu")(
        inputs
    )
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu")(
        x
    )
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn1d_forecaster")
    return model


def build_gru(input_shape: Tuple[int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.GRU(32)(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="gru_forecaster")
    return model


def compile_keras_model(
    model: tf.keras.Model, learning_rate: float = 1e-3
) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model
