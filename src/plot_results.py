from __future__ import annotations

import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class PlotConfig:
    reports_dir: str = "reports"
    figures_dir: str = "figures"
    n_hours_to_plot: int = 7 * 24  # une semaine


CFG = PlotConfig()


MODEL_DISPLAY_NAMES = {
    "baseline": "Baseline",
    "ridge": "Ridge",
    "mlp": "MLP",
    "cnn1d": "CNN 1D",
    "gru": "GRU",
}


HISTORY_FILES = {
    "MLP": "history_mlp.json",
    "CNN 1D": "history_cnn1d.json",
    "GRU": "history_gru.json",
}


def ensure_figures_dir() -> None:
    os.makedirs(CFG.figures_dir, exist_ok=True)


def load_metrics() -> pd.DataFrame:
    path = os.path.join(CFG.reports_dir, "metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\nLance d'abord : python -m src.evaluate"
        )
    return pd.read_csv(path, index_col=0)


def load_predictions() -> pd.DataFrame:
    path = os.path.join(CFG.reports_dir, "predictions_test.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\nLance d'abord : python -m src.evaluate"
        )
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    return df


def plot_metric_bar(
    metrics: pd.DataFrame, metric: str, output_name: str, ylabel: str
) -> None:
    metrics_sorted = metrics.sort_values(metric)

    plt.figure(figsize=(7, 4))
    plt.bar(metrics_sorted.index, metrics_sorted[metric])
    plt.ylabel(ylabel)
    plt.xlabel("Modèle")
    plt.title(f"Comparaison des modèles selon {ylabel}")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.figures_dir, output_name), dpi=200)
    plt.close()


def plot_predictions_one_week(predictions: pd.DataFrame) -> None:
    df = predictions.iloc[: CFG.n_hours_to_plot].copy()

    plt.figure(figsize=(11, 5))
    plt.plot(df["time"], df["y_true"], label="Vérité terrain", linewidth=2)

    for col in ["baseline", "ridge", "mlp", "cnn1d", "gru"]:
        if col in df.columns:
            plt.plot(
                df["time"], df[col], label=MODEL_DISPLAY_NAMES.get(col, col), alpha=0.85
            )

    plt.xlabel("Date")
    plt.ylabel("Température (°C)")
    plt.title("Prévision de température sur une semaine de test")
    plt.legend(ncol=2)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.figures_dir, "predictions_one_week.png"), dpi=200)
    plt.close()


def plot_learning_curve(model_name: str, history_filename: str) -> None:
    path = os.path.join(CFG.reports_dir, "histories", history_filename)
    if not os.path.exists(path):
        print(f"Historique absent, figure ignorée : {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)

    plt.figure(figsize=(6, 4))
    plt.plot(history["loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Époque")
    plt.ylabel("MSE normalisée")
    plt.title(f"Courbe d'apprentissage -- {model_name}")
    plt.legend()
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(
        os.path.join(CFG.figures_dir, f"learning_curve_{safe_name}.png"), dpi=200
    )
    plt.close()


def save_latex_table(metrics: pd.DataFrame) -> None:
    table = metrics.copy()
    table = table[["MAE_degC", "RMSE_degC", "R2"]]
    table = table.rename(
        columns={
            "MAE_degC": "MAE (°C)",
            "RMSE_degC": "RMSE (°C)",
            "R2": "$R^2$",
        }
    )

    latex = table.to_latex(
        float_format="%.3f",
        escape=False,
        column_format="lccc",
        caption="Performances des modèles sur l'ensemble de test.",
        label="tab:resultats_temperature",
    )

    output_path = os.path.join(CFG.reports_dir, "metrics_latex_table.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"Tableau LaTeX sauvegardé dans : {output_path}")


def main() -> None:
    ensure_figures_dir()

    metrics = load_metrics()
    predictions = load_predictions()

    plot_metric_bar(
        metrics=metrics,
        metric="MAE_degC",
        output_name="model_comparison_mae.png",
        ylabel="MAE (°C)",
    )

    plot_metric_bar(
        metrics=metrics,
        metric="RMSE_degC",
        output_name="model_comparison_rmse.png",
        ylabel="RMSE (°C)",
    )

    plot_predictions_one_week(predictions)

    for model_name, history_filename in HISTORY_FILES.items():
        plot_learning_curve(model_name, history_filename)

    save_latex_table(metrics)

    print("\nFigures générées dans le dossier figures/ :")
    print("- model_comparison_mae.png")
    print("- model_comparison_rmse.png")
    print("- predictions_one_week.png")
    print("- learning_curve_mlp.png")
    print("- learning_curve_cnn_1d.png")
    print("- learning_curve_gru.png")


if __name__ == "__main__":
    main()
