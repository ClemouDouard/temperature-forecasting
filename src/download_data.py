import os

import pandas as pd
import requests

from src.config import CITY, END_DATE, FEATURES, LATITUDE, LONGITUDE, START_DATE

OUTPUT_DIR = "data/raw"
OUTPUT_FILE = f"{OUTPUT_DIR}/weather_{CITY}_hourly.csv"


def download_open_meteo_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(FEATURES),
        "timezone": "Europe/Paris",
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise ValueError("La réponse API ne contient pas de données horaires.")

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Données sauvegardées dans : {OUTPUT_FILE}")
    print(df.head())
    print(df.tail())
    print(df.info())


if __name__ == "__main__":
    download_open_meteo_data()
