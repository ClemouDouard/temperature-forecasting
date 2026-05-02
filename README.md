# Temperature Forecasting with Deep Learning

> Educational project for the FIDLE CNRS MOOC report.

This repository contains a small deep learning project developed as part of a report for the **FIDLE CNRS MOOC — Introduction au Deep Learning**.

The goal is to predict the future temperature in Paris from past hourly weather observations. The project compares simple baselines and neural network models in order to illustrate several concepts from the MOOC: data preprocessing, supervised learning, dense neural networks, temporal models, optimization, validation and model evaluation.

## Project overview

The task is:

> Predict the temperature in 6 hours from the last 24 hours of meteorological observations.

The input data are hourly weather variables collected for Paris between 2021 and 2024.  
The target variable is the temperature at horizon `+6h`.

The models compared are:

| Model | Description |
|---|---|
| Persistent baseline | Predicts that the future temperature is equal to the last observed temperature |
| Ridge regression | Linear regression with L2 regularization |
| MLP | Dense neural network applied to the flattened input window |
| CNN 1D | Temporal convolutional neural network |
| GRU | Lightweight recurrent neural network |

## Data source

The data are retrieved from the **Open-Meteo Historical Weather API**.

The variables used are:

- temperature at 2 meters;
- relative humidity at 2 meters;
- sea-level pressure;
- cloud cover;
- wind speed at 10 meters;
- precipitation.

Additional cyclic time features are added during preprocessing:

- hour of day encoded with sine and cosine;
- day of year encoded with sine and cosine.

The raw data are not stored in the repository. They can be downloaded by running the data download script.

## Usage

Run the following commands:
```bash
poetry lock
python -m src.download_data
python -m src.preprocess
python -m src.train
python -m src.evaluate
python -m src.plot_results
```
