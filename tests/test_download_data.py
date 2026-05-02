import pandas as pd

df = pd.read_csv("data/raw/weather_Paris_hourly.csv")
print(df.head())
print(df.shape)
print(df.isna().sum())
