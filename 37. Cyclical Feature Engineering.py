import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta


def perform_cyclical_engineering(df):
    idx = df.index
    quarter = idx.quarter
    month = idx.month
    week = idx.week
    day = idx.day

    df["sin_quarter"] = np.sin(2. * np.pi * quarter / 4)
    df["cos_quarter"] = np.cos(2. * np.pi * quarter / 4)
    df["sin_month"] = np.sin(2. * np.pi * month / 12)
    df["cos_month"] = np.cos(2. * np.pi * month / 12)
    df["sin_week"] = np.sin(2. * np.pi * week / 52)
    df["cos_week"] = np.cos(2. * np.pi * week / 52)
    df["sin_day"] = np.sin(2. * np.pi * day / 365)
    df["cos_day"] = np.cos(2. * np.pi * day / 365)

    return df


df = yf.download("AAPL", start=datetime(2000, 1, 1), end=datetime.now() - timedelta(1))
df = perform_cyclical_engineering(df)

print(df)
