import numpy as np
import pandas as pd
from pathlib import Path

DATA_SOURCE = Path("C:\\Users\\domin\\PycharmProjects\\Miscellaneous\\ASSETS.h5")


def calculate_CCHV(df, window=22):
    """
    Calculate Close-to-Close Historical Volatility.
    """
    rolling_std = df["Log_Rets"].rolling(window=window).std()
    return np.sqrt(252) * rolling_std


def calculate_PARHV(df, window=22):
    """
    Calculate Parkinson Historical Volatility.
    """
    log_ratio = np.log(df["High"] / df["Low"]) ** 2
    rolling_sum = log_ratio.rolling(window=window).sum()
    return np.sqrt(252 / (4 * window * np.log(2)) * rolling_sum)


def calculate_GKHV(df, window=22):
    """
    Calculate Garman-Klass Historical Volatility.
    """
    volatility = 0.5 * np.log(df["High"] / df["Low"]) ** 2
    log_ratio = (2 * np.log(2) - 1) * np.log(df["Close"] / df["Open"]) ** 2
    rolling_sum = (volatility - log_ratio).rolling(window=window).sum()
    return np.sqrt(252 / 22 * rolling_sum)


def calculate_GKYZHV(df, window=22):
    """
    Calculate Garman-Klass Yang-Zhang Historical Volatility.
    """
    returns = 0.5 * np.log(df["Open"] / df["Close"].shift(1)) ** 2
    volatility = 0.5 * np.log(df["High"] / df["Low"]) ** 2
    log_ratio = (2 * np.log(2) - 1) * np.log(df["Close"] / df["Open"]) ** 2
    rolling_sum = (returns + volatility - log_ratio).rolling(window=window).sum()
    return np.sqrt(252 / window * rolling_sum)


with pd.HDFStore(DATA_SOURCE, "r") as store:
    df = store.get("ASSET/data")

df = df.loc[pd.IndexSlice[:, "KO"], ["High", "Low", "Open", "Close", "Volume"]].droplevel(level=1)
df["Returns"] = df["Close"].pct_change()
df["Log_Rets"] = np.log(df["Close"] / df["Close"].shift(1))

df["CCHV"] = calculate_CCHV(df)
df["PARHV"] = calculate_PARHV(df)
df["GKHV"] = calculate_GKHV(df)
df["GKYZHV"] = calculate_GKYZHV(df)

print(df)
