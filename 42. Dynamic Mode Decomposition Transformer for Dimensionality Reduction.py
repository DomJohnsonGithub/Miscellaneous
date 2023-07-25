import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import talib as ta
from deeptime.kernels import GaussianKernel
from deeptime.decomposition import DMD, EDMD, KernelEDMD
from sklearn.preprocessing import PowerTransformer
from sklearn.base import BaseEstimator, TransformerMixin

sns.set_style("darkgrid")

idx = pd.IndexSlice

# Get Data
symbol_names = ["^GSPC", "EURUSD=X"]
df = yf.download(symbol_names, start=datetime(1999, 1, 1), end=datetime.now() - timedelta(1)).drop(
    columns=["Adj Close", "Volume"]).dropna().stack(1).swaplevel(0).sort_index(level=0)
df.index.names = ["Ticker", "Date"]

# Feature Engineering
lags = [1, 2, 3, 4, 5]

# Returns
for lag in lags:
    df[f"returns_{lag}"] = df.groupby(level="Ticker").Close.pct_change(lag)

# Lag Returns
for lag in lags:
    df[f"returns_1_t-{lag}"] = df.groupby(level="Ticker")["returns_1"].shift(lag)

# Returns Momentum
for lag1 in lags:
    for lag2 in lags[1:]:
        if lag2 > lag1:
            df[f"momentum_{lag2}_{lag1}"] = df[f"returns_{lag2}"] - df[f"returns_{lag1}"]


# Ranges
def diff(data):
    return (data["Close"] - data["Close"].shift()) / (data["High"].shift() - data["Low"].shift())


def diff_v(data):
    return (data["High"] - data["Low"].shift()) / (data["High"].shift() - data["Low"].shift())


def sec_diff(data):
    return (data["diff"] - data["diff"].shift()) / (data["High"].shift() - data["Low"].shift())


def close_to_open_returns(data):
    return data.Close / data.Open.shift() - 1


df["diff"] = df.groupby(level="Ticker", group_keys=False).apply(diff)
df["diff_v"] = df.groupby(level="Ticker", group_keys=False).apply(diff_v)
df["sec_diff"] = df.groupby(level="Ticker", group_keys=False).apply(sec_diff)
df["co_returns"] = df.groupby(level="Ticker", group_keys=False).apply(close_to_open_returns)
df["high_low"] = (df.High - df.Low).diff()
df["open_low"] = df.Open - df.Close


# Sum of Returns
def sum_rets(data, n):
    return data["returns_1"].rolling(n).sum()


for i in symbol_names:
    rets = df.loc[idx[f"{i}", :], "returns_1"].droplevel(0)
    rets = pd.DataFrame(rets, index=rets.index, columns=["returns_1"])
    for window in lags[1:]:
        df.loc[idx[f"{i}", :], f"sum_returns_{window}"] = rets["returns_1"].rolling(window).sum().values


# Standard Deviation, Skewness, Kurtosis
def sd(data):
    return data.rolling(5).std().diff()


def skewness(data):
    return data.rolling(5).skew().diff()


def kurtosis(data):
    return data.rolling(5).kurt().diff()


df["standard_deviation"] = df.groupby(level="Ticker", group_keys=False)["returns_1"].apply(sd)
df["skewness"] = df.groupby(level="Ticker", group_keys=False)["returns_1"].apply(skewness)
df["kurtosis"] = df.groupby(level="Ticker", group_keys=False)["returns_1"].apply(kurtosis)


# Momentum and Trend Indicators
def rsi(data):
    return ta.RSI(data, timeperiod=5)


def cci(data):
    return ta.CCI(data.High, data.Low, data.Close, timeperiod=5)


def stoch(data):
    return ta.STOCHF(data.High, data.Low, data.Close, fastk_period=5, fastd_period=3, fastd_matype=0)[0]


def macd(data):
    return ta.MACD(data, fastperiod=3, slowperiod=5, signalperiod=2)[0]


def aroon(data):
    return ta.AROONOSC(data.High, data.Low, timeperiod=5)


def bop(data):
    return ta.BOP(data.Open, data.High, data.Low, data.Close)


def adx(data):
    return ta.ADX(data.High, data.Low, data.Close, timeperiod=5)


df["RSI"] = df.groupby(level="Ticker", group_keys=False).Close.apply(rsi)
df["CCI"] = df.groupby(level="Ticker", group_keys=False).apply(cci)
df["STOCH"] = df.groupby(level="Ticker", group_keys=False).apply(stoch)
df["AROON"] = df.groupby(level="Ticker", group_keys=False).apply(aroon)
df["ASOI"] = df.iloc[:, -4:].mean(axis=1)
df["BOP"] = df.groupby(level="Ticker", group_keys=False).apply(bop)
df["MACD"] = df.groupby(level="Ticker", group_keys=False).Close.apply(macd)
df["ADX"] = df.groupby(level="Ticker", group_keys=False).apply(adx).diff()
df = df.drop(columns=["RSI", "CCI", "STOCH", "AROON"])

# Cyclical Features
df = df.unstack(0)
index = df.index
quarter, month, week, day = index.quarter, index.month, index.isocalendar().week, index.day
df = df.stack(1).swaplevel(0).sort_index(level=0)

for transform in ["sin", "cos"]:
    for freq, frequency, num in zip(["quarter", "month", "week", "day"],
                                    [quarter, month, week, day], [4, 12, 52, 365]):
        if transform == "sin":
            df[f"{transform}_{freq}"] = np.tile(np.sin(2 * np.pi * frequency / num), len(symbol_names))
        else:
            df[f"{transform}_{freq}"] = np.tile(np.cos(2 * np.pi * frequency / num), len(symbol_names))

sin, cos = "sin", "cos"
for freq in ["quarter", "month", "week", "day"]:
    df[f"{freq}"] = df[f"{sin}_{freq}"] - df[f"{cos}_{freq}"]
df = df.drop(
    columns=[f"{transform}_{freq}" for transform in ["sin", "cos"] for freq in ["quarter", "month", "week", "day"]])

# Remove non-stationary variables and drop NaN values
df = df.iloc[:, 4:].dropna()

# SP500 Returns Data
data = df.loc[idx["^GSPC", :], "returns_1":"sum_returns_5"].droplevel(0)
print(data.columns)
print(np.shape(data))


class DynamicModeDecompositionTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.n_components = int
        self.modes = None

    def fit(self, X, y=None):
        U, S, V = np.linalg.svd(X)
        cum_sum_variance = np.cumsum(np.square(S) / (np.sum(np.square(S))))
        self.n_components = int(min(np.argwhere(cum_sum_variance >= 0.95)))
        dmd = DMD(mode="exact", rank=self.n_components)
        fit = dmd.fit((X[:-1], X[1:])).fetch_model()
        self.modes = fit.modes

    def transform(self, X, y=None):
        transformed_data = (X @ self.modes.T).real
        unique_data = np.unique(transformed_data.T, axis=0)
        return unique_data.T

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)


p = PowerTransformer()
sc_data = p.fit_transform(data)

dmd = DynamicModeDecompositionTransformer()
data = dmd.fit_transform(sc_data)

plt.plot(data)
plt.show()
