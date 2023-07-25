from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as pdr
import seaborn as sns
from statsmodels.tsa.api import bds
import talib as ta
import warnings
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, kpss, acf
from arch.unitroot import PhillipsPerron, ZivotAndrews, VarianceRatio

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")


def fetch_data(symbol, from_date, to_date, cols_to_drop):
    """ Fetch OHLC data."""
    df = yf.download(symbol, from_date, to_date)
    df.drop(columns=cols_to_drop, inplace=True)

    return df


def outlier_treatment(data, lookback, n, method):
    """Use moving average to get a residual series from the
        original dataframe. We use the IQR and quantiles to
        make anomalous data-points nan values. Then we replace
        these nan values using interpolation with a linear method.
    """
    ma = pd.DataFrame(index=data.index)  # moving averages of each column
    for i, j in data.items():
        ma[f"{i}"] = ta.SMA(j.values, timeperiod=lookback)

    res = data - ma  # residual series

    Q1 = res.quantile(0.25)  # Quantile 1
    Q3 = res.quantile(0.75)  # Quantile 3
    IQR = Q3 - Q1  # IQR

    lw_bound = Q1 - (n * IQR)  # lower bound
    up_bound = Q3 + (n * IQR)  # upper bound

    res[res <= lw_bound] = np.nan  # set values outside range to NaN
    res[res >= up_bound] = np.nan

    res = res.interpolate(method=method)  # interpolation replaces NaN values

    prices = pd.DataFrame((res + ma))  # recompose original dataframe
    prices.dropna(inplace=True)  # drop NaN values

    return prices


def statistical_stationarity_tests(data):
    adf = adfuller(x=data, autolag="AIC")
    kpsst = kpss(x=data, nlags="auto")
    pp = PhillipsPerron(y=data)
    za = ZivotAndrews(y=data, method="AIC", lags=30)
    vr = VarianceRatio(y=data, lags=30)

    print("\nADF t-stat: ", adf[0], ", p-value: ", adf[1], ", Critical Values: ", adf[4])
    print("\nKPSS t-stat", kpsst[0], ", p-value: ", kpsst[1], ", Critical Values: ", kpsst[3])
    print("\n", pp)
    print("\n", za)
    print("\n", vr)

    autocrnf = acf(data)
    plt.bar(np.arange(1, len(autocrnf) + 1), autocrnf)
    plt.show()


if __name__ == "__main__":
    # Fetch OHLC Data
    symbol = "EURUSD=X"  # ticker
    from_date = datetime(2000, 1, 1)
    to_date = datetime.now() - timedelta(1)
    drop_columns = ["Adj Close", "Volume"]

    df = fetch_data(symbol=symbol, from_date=from_date,
                    to_date=to_date, cols_to_drop=drop_columns)

    df["returns"] = df.Close.pct_change()  # returns data

    # Remove Outliers
    df = outlier_treatment(df.dropna(), lookback=10, n=2, method="linear")
    print(df)

    # Stationarity Tests
    statistical_stationarity_tests(data=df.Close)  # non-stationary
    print("")
    statistical_stationarity_tests(data=df.returns)  # stationary


