from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import talib as ta
import warnings
import yfinance as yf
from finta.finta import TA

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")


def fetch_data(symbol, from_date, to_date, drop_cols=False, cols_to_drop=None):
    """ Fetch OHLC data."""
    df = yf.download(symbol, from_date, to_date)
    if drop_cols == True:
        df = df.drop(columns=cols_to_drop)

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


if __name__ == "__main__":

    # Fetch Data

    symbol = "HPE"  # ticker
    from_date = datetime(2000, 1, 1)
    to_date = datetime(2022, 2, 25)

    df = fetch_data(symbol=symbol, from_date=from_date,
                    to_date=to_date, drop_cols=True, cols_to_drop=["Adj Close"])

    returns = df.Close.diff().dropna()
    data = pd.DataFrame(returns.values, index=returns.index, columns=["returns"])
    print(data)

    n = 50

    data["EMA"] = ta.EMA(data.returns, timeperiod=n)
    data["HULL_1"] = ta.WMA(2*ta.WMA(data.returns, timeperiod=int(np.round(n/2, 0))) - ta.WMA(data.returns, timeperiod=n), timeperiod=int(np.round(np.sqrt(n), 0)))
    data["HULL_2"] = ta.EMA(2*ta.EMA(data.returns, timeperiod=int(np.round(n/2, 0))) - ta.EMA(data.returns, timeperiod=n), timeperiod=int(np.round(np.sqrt(n), 0)))
    data["HULL_3"] = ta.TEMA(2*ta.TEMA(data.returns, timeperiod=int(np.round(n/2, 0))) - ta.TEMA(data.returns, timeperiod=n), timeperiod=int(np.round(np.sqrt(n), 0)))

    plt.plot(data.returns, c="black")
    plt.plot(data.EMA, c="orange")
    plt.plot(data.HULL_1, c="red")
    plt.plot(data.HULL_2, c="blue")
    plt.plot(data.HULL_3, c="purple")
    plt.show()











