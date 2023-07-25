import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader as pdr
from datetime import datetime
import talib as ta
import yfinance as yf

sns.set_style("darkgrid")

stocks = "TMO"
start, end = datetime(2000, 1, 1), datetime.now()
df = yf.download(stocks, start, end).drop(columns=["Open", "Adj Close"])
df.dropna(inplace=True)
print(df)


def vwap(price, volume):
    q = volume
    p = price
    return (p * q).cumsum() / q.cumsum()


df["vwap"] = vwap(df.Close, df.Volume)

tp = (df.High + df.Low + df.Close) / 3
vwap = (tp * df.Volume).cumsum() / df.Volume.cumsum()

df.Close.plot()
df.vwap.plot(c="red")
vwap.plot(c="green")
plt.show()


def rolling_vwap(prices, volumes, window_size):
    """
    Calculate the rolling Volume-Weighted Average Price (VWAP).

    Parameters:
        prices (list or numpy.array): Historical prices.
        volumes (list or numpy.array): Corresponding volumes for each price.
        window_size (int): The size of the rolling window.

    Returns:
        list: Rolling VWAP values.
    """
    if len(prices) != len(volumes):
        raise ValueError("Lengths of 'prices' and 'volumes' must be the same.")

    rolling_vwaps = []
    sum_price_volume = 0
    sum_volume = 0

    for i in range(len(prices)):
        sum_price_volume += prices[i] * volumes[i]
        sum_volume += volumes[i]

        if i >= window_size - 1:
            if sum_volume != 0:
                rolling_vwap = sum_price_volume / sum_volume
            else:
                rolling_vwap = None

            rolling_vwaps.append(rolling_vwap)

            # Remove the oldest data point from the rolling sum
            oldest_price_volume = prices[i - window_size + 1] * volumes[i - window_size + 1]
            sum_price_volume -= oldest_price_volume
            sum_volume -= volumes[i - window_size + 1]

    return rolling_vwaps


window_size = 50
vwap = rolling_vwap(df.Close, df.Volume, window_size=window_size)

plt.plot(df.Close.values)
plt.plot((window_size - 1) * [np.nan] + vwap)
plt.show()