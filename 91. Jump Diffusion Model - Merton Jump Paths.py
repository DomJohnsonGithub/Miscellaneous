from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as pdr
import math
import talib as ta
from pykalman import KalmanFilter
import yfinance as yf

sns.set_style("darkgrid")

stock = "^GSPC"
df = yf.download(
    stock,
    start=datetime(2000, 1, 1),
    end=datetime.now() - timedelta(1)).drop(
    columns=["Adj Close"])

df["returns"] = np.log1p(df.Close.pct_change())
df.dropna(inplace=True)


# Jump Diffusion

def merton_jump_paths(S, T, r, sigma, lam, m, v, steps, Npaths):
    size = (steps, Npaths)
    dt = T / steps
    poi_rv = np.multiply(np.random.poisson(lam * dt, size=size),
                         np.random.normal(m, v, size=size)).cumsum(axis=0)
    geo = np.cumsum(((r - sigma ** 2 / 2 - lam * (m + v ** 2 * 0.5)) * dt + sigma * np.sqrt(dt) * np.random.normal(size=size)), axis=0)

    return np.exp(geo + poi_rv) * S


S = 100  # current stock price
T = 1  # time to maturity
r = 0.02  # risk-free rate
m = 0  # mean of jump size
v = 0.3 / 100  # standard deviation of jump
lam = 1  # intensity of jump i.e. number of jumps per annum
steps = 21  # time steps
Npaths = 30  # number of paths to simulate
sigma = 0.2  # annaul standard deviation , for weiner process

data = df.loc[:, ["Close", "returns"]].copy()
S = data.Close[-1]
sigma = data.returns.std() * np.sqrt(252) / 100
m = 0.0065

j = merton_jump_paths(S, T, r, sigma, lam, m, v, steps, Npaths)

average_close_price = []
runs = 500
for i in range(runs):
    j = merton_jump_paths(S, T, r, sigma, lam, m, v, steps, Npaths)
    x = j - (j[0] - S)
    average_close_price.append(np.mean(x[-1, :]))
    a = np.arange(len(data))
    plt.plot(a, data.Close.to_list(), c="purple")
    a = np.append(a, np.arange(a[-1], len(x) + a[-1]))
    plt.plot(a[-len(x):], x, c="orange")

plt.show()
print(average_close_price)
print(np.mean(average_close_price))

sns.distplot(average_close_price, bins="rice", kde=True)
plt.axvline(x=S, c="red")
plt.show()
