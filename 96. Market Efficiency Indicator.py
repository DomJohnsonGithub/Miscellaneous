from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import talib as ta
import yfinance as yf
from scipy.stats import entropy

sns.set_style("darkgrid")

stock = "GS"
ts = yf.download(
    stock,
    start=datetime(2000, 1, 1),
    end=datetime.now() - timedelta(1)).drop(
    columns=["Adj Close"]).dropna()

ts["log_rets"] = np.log1p(ts.Close.pct_change())
ts.dropna(inplace=True)

# MARKET EFFICIENCY INDICATOR

from nolds import dfa, hurst_rs, sampen, corr_dim
from scipy.stats import entropy

HE = ts.log_rets.rolling(252).apply(lambda x: hurst_rs(x))
FD = ts.Close.rolling(252).apply(lambda x: corr_dim(x, emb_dim=2))
ENT = ts.log_rets.rolling(252).apply(lambda x: entropy(np.histogram(np.where(x > 0, 2, 1), bins="rice")[0]))

lagged = pd.DataFrame(ts.log_rets.values, columns=["t"], index=ts.index)
lagged["t+1"] = lagged.t.shift(1)
lagged.dropna(inplace=True)
P = lagged.t.rolling(252).corr(lagged["t+1"])

HE.dropna(inplace=True)
FD.dropna(inplace=True)
ENT.dropna(inplace=True)
P.dropna(inplace=True)


EFFICIENCY_INDEX = pd.DataFrame(np.sqrt((np.array(HE)[1:] - 0.5)**2 + (np.array(FD)[1:] - 1.5)**2 + ((np.array(ENT)[1:] - 1)/2)**2 + (np.array(P)/2)**2),
                                index=P.index, columns=["EI"])

EFFICIENCY_INDEX["Signal"] = EFFICIENCY_INDEX.EI.rolling(100).apply(lambda x: np.percentile(x, q=25))

fig, axes = plt.subplots(nrows=2, ncols=1, dpi=50, sharex=True)
axes[0].plot(ts.Close, c="blue")
axes[1].plot(EFFICIENCY_INDEX["EI"], c="red")
axes[1].plot(EFFICIENCY_INDEX["Signal"], c="black")
plt.show()