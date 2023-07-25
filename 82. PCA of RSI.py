import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import talib as ta
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sns.set_style("darkgrid")

df = yf.download("WMT", datetime(2000, 1, 1), datetime.now()-timedelta(1))

for i in range(3, 51):
    df[f"RSI_{i}"] = ta.RSI(df.Close, timeperiod=i)
df.dropna(inplace=True)

sc = StandardScaler()
msc = MinMaxScaler()
pca = PCA(n_components=1)

rsi = pd.DataFrame(msc.fit_transform(pca.fit_transform(sc.fit_transform(df.iloc[:, 6:]))) * 100, index=df.index, columns=["RSI_PCA"])
up, _, down = ta.BBANDS(rsi.RSI_PCA, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

fig, axes = plt.subplots(nrows=2, ncols=1, dpi=50, sharex=True)
axes[0].plot(df.Close)
axes[1].plot(rsi)
axes[1].plot(up, c="black", ls="--")
axes[1].plot(down, c="black", ls="--")
plt.show()