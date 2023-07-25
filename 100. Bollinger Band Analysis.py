import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import talib as ta


sns.set_style("darkgrid")


DATA_SOURCE = "C:\\Users\\domin\\PycharmProjects\\Miscellaneous\\Data.h5"
with pd.HDFStore(DATA_SOURCE, "r") as store:
    df = store.get("Data")

print(df)

# Bollinger Bands
df["UP_BB"], _, df["LOW_BB"] = ta.BBANDS(df.Close, timeperiod=10, nbdevup=3, nbdevdn=3, matype=0)

x1 = df["UP_BB"]

x2 = x1.pct_change() # pct_change
x3 = x2.apply(np.log1p) # pct_change with log(1+x)

x4 = x1 - x1.rolling(window=20).mean() # mean-centering
x5 = x4.apply(np.log1p) # mean-centering with log(1+p)

x6 = x1.diff() # differencing
x7 = np.log(x6) # log of differenced series

fig, axes = plt.subplots(nrows=4, ncols=2, dpi=50)

axes[0, 0].plot(df.Close, lw=1, c="black")
axes[0, 0].plot(df.UP_BB, lw=0.8, c="green")
axes[0, 0].plot(df.LOW_BB, lw=0.8, c="green")
axes[0, 0].set_title("EUR/USD Close Price with Bollinger Bands")

sns.distplot(x1, ax=axes[0, 1])
axes[0, 1].set_title("X1: Bollinger Band (BB) Distribution")

sns.distplot(x2, ax=axes[1, 0])
axes[1, 0].set_title("X2: BB.pct_change() Distribution")

sns.distplot(x3, ax=axes[1, 1])
axes[1, 1].set_title("X3: BB.pct_change().apply(np.log1p) Distribution")

sns.distplot(x4, ax=axes[2, 0])
axes[2, 0].set_title("X4: BB - BB.rolling(20).mean() Distribution")

sns.distplot(x5, ax=axes[2, 1])
axes[2, 1].set_title("X5: (BB - BB.rolling(20).mean()).apply(np.log1p) Distrb'n")

sns.distplot(x6, ax=axes[3, 0])
axes[3, 0].set_title("X6: BB.diff() Distrb'n")

sns.distplot(x6, ax=axes[3, 1])
axes[3, 1].set_title("X7: np.log(BB.diff()) Distrb'n")
plt.show()