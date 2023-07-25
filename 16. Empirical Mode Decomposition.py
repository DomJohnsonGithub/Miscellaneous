import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib as ta
from datetime import datetime
import seaborn as sns
import warnings
import yfinance as yf
from PyEMD import EMD

def run():
    # Import data #
    symbol = ["^GSPC", "GC=F"]
    df = yf.download(symbol[0], datetime(2000, 1, 1),
                     datetime.now())["Close"]
    print(df)

    # Empirical Mode Decomposition
    emd = EMD.EMD()

    # Intrinsic Mode Functions
    imfs = emd.emd(df.values).T
    n_imfs = np.shape(imfs)[1]

    fig, axes = plt.subplots(nrows=n_imfs, ncols=1, dpi=50, sharex=True)
    for i in range(n_imfs):
        if i < n_imfs - 1:
            axes[i].plot(imfs[:, i], label=f"IMF {i+1}")
            axes[i].legend(loc="upper right")
        else:
            axes[i].plot(imfs[:, i], label=f"Residual")
            axes[i].legend(loc="upper right")
    plt.suptitle("Intrinsic Mode Functions")
    plt.show()


if __name__ == "__main__":
    run()