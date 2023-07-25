import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib as ta
from datetime import datetime
import seaborn as sns
import warnings
import yfinance as yf
from scipy import stats
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")


def dealing_with_outliers_ma_res_iqr(data, lookback, n, method):
    """Use moving average to get a residual series from the
        original dataframe. We use the IQR and quantiles to
        make anomalous data-points nan values. Then we replace
        these nan values using interpolation with a linear method.
    """
    # Create a dataframe with moving averages for each column
    ma = pd.DataFrame(index=data.index)
    for i, j in data.items():
        ma[f"{i}"] = ta.SMA(data[f"{i}"].values, timeperiod=lookback)

    # Subtract the moving averages from the original dataframe
    res = data - ma

    # Computing the IQR
    Q1 = res.quantile(0.25)
    Q3 = res.quantile(0.75)
    IQR = Q3 - Q1

    # Calculate the upper and lower bounds
    lw_bound = Q1 - (n * IQR)
    up_bound = Q3 + (n * IQR)

    # Values outside the range will become nana values
    res[res <= lw_bound] = np.nan
    res[res >= up_bound] = np.nan

    # Use interpolation to replace these nan values
    res = res.interpolate(method=method)

    # Recompose the original dataframe
    prices = pd.DataFrame((res + ma))
    prices.dropna(inplace=True)

    return prices


def transfer_entropy(X, Y, delay=1, gaussian_sigma=None):
    if len(X) != len(Y):
        raise ValueError('Time series entries need to have the same length')

    n = float(len(X[delay:]))
    binX = int((max(X) - min(X)) / (2 * stats.iqr(X) / (len(X) ** (1.0 / 3))))
    binY = int((max(Y) - min(Y)) / (2 * stats.iqr(Y) / (len(Y) ** (1.0 / 3))))

    x3 = np.array([X[delay:], Y[:-delay], X[:-delay]])
    x2 = np.array([X[delay:], Y[:-delay]])
    x2_delay = np.array([X[delay:], X[:-delay]])

    p3, bin_p3 = np.histogramdd(sample=x3.T, bins=[binX, binY, binX])
    p2, bin_p2 = np.histogramdd(sample=x2.T, bins=[binX, binY])
    p2delay, bin_p2delay = np.histogramdd(sample=x2_delay.T, bins=[binX, binX])
    p1, bin_p1 = np.histogramdd(sample=np.array(X[delay:]), bins=binX)

    p1 = p1 / n
    p2 = p2 / n
    p2delay = p2delay / n
    p3 = p3 / n

    if gaussian_sigma is not None:
        s = gaussian_sigma
        p1 = gaussian_filter(p1, sigma=s)
        p2 = gaussian_filter(p2, sigma=s)
        p2delay = gaussian_filter(p2delay, sigma=s)
        p3 = gaussian_filter(p3, sigma=s)

    Xrange = bin_p3[0][:-1]
    Yrange = bin_p3[1][:-1]
    X2range = bin_p3[2][:-1]

    elements = []
    for i in range(len(Xrange)):
        px = p1[i]
        for j in range(len(Yrange)):
            pxy = p2[i][j]
            for k in range(len(X2range)):
                pxx2 = p2delay[i][k]
                pxyx2 = p3[i][j][k]
                arg1 = float(pxy * pxx2)
                arg2 = float(pxyx2 * px)
                if arg1 == 0.0:
                    arg1 = float(1e-8)
                if arg2 == 0.0:
                    arg2 = float(1e-8)
                term = pxyx2 * np.log2(arg2) - pxyx2 * np.log2(arg1)
                elements.append(term)

    TE = np.sum(elements)
    return TE


def compute_transfer_entropy_parallelised(data, lag):
    return transfer_entropy(data.iloc[:, 0], data.iloc[:, 1], delay=lag)


def run():
    # Import data #
    symbol = ["^GSPC", "GC=F"]
    df1 = yf.download(symbol[0], datetime(2000, 1, 1),
                      datetime.now()).drop(columns=["Adj Close", "Volume"])

    df2 = yf.download(symbol[1], datetime(2000, 1, 1),
                      datetime.now()).drop(columns=["Adj Close", "Volume"])

    # Clean data #
    df1 = dealing_with_outliers_ma_res_iqr(data=df1, lookback=10, n=2, method="linear")
    df2 = dealing_with_outliers_ma_res_iqr(data=df2, lookback=10, n=2, method="linear")

    closes = pd.DataFrame([df1["Close"], df2["Close"]]).T
    closes.columns = symbol
    closes = closes.loc["2001-01-01":, :]

    returns = closes.pct_change()  # returns data
    returns.dropna(inplace=True)

    # Transfer entropy
    lags = np.arange(1, 100)

    # Parallelization
    results = Parallel(n_jobs=11)(delayed(compute_transfer_entropy_parallelised)(returns, lag) for lag in lags)

    plt.scatter(lags, results)
    plt.xlabel("Lag")
    plt.ylabel("Transfer Entropy")
    plt.xticks(lags)
    plt.show()


if __name__ == "__main__":
    run()




