import numpy as np
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
from datetime import datetime
from sklearn.svm import SVR
from scipy.signal import argrelextrema, argrelmin
from kneed import KneeLocator
from scipy.linalg import hankel
import seaborn as sns
import warnings
import yfinance as yf

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
        ma[f"{i}"] = ta.SMA(j.values, timeperiod=lookback)

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


if __name__ == "__main__":
    # Import Data
    symbol = "EURUSD=X"
    df = yf.download(symbol, datetime(2000, 1, 1),
                     datetime.now()).drop(columns=["Adj Close", "Volume"])

    # df["returns"] = df.Close.pct_change()
    # df.dropna(inplace=True)

    # Remove Outliers
    df = dealing_with_outliers_ma_res_iqr(data=df, lookback=10, n=2, method="linear")

    # Create subset of data to deal with end effects by adding data - will remove this later
    n = 200
    subset = np.concatenate((df.Close.values, df.Close.values[-1:]*np.ones(n)))

    # Hankel Matrix of the Close Price Series
    hankel_matrix = hankel(subset)

    # Singular Value Decomposition
    U, S, VT = np.linalg.svd(hankel_matrix)

    # Detect where the Singular Values begin to stabilize - the largest singular values
    # via gradient nearing 0,"elbow point" of scree plot, centre of mass and a heuristic
    # S = (np.square(S) / (np.sum(np.square(S)))) <---- this is used for percent variance explained

    # 1. Gradient Method
    S_ar = np.array(S)
    grad = S_ar[(-0.5 <= np.gradient(S_ar)) & (np.gradient(S_ar) < 0)][0]
    grad = int(np.where(S == grad)[0])

    # 2. Elbow Point of Scree Plot
    kn = KneeLocator(x=np.arange(1, len(S) + 1), y=S, curve="convex", direction="decreasing")
    elbow_point = kn.elbow

    # 3. Centre of Mass
    centre_of_mass = (np.sum(S * np.arange(1, len(S) + 1)) / np.sum(S))
    centre_mass_thresh = int(0.1 * centre_of_mass)

    # 4. Heuristic
    heuristic_sel = int(np.sqrt(len(S) / 2))

    # Average of the different methods
    print("Gradient Method: ", grad)
    print("Elbow Point Method", elbow_point)
    print("Centre Mass Method", centre_mass_thresh)
    print("Heuristic Method", heuristic_sel)

    first_k_singulars = int(np.round((grad + elbow_point + centre_mass_thresh + heuristic_sel) / 4, 0))
    print("\nFirst effective k Singular Values:", first_k_singulars)

    # Visualize the Singular Values
    fig, axes = plt.subplots(nrows=2, ncols=1, dpi=50, sharex=True)
    axes[0].scatter(np.arange(1, len(S) + 1), S, c="black", label="Singular Values")
    axes[0].set_ylim(np.min(S) - 1, np.max(S) + 1)
    axes[0].axvline(x=grad, lw=0.8, c="blue")
    axes[0].axvline(x=elbow_point, lw=0.8, c="blue")
    axes[0].axvline(x=centre_mass_thresh, lw=0.8, c="blue")
    axes[0].axvline(x=heuristic_sel, lw=0.8, c="blue")
    axes[0].axvline(x=first_k_singulars, lw=0.8, c="red")
    axes[0].legend(loc="upper right")

    axes[1].plot(np.arange(1, len(S) + 1), np.gradient(S), c="black", label="Gradient of Singular Values")
    axes[1].axhline(y=0, c="blue", lw=0.8)
    axes[1].set_ylim(-10, np.max(np.gradient(S)) + 0.5)
    axes[1].legend(loc="lower right")

    plt.suptitle("Scree Plot - Singular Values")
    multi = MultiCursor(fig.canvas, fig.axes, useblit=True, horizOn=True, vertOn=True, color="black", lw=0.5)
    plt.show()

    # Set values after the first k effective singular values to zero
    S = [0 if i > first_k_singulars else j for i, j in zip(range(len(S)), S)]

    # Reconstruct the hankel matrix using SVd components
    close = U @ np.diag(S) @ VT

    # Average across the forward diagonal and keep the length of the original time series
    max_col = len(close[0])
    max_row = len(close)
    fdiag = [[] for _ in range(max_row + max_col - 1)]
    for x in range(max_col):
        for y in range(max_row):
            fdiag[x + y].append(close[y][x])

    avg_fdiag = []  # our smoothed series - it needs to be halved as it is a Hankel matrix
    for i, j in zip(fdiag, range(1, len(fdiag) + 1)):
        avg_fdiag.append(np.sum(i) / j)

    close = avg_fdiag[:len(subset)]

    # Visualize the smoothed time series
    plt.plot(subset)
    plt.plot(close)
    plt.plot(close[:len(df)])  # <-------- denoised data we would want to keep
    plt.show()


