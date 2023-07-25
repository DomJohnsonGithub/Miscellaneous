from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import talib as ta
from tqdm import tqdm
import warnings
import yfinance as yf
import os

os.environ["R_HOME"] = r"C:\Program Files\R\R-4.3.1"
os.environ["PATH"] = r"C:\Program Files\R\R-4.3.1\bin\x64\R.exe\bin\x64" + ";" + os.environ["PATH"]

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter

numpy2ri.activate()

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


def ami_optimal_time_delay(data, lags):
    dataframe = pd.DataFrame(data)
    for i in range(1, lags + 1):
        dataframe[f"Lag_{i}"] = dataframe.iloc[:, 0].shift(i)
    dataframe.dropna(inplace=True)

    bins = int(np.round(2 * (len(dataframe)) ** (1 / 3), 0))

    def calc_mi(x, y, bins):
        c_xy = np.histogram2d(x, y, bins)[0]
        mi = mutual_info_score(None, None, contingency=c_xy)
        return mi

    def mutual_information(dataframe, lags, bins):
        mutual_information = []
        for i in tqdm(range(1, lags + 1)):
            mutual_information.append(calc_mi(dataframe.iloc[:, 0], dataframe[f"Lag_{i}"], bins=bins))

        return np.array(mutual_information)

    average_mi = mutual_information(dataframe, lags, bins)
    first_minima = argrelmin(average_mi)[0][0]

    return average_mi, first_minima


if __name__ == "__main__":

    # Fetch OHLC Data
    symbol = "EURUSD=X"  # ticker
    from_date = datetime(2000, 1, 1)
    to_date = datetime.now()
    drop_columns = ["Adj Close", "Volume"]

    df = fetch_data(symbol=symbol, from_date=from_date,
                    to_date=to_date, cols_to_drop=drop_columns)

    # Treat Outliers
    df = outlier_treatment(df, lookback=10, n=2, method="linear")

    # Using rpy2, transform data to r object
    data = numpy2ri.numpy2rpy(np.array(df.Close))  # data from numpy to r
    fractal = importr("fractal")  # fractal package

    determinism_test = fractal.determinism(
        data,  # univariate time series
        6,  # max dimension
        62,  # time lag
        int(np.round(len(data)/10, 0)),  # the number of points along the trajectory of the current point that
        # must be exceeded in order for another point in the phase space to be considered a neighbor candidate.
        ro.NULL, ro.NULL, ro.NULL,  # scale.min, scale.max, resolution
        "aaft",  # method - takes "aaft", "phase", "ce", "dh"
        10,  # number of surrogate realizations
        True,  # attach summary results
        seed=0)  # initial seed value for generating surrogate realizations

    # Original Data E-Statistics
    orig_scale = determinism_test[0][2]
    E_dims = [np.expand_dims(determinism_test[0][1][:, i], axis=1) for i in range(6)]

    weights = np.flip((np.arange(1, len(orig_scale) + 1))) * np.flip(
        np.linspace(1, len(orig_scale) + 1, len(orig_scale))) ** 12
    lines = [np.polyfit(np.squeeze(orig_scale[-len(E_dims[i][~np.isnan(E_dims[i])]):]), E_dims[i][~np.isnan(E_dims[i])],
                        deg=1, w=weights[:len(E_dims[i][~np.isnan(E_dims[i])])]) for i in range(6)]
    lines = np.array(lines)

    # Visualize the Kaplan Test
    for i in range(6):
        plt.plot(orig_scale, determinism_test[0][1][:, i], label=f"m={i}")
        plt.plot(orig_scale, orig_scale*lines[i, 0] + lines[i, 1], ls="--", c="black")
    plt.legend(loc="lower right")
    plt.show()

    # E-Statistics for Surrogate Data
    E_dims_surrogates = [determinism_test[1][i][1] for i in range(10)]
    surrogate_scales = [determinism_test[1][i][2] for i in range(10)]

    weights = [np.flip((np.arange(1, len(surrogate_scales[i]) + 1))) * np.flip(
        np.linspace(1, len(surrogate_scales[i]) + 1, len(surrogate_scales[i]))) ** 12 for i in range(10)]

    lines_S = [np.polyfit(np.squeeze(surrogate_scales[i][-len(E_dims_surrogates[i][:, j][~np.isnan(E_dims_surrogates[i][:, j])]):]),
                        E_dims_surrogates[i][:, j][~np.isnan(E_dims_surrogates[i][:, j])], deg=1, w=weights[i][:len(
            E_dims_surrogates[i][:, j][~np.isnan(E_dims_surrogates[i][:, j])])]) for i in range(10) for j in range(6)]
    lines_surrogates = np.array(lines_S)
    lines_surrogates = np.reshape(lines_surrogates, (10, 6, 2))

    # Visualize Kaplan E-statistics for Surrogate Data
    fig, axes = plt.subplots(nrows=5, ncols=2, dpi=50)
    for i in range(5):
        for m in range(6):
            axes[i, 0].plot(determinism_test[1][i][2], determinism_test[1][i][1][:, m], label=f"m={m}")
            axes[i, 0].plot(surrogate_scales[i], surrogate_scales[i]*lines_surrogates[i][m, 0] + lines_surrogates[i][m, 1], c="black", ls="--")
            axes[i, 0].legend(loc="lower right")
    for i, j in zip(range(5), [5, 6, 7, 8, 9]):
        for m in range(6):
            axes[i, 1].plot(determinism_test[1][j][2], determinism_test[1][j][1][:, m], label=f"m={m}")
            axes[i, 1].plot(surrogate_scales[j], surrogate_scales[j]*lines_surrogates[j][m, 0] + lines_surrogates[j][m, 1], c="black", ls="--")
            axes[i, 1].legend(loc="lower right")
    plt.suptitle("Kaplan's E-Statistic to Measure Determinism")
    plt.subplots_adjust(left=0.017, right=0.994, top=0.968, bottom=0.02, wspace=0.036, hspace=0.02)
    plt.show()

    # Average over the slopes and intercepts for each dimension
    dim_slopes = np.array([lines_surrogates[i][j, 0] for i in range(len(lines_surrogates)) for j in range(6)])
    dim_mean_slopes = np.mean(np.reshape(dim_slopes, (10, 6)), axis=0)

    dim_intercepts = np.array([lines_surrogates[i][j, 1] for i in range(len(lines_surrogates)) for j in range(6)])
    dim_mean_intercepts = np.mean(np.reshape(dim_intercepts, (10, 6)), axis=0)

    # Original Slope and Intercept minus Mean Surrogate Slope and Intercept
    slopes = lines[:, 0] - dim_mean_slopes
    intercepts = lines[:, 1] - dim_mean_intercepts
    dims = np.arange(1, 7, 1)

    # Visualize Difference between Original Slopes/Intercepts with Average Surrogate Slopes/Intercepts at each Dimension
    fig, axes = plt.subplots(nrows=3, ncols=1, dpi=50, sharex=True)
    axes[0].bar(x=dims, height=lines[:, 0], color="red", label="Original Slopes")
    axes[0].bar(x=dims, height=dim_mean_slopes, color="blue", label="Mean Surrogate Slopes")
    axes[0].legend(loc="best")

    axes[1].bar(x=dims, height=lines[:, 1], color="red", label="Original Intercepts")
    axes[1].bar(x=dims, height=dim_mean_intercepts, color="blue", label="Mean Surrogate Intercepts", alpha=0.5)
    axes[1].legend(loc="best")

    axes[2].plot(dims, slopes, c="seagreen", label="Difference of Slopes")
    axes[2].plot(dims, intercepts, c="orange", label="Difference of Intercepts")
    axes[2].axhline(y=0, ls="--", c="black")
    axes[2].legend(loc="best")
    plt.show()

    print("Straight lines [E(r) A + Br] are fit to the curves, as shown by the dashed lines.")
    print("Completely nondeterministic systems (i.e., white noise) the slope B will be close to "
          "zero, and the intercept A will be close to the value of E at large r.")
    print("Decreasing values of intercept and increasing slope provide increasing")
    print("evidence of deterministic structure. An ideal noise-free low-dimensional")
    print("deterministic system would be expected to produce an intercept of zero,")
    print("at least in the limit of infinite data, and a well defined positive slope.")
    print("\nAlso...")
    print("The surrogate difference for cumulative e intercept A is negative and for slope B is positive at all embedding dimension values.")
    print("This indicates that the original price series is more deterministic than its surrogates.")



