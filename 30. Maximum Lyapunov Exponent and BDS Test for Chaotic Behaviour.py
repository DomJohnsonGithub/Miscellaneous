from datetime import datetime
from kneed import KneeLocator
from itertools import chain
from nolds import corr_dim, lyap_r, lyap_e
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import MultiCursor
import pandas as pd
import pywt
import pandas_datareader.data as pdr
from scipy.linalg import hankel
from scipy.signal import argrelextrema, argrelmin, welch
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, mean_absolute_percentage_error)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from spectrum import Periodogram, parma
from statsmodels.tsa.api import bds
import talib as ta
from tqdm import tqdm
from gtda.time_series import SingleTakensEmbedding
from nolds import sampen
import yfinance as yf
import warnings

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


def plot_price(data):
    fig = plt.figure(dpi=50)
    plt.plot(data, c="black")
    plt.tight_layout()
    plt.show()


def create_lags_from_price(df, number_lags):
    for i in tqdm(range(1, number_lags)):
        df[f"Lag{i}"] = df.Close.shift(-i)

    return df


def rice_criterion_bins(df):
    rice_criterion = int(np.round(2 * (len(df)) ** (1 / 3), 0))

    return rice_criterion


def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def mutual_information(df, number_lags, bins):
    mutual_information = []
    for i in tqdm(range(1, number_lags)):
        mutual_information.append(calc_MI(df.Close, df[f"Lag{i}"], bins=bins))

    return np.array(mutual_information)


def optimal_time_delay(mutual_information):
    first_minima = argrelmin(mutual_information)[0][0]
    print("\nOptimal Time Delay:", first_minima)

    return first_minima


def plot_AMI(mutual_information, number_lags, x):
    plt.bar(np.arange(1, number_lags), mutual_information, color="red", edgecolor="white")
    plt.axvline(x=x, c="green", lw=1)
    plt.title("Average Mutual Information")
    plt.xlabel("Lag / Time Delay (tau)")
    plt.ylabel("AMI")
    plt.show()


def visualize_phase_space(data):
    fig = plt.figure(dpi=50)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    plt.suptitle("Reconstructed Phase Space")
    plt.tight_layout()
    plt.show()


def Lyapunov_Exponents(data, emb_dim, lags):
    LE = []
    for i in range(1, lags):
        LE.append(lyap_r(data, emb_dim=emb_dim, lag=i))

    fig = plt.figure(dpi=50)
    plt.plot(np.arange(1, lags), np.array(LE), c="black")
    plt.xlabel("Separation Time - Time Delays")
    plt.ylabel("LLE")
    plt.title("Lyapunov Exponents")
    plt.tight_layout()
    plt.show()


def Maximal_Lyapunov_Exponent(data, embedding_dimension, time_delay):
    LLE = lyap_r(data, emb_dim=embedding_dimension, lag=time_delay)

    if LLE > 0:
        print("\nMaximum Largest Lyapunov Exponent is", np.round(LLE, 4),
              "; Phase Space exhibits properties of Chaotic Behaviour")
    else:
        print("\nMaximum Largest Lyapunov Exponent is", np.round(LLE, 4),
              "; Phase Space does not exhibit Chaotic Behaviour")


def power_spectral_density(data):
    fig, axes = plt.subplots(nrows=2, ncols=1, dpi=50)
    p = Periodogram(data)
    p.plot(ax=axes[0])

    axes[1].psd(data)

    plt.suptitle("Periodogram's -  Fourier PSD vs Welch PSD")
    plt.tight_layout()
    plt.show()


def lowpassfilter_WD(signal, thresh=0.63, wavelet="db1", mode="per"):
    thresh = thresh * np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet=wavelet, mode=mode)
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="hard") for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet=wavelet, mode=mode)

    return reconstructed_signal[:]


def compare_signals(data1, data2):
    fig = plt.figure(dpi=50)
    plt.plot(data1, c="black")
    plt.plot(data2, c="red")
    plt.suptitle("Original Signal vs Wavelet Transformed Signal")
    plt.tight_layout()
    plt.show()


def two_dimensional_phase_space_viz(signal1, signal2):
    sc = StandardScaler()
    pca = PCA(n_components=2)
    takens1_sc = sc.fit_transform(signal1)
    takens2_sc = sc.fit_transform(signal2)
    takens1_pca = pca.fit_transform(takens1_sc)
    takens2_pca = pca.fit_transform(takens2_sc)

    fig, axes = plt.subplots(nrows=1, ncols=2, dpi=50)
    axes[0].scatter(takens1_pca[:, 0], takens1_pca[:, 1], c="black")
    axes[0].set_title("Before Noise Reduction")
    axes[1].scatter(takens2_pca[:, 0], takens2_pca[:, 1], c="black")
    axes[1].set_title("After Noise Reduction")
    plt.suptitle("2-Dimensional Projection of the Reconstructed Phase Space")
    plt.tight_layout()
    plt.show()


def create_logreturns(df):
    df["log_returns"] = np.log(1 + df.Close.pct_change())

    return df


def bds_test_chaotic_behaviour(data):
    x = np.log(data / data.shift(1))
    x = x.dropna()

    results = []
    results1 = []
    for i in np.arange(0.25, 2.25, 0.25):
        results.append(bds(np.array(x), max_dim=5, distance=x.std() * i))
    for i in np.arange(0.1, 1.1, 0.1):
        results1.append(bds(np.array(x), max_dim=5, distance=(x.max() - x.min()) * i))

    arrays = [
        np.array(["0.25σ", "0.25σ", "0.5σ", "0.5σ", "0.75σ", "0.75σ", "1σ", "1σ", "1.25σ", "1.25σ", "1.5σ", "1.5σ",
                  "1.75σ", "1.75σ", "2σ", "2σ"]),
        np.array(["bsd", "pval", "bsd", "pval", "bsd", "pval", "bsd", "pval",
                  "bsd", "pval", "bsd", "pval", "bsd", "pval", "bsd", "pval"])
    ]

    bsd_statistics = pd.DataFrame(np.round(np.array(results).reshape(16, 4), 5), index=arrays, columns=[2, 3, 4, 5])
    print(bsd_statistics)

    arrays = [
        np.array(
            ["0.1r", "0.1r", "0.2r", "0.2r", "0.3r", "0.3r", "0.4r", "0.4r", "0.5r", "0.5r", "0.6r", "0.6r", "0.7r",
             "0.7r", "0.8r", "0.8r",
             "0.9r", "0.9r", "1r", "1r"]),
        np.array(["bsd", "pval", "bsd", "pval", "bsd", "pval", "bsd", "pval", "bsd", "pval", "bsd", "pval", "bsd",
                  "pval", "bsd", "pval",
                  "bsd", "pval", "bsd", "pval"])
    ]
    bsd_statistics1 = pd.DataFrame(np.round(np.array(results1).reshape(20, 4), 5), index=arrays,
                                   columns=[2, 3, 4, 5])
    print(bsd_statistics1)


if __name__ == "__main__":
    # Fetch OHLC Data
    symbol = "EURUSD=X"  # ticker
    from_date = datetime(2000, 1, 1)
    to_date = datetime.now()
    drop_columns = ["Adj Close", "Volume"]

    df = fetch_data(symbol=symbol, from_date=from_date,
                    to_date=to_date, cols_to_drop=drop_columns)

    # Outlier Treatment #
    df = outlier_treatment(data=df, lookback=10, n=2, method="linear")
    print(df)

    # Visualize the Close Price of our Dataset
    # plot_price(data=df.Close)

    # Create Lagged Version of the Close Price
    number_of_lags = 201
    df = create_lags_from_price(df, number_of_lags)

    # Average Mutual Information - Finding the Optimal Time Delay #
    data = df.copy()  # copy the dataframe for use here
    data.dropna(inplace=True)  # remove NaN values

    # Rice Criterion for selecting number of bins in Probability Density Function (Histogram)
    rice_criterion = rice_criterion_bins(data)

    # Calculate the Mutual Information Score between Close Price and Lagged Versions of itself
    MI = mutual_information(data, number_lags=number_of_lags, bins=rice_criterion)

    # First Local Minima is our chosen Optimal Time Delay for Takens Embedding Theorem
    # for Phase Space Reconstruction
    first_minima = optimal_time_delay(MI)

    # Visualize the Average Mutual Information
    plot_AMI(MI, number_lags=number_of_lags, x=first_minima)

    # Use Takens Embedding Theorem - Create an n-dimensional time delayed embedded coordinate vector
    te = SingleTakensEmbedding(parameters_type='search', n_jobs=11, time_delay=int(first_minima), dimension=3)
    takens = te.fit_transform(df.Close)

    # Visualize the Reconstructed Phase Space - Attractor #
    visualize_phase_space(takens)

    # Determining whether our system exhibits Chaotic Behaviour #
    # Use the Takens Embedding Theorem to examine the Lyapunov Exponents - want > 0 for deterministic chaos
    # Lyapunov_Exponents(data=df.Close, emb_dim=3, lags=number_of_lags)

    # Largest Maximal Lyapunov Exponent
    Maximal_Lyapunov_Exponent(data=df.Close, embedding_dimension=3, time_delay=first_minima)

    # Fourier Power Spectrum - Before Wavelet Transformation
    power_spectral_density(data=df.Close)  # look for broadband noise

    returns = (df.Close
               .copy()
               .pct_change()
               .dropna())  # de-trend the data so we have zero mean
    power_spectral_density(data=returns)  # look for broadband noise

    # ------------------------------------------------------------------------------------------------------------------
    # # De-noise the original data with Wavelet Transform to compare our Phase Spaces and PSD #
    # denoised_signal = lowpassfilter_WD(df.Close, thresh=0.05, wavelet="db5", mode="smooth")  # noise reduction
    # df["WT_Close"] = denoised_signal  # add signal back into dataframe
    #
    # # Visualize the Signals
    # compare_signals(data1=df.Close, data2=df.WT_Close)
    #
    #
    # # Determine Embedding Dimension and Time Delay for Clean Data #
    # # Dataframe Manipulations
    # data1 = df["WT_Close"].copy()  # copy the dataframe for use here
    # data1 = pd.DataFrame(data1)
    # data1.columns = ["Close"]
    # data1 = create_lags_from_price(data1, number_lags=number_of_lags)  # produce lags of denoised close price
    # data1.dropna(inplace=True)  # remove NaN values
    #
    # # Time Delay and Embedding Dimension
    # rice_criterion = rice_criterion_bins(data1)
    # mutual_information = mutual_information(df=data1,
    #                                         number_lags=number_of_lags,
    #                                         bins=rice_criterion)  # average mutual information
    # first_minima1 = optimal_time_delay(mutual_information)  # first local minima for time delay
    # plot_AMI(mutual_information, number_lags=number_of_lags, x=first_minima1)
    #
    # # Taken Matrix
    # te = SingleTakensEmbedding(parameters_type='search', n_jobs=11, dimension=3, time_delay=first_minima1)
    # takens2 = te.fit_transform(df.WT_Close)  # time-delay embedding theorem
    #
    # # Compare the Phase Space Reconstructions #
    # two_dimensional_phase_space_viz(signal1=takens1, signal2=takens2)

    # Sample Entropy vs Correlation Dimension
    sample_entropy = [sampen(df.Close, emb_dim=i) for i in range(1, 21)]
    correlation_dimension = [corr_dim(df.Close, emb_dim=i) for i in range(1, 21)]

    plt.scatter(sample_entropy, correlation_dimension, c="red")
    plt.show()

    # BDS Test
    bds_test_chaotic_behaviour(data.Close)  # the BDS Test creates he differenced series

