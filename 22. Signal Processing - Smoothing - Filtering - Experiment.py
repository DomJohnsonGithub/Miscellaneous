import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
import talib as ta
import pandas_datareader.data as pdr
import matplotlib.gridspec as gridspec
from datetime import datetime
from pykalman import KalmanFilter
from scipy.signal import lfilter, savgol_filter, wiener
from tsmoothie.smoother import *
import statsmodels.api as sm
import pywt
import seaborn as sns
import warnings
import yfinance as yf

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")


# OUTLIER REMOVAL FUNCTIONS #

def dealing_with_outliers_ma_std(data, rolling_period, threshold, method):
    """
    This method deals with outliers by using a moving average and
    standard deviation to make values outside of the upper and lower
    bounds into nan values and then fills these with an interpolation
    technique.
    """
    # Calculate the rolling moving average and standard deviation
    ma = data.rolling(window=rolling_period).mean()
    std = data.rolling(window=rolling_period).std()

    # Calculate the upper and lower bounds
    upper_bound = ma + (threshold * std)
    lower_bound = ma - (threshold * std)

    # Change values outside the bounds with nan values
    data[data >= upper_bound] = np.nan
    data[data <= lower_bound] = np.nan

    # Replace these nan values with interpolation
    data = data.interpolate(method=method)

    return data


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


# DE-NOISING TECHNIQUES #

def fft_denoiser(data, n_components, to_real=True):
    """
    Fast Fourier Transform:
    - removes noise from the time series to study the underlying signal
    - it moves the time series from the time domain to the frequency domain
      to filter out the frequencies that pollute the data. Then, we just
      have to apply the inverse Fourier transform to get a filtered version
      of our time series.
    """
    n = len(data)

    fft_df = pd.DataFrame()
    PSD = pd.DataFrame()
    _mask = pd.DataFrame()
    cleaned_df = pd.DataFrame(index=data.index)

    for i, j in data.items():
        fft_df[f"{i}"] = np.fft.fft(j, n)
        PSD[f"{i}"] = fft_df[f"{i}"] * np.conj(fft_df[f"{i}"]) / n
        _mask[f"{i}"] = PSD[f"{i}"] > n_components
        fft_df[f"{i}"] *= _mask[f"{i}"]
        cleaned_df[f"{i}"] = np.fft.ifft(fft_df[f"{i}"])

    return cleaned_df


def kalman_filter(data):
    """
    The Kalman Filter is a unsupervised algorithm for
    tracking a single object in a continuous state space.

    The Kalman Filter is essentially a Bayesian Linear
    Regression that can optimally estimate the hidden
    state of a process using its observable variables.
    """
    # Configure the Kalman Filter
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=0,
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=.0001)  # jansen 0.01

    # estimate the hidden state via forward propagation
    f_state_means, f_state_covs = kf.filter(data)

    # estimate the hidden state via backward propagation
    b_state_means, b_state_covs = kf.smooth(data)

    kalman_filter_df = pd.DataFrame(index=data.index)
    kalman_filter_df["K_FILTER"] = f_state_means
    kalman_filter_df["K_SMOOTH"] = b_state_means

    return kalman_filter_df


def iir_filter(data, n):
    """
    The larger n is, the smoother the curve will be.
    """
    n = n
    b = [1.0 / n] * n
    a = 1

    iir_filtered = pd.Series(lfilter(b, a, data), index=data.index)

    return iir_filtered


def savitzky_golay_filter(data, window_length, polyorder):
    """This is a 1D filter."""
    savgol_filtered = pd.Series(savgol_filter(data, window_length=window_length,
                                              polyorder=polyorder), index=data.index)

    return savgol_filtered


def weiner_filter(data, window_length):
    weiner_filtered = pd.Series(wiener(data, mysize=window_length),
                                index=data.index)

    return weiner_filtered


def convolution_smoother(data, window_length, window_type):
    smoother = ConvolutionSmoother(window_len=window_length, window_type=window_type)
    smoother.smooth(data)
    conv_series = pd.Series(smoother.smooth_data[0], index=data.index)

    return conv_series


def spectral_smoothing_with_fourier(data, smooth_fraction, pad_len):
    spectral_series = SpectralSmoother(smooth_fraction=smooth_fraction, pad_len=pad_len)
    spectral_series.smooth(data)
    spectral_series = pd.Series(spectral_series.smooth_data[0], index=data.index)

    return spectral_series


def lowess(data, smooth_fraction):
    lowess = sm.nonparametric.lowess(data, data.index, frac=smooth_fraction)
    lowess = pd.Series(lowess[:, 1], index=data.index)

    return lowess


def wavelet_transformation(data, wavelet):
    """
    Function to perform a SWT/MODWT on the price series.
    """
    coefficients = pywt.swt(data=data, wavelet=wavelet, norm=True)
    coefficients = pd.Series(coefficients[0][0], index=data.index)

    return coefficients


if __name__ == "__main__":
    # Import data #
    symbol = "EURUSD=X"
    df = yf.download(symbol, datetime(2000, 1, 1),
                     datetime.now()).drop(columns=["Adj Close", "Volume"])

    # ----------------------------------------------------------------------------------
    # OUTLIER REMOVAL #

    # Deal with outliers using MA and Standard Deviation
    rolling_period = 20
    threshold = 2
    method = "linear"
    df1 = dealing_with_outliers_ma_std(data=df.copy(), rolling_period=rolling_period,
                                       threshold=threshold, method=method)

    # Deal with outliers using MA, a residual series and then IQR
    lookback = 20
    n = 2
    method = "linear"
    df2 = dealing_with_outliers_ma_res_iqr(data=df.copy(), lookback=lookback,
                                           n=n, method=method)

    # Visualize the Close prices to see which method deals with outliers the best
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, dpi=50, sharex=True)
    ax1.plot(df.Close)
    ax1.set_title("Original series")
    ax2.plot(df1.Close)
    ax2.set_title("Remove outliers using MA and SD")
    ax3.plot(df2.Close)
    ax3.set_title("Remove ma using residual series and IQR")
    plt.show()

    # ----------------------------------------------------------------------------------
    # REMOVING NOISE WITH KALMAN FILTER, FOURIER TRANSFORM, IIR FILTER,
    # SAVITZKY-GOLAY FILTER, WIENER FILTER, CONVOLUTION FILTER,
    # SPECTRAL SMOOTHING WITH FOURIER, LOWESS

    fourier_df1 = fft_denoiser(data=df2, n_components=0.1)
    fourier_df2 = fft_denoiser(data=df2, n_components=0.25)
    fourier_df3 = fft_denoiser(data=df2, n_components=0.5)
    fourier_df4 = fft_denoiser(data=df2, n_components=1)

    plt.plot(df2.Close)
    plt.plot(fourier_df1.Close, label="n_components: 0.1")
    plt.plot(fourier_df2.Close, label="n_components: 0.25")
    plt.plot(fourier_df3.Close, label="n_components: 0.5")
    plt.plot(fourier_df4.Close, label="n_components: 1")
    plt.legend(loc="best")
    plt.suptitle("Fourier Transform - Components")
    plt.show()

    kalman_filter_df = kalman_filter(data=df2.Close)

    df2.Close.plot(c="black", label="Close")
    df2.Close.rolling(100).mean().plot(c="green", label="100 MA")
    kalman_filter_df["K_FILTER"].plot(c="red", label="KF")
    kalman_filter_df["K_SMOOTH"].plot(c="orange", label="KS")
    plt.suptitle("Kalman Filter and Smoother")
    plt.legend()
    plt.show()

    n = 20
    iir_filtered = iir_filter(data=df2.Close, n=n)

    window_length = 21
    polyorder = 3
    savgol_filtered = savitzky_golay_filter(data=df2.Close,
                                            window_length=window_length,
                                            polyorder=polyorder)

    window_length = 20
    weiner_filtered = weiner_filter(data=df2.Close, window_length=window_length)

    window_length = 20
    window_type = "hamming"  # constant, ones, hanning, hamming, bartlett, blackman
    convolution_smoothed = convolution_smoother(data=df2.Close, window_length=window_length,
                                                window_type=window_type)

    smooth_fraction = 0.3
    pad_len = 20
    spectral_smoothed = spectral_smoothing_with_fourier(data=df2.Close, smooth_fraction=smooth_fraction,
                                                        pad_len=pad_len)

    smooth_fraction = 0.01
    lowess = lowess(data=df2.Close, smooth_fraction=smooth_fraction)

    # # Viz the other filters
    fig, axes = plt.subplots(ncols=1, nrows=2, dpi=50)
    axes[0].plot(df2.Close, c="black", label="Close Price")
    axes[0].plot(iir_filtered, c="blue", label="IIR filter")
    axes[0].plot(savgol_filtered, c="green", label="SAVGOL filter")
    axes[0].plot(weiner_filtered, c="red", label="WEINER filter")
    axes[0].set_ylim(np.min(df2.Close) - 0.01, np.max(df2.Close) + 0.01)
    axes[0].legend()

    axes[1].plot(df2.Close, c="black", label="Close Price")
    axes[1].plot(convolution_smoothed, c="yellow", label="CONVOLUTION")
    axes[1].plot(spectral_smoothed, c="orange", lw=0.7, label="SPECTRAL smoothed")
    axes[1].plot(lowess, c="blue", label="LOWESS")
    axes[1].legend()

    plt.suptitle("Smoothing Techniques")
    plt.show()

    # ----------------------------------------------------------------------------------
    # WAVELET TRANSFORM
    # we perform a MODWT as this is most appropriate for real-world applications
    # using haar and db2
    wavelet1 = "haar"
    wavelet2 = "db2"

    wt_1 = wavelet_transformation(data=df2.Close, wavelet=wavelet1)
    wt_2 = wavelet_transformation(data=df2.Close, wavelet=wavelet2)

    # df2.Close.plot(c="black", alpha=0.7)
    # ((wt_1+wt_2)/2).plot(c="red")
    # plt.show()

    # ----------------------------------------------------------------------------------
    # EXPERIMENT WITH INDICATORS THAT ARE SMOOTHED/FILTERERD PRICE SERIES
    df2["EMA"] = ta.EMA(df2.Close, timeperiod=20)

    df2["RSI"] = ta.RSI(df2.Close, timeperiod=14)
    df2["EMA_RSI"] = ta.RSI(df2.EMA, timeperiod=14)
    df2["KFS_RSI"] = ta.RSI(kalman_filter_df["K_SMOOTH"], timeperiod=14)
    df2["KFF_RSI"] = ta.RSI(kalman_filter_df["K_FILTER"], timeperiod=14)
    df2["FOURIER_RSI"] = ta.RSI(fourier_df1["Close"], timeperiod=14)
    df2["WAVELET_RSI"] = ta.RSI((wt_1 + wt_2) / 2, timeperiod=14)

    # Visualize all the indicators - appears the fourier, kalman and ema RSI's are like Shaff Trend Cycle
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(nrows=18, ncols=2, figure=fig)

    ax1 = fig.add_subplot(gs[0:5, :])
    ax1.set_ylim(np.min(df2.Low) - 0.01, np.max(df2.High) + 0.01)
    ax1.plot(df2.Close, c="black", lw=1.)
    ax2 = fig.add_subplot(gs[5:7, :], sharex=ax1)
    ax2.plot(df2.RSI, c="magenta", label="RSI", lw=0.7)
    ax2.legend(loc="upper left")
    ax3 = fig.add_subplot(gs[7:9, :], sharex=ax2)
    ax3.plot(df2.EMA_RSI, c="magenta", label="EMA_RSI", lw=0.7)
    ax3.legend(loc="upper left")
    ax4 = fig.add_subplot(gs[9:11, :], sharex=ax3)
    ax4.plot(df2.KFS_RSI, c="magenta", label="KFS_RSI", lw=0.7)
    ax4.legend(loc="upper left")
    ax5 = fig.add_subplot(gs[11:13, :], sharex=ax4)
    ax5.plot(df2.KFF_RSI, c="magenta", label="KFF_RSI", lw=0.7)
    ax5.legend(loc="upper left")
    ax6 = fig.add_subplot(gs[13:15, :], sharex=ax5)
    ax6.plot(df2.FOURIER_RSI, c="magenta", label="FOURIER_RSI", lw=0.7)
    ax6.legend(loc="upper left")
    ax7 = fig.add_subplot(gs[15:17, :], sharex=ax6)
    ax7.plot(df2.WAVELET_RSI, c="magenta", label="WAVELET_RSI", lw=0.7)
    ax7.legend(loc="upper left")

    plt.suptitle("Experiment - Technical Indicator after Smoothing")
    plt.subplots_adjust(left=0.021, bottom=0.0, right=0.998, top=0.993, hspace=0.029)
    multi = MultiCursor(fig.canvas, fig.axes, useblit=True, horizOn=True, vertOn=True, color="royalblue", lw=0.5)
    plt.show()

    # Appears to be some divergence between RSI and the wavelet_RSI
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(nrows=12, ncols=2, figure=fig)

    ax1 = fig.add_subplot(gs[0:5, :])
    ax1.set_ylim(np.min(df2.Low) - 0.01, np.max(df2.High) + 0.01)
    ax1.plot(df2.Close, c="black", lw=1.)
    ax2 = fig.add_subplot(gs[5:7, :], sharex=ax1)
    ax2.plot(df2.RSI, c="darkblue", label="RSI", lw=0.7)
    ax2.legend(loc="upper left")
    ax3 = fig.add_subplot(gs[7:9, :], sharex=ax2)
    ax3.plot(df2.WAVELET_RSI, c="darkblue", label="WAVELET_RSI", lw=0.7)
    ax3.legend(loc="upper left")
    ax4 = fig.add_subplot(gs[9:11, :], sharex=ax3)
    ax4.plot(df2.EMA_RSI, c="red", label="EMA_RSI", lw=0.7)
    ax4.legend(loc="upper left")

    plt.suptitle("Experiment - Divergence between RSI and Wavelet RSI")
    plt.subplots_adjust(left=0.021, bottom=0.0, right=0.998, top=0.993, hspace=0.029)
    multi = MultiCursor(fig.canvas, fig.axes, useblit=True, horizOn=True, vertOn=True, color="royalblue", lw=0.5)
    plt.show()
