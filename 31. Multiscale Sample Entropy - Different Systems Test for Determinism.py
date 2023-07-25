from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as pdr
import pywt
import seaborn as sns
import talib as ta
from tqdm import tqdm
import warnings
from itertools import accumulate
from fbm import fbm
from colorednoise import powerlaw_psd_gaussian
from gtda.time_series import SingleTakensEmbedding
from nolds.measures import sampen
import yfinance as yf

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


def lowpassfilter_WD(signal, thresh=0.63, wavelet="db1", mode="per"):
    thresh = thresh * np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet=wavelet, mode=mode)
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="hard") for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet=wavelet, mode=mode)

    return reconstructed_signal[1:]


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


def rossler_system(a, b, c, t, tf, h):
    def derivative(r, t):
        x = r[0]
        y = r[1]
        z = r[2]
        return np.array([- y - z, x + a * y, b + z * (x - c)])

    time = np.array([])
    x = np.array([])
    y = np.array([])
    z = np.array([])
    r = np.array([0.1, 0.1, 0.1])

    while (t <= tf):
        time = np.append(time, t)
        z = np.append(z, r[2])
        y = np.append(y, r[1])
        x = np.append(x, r[0])

        k1 = h * derivative(r, t)
        k2 = h * derivative(r + k1 / 2, t + h / 2)
        k3 = h * derivative(r + k2 / 2, t + h / 2)
        k4 = h * derivative(r + k3, t + h)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6

        t = t + h

    return x, y, z


def logistic_equation_orbit(seed, r, n_iter, n_skip=0):
    X_t = []
    T = []
    t = 0
    x = seed
    logistic = lambda r, x: r*x*(1-x)
    # Iterate the logistic equation, printing only if n_skip steps have been skipped
    for i in range(n_iter + n_skip):
        if i >= n_skip:
            X_t.append(x)
            T.append(t)
            t += 1
        x = logistic(r, x)

    return X_t


def henon_attractor(x, y, a=1.4, b=0.3):
    '''Computes the next step in the Henon
    map for arguments x, y with kwargs a and
    b as constants.
    '''
    x_next = 1 - a * x ** 2 + y
    y_next = b * x
    return x_next, y_next


def quadratic_map(seed, p, n_iter, n_skip=0):
    X_t = []
    T = []
    t = 0
    x = seed
    quad = lambda p, x: p - x**2
    # Iterate the logistic equation, printing only if n_skip steps have been skipped
    for i in range(n_iter + n_skip):
        if i >= n_skip:
            X_t.append(x)
            T.append(t)
            t += 1
        x = quad(p, x)

    return X_t


def ikeda_map(seed, u, n_iter, n_skip=0):
    X_t = []
    Y_t = []
    T = []
    t = 0
    x = seed
    y = seed
    ikeda_x = lambda u, x, y: 1 + u*(x*np.cos(0.4 - 6.0/(1. + x**2 + y**2)) - y*np.sin(0.4 - 6.0/(1. + x**2 + y**2)))
    ikeda_y = lambda u, x, y: u*(x*np.sin(0.4 - 6.0/(1. + x**2 + y**2)) + y*np.cos(0.4 - 6.0/(1. + x**2 + y**2)))
    # Iterate the logistic equation, printing only if n_skip steps have been skipped
    for i in range(n_iter + n_skip):
        if i >= n_skip:
            X_t.append(x)
            Y_t.append(y)
            T.append(t)
            t += 1
        x = ikeda_x(u, x, y)
        y = ikeda_y(u, x, y)

    return X_t, Y_t


def coarse_grain_timeseries(one_dimension_ts, max_scale_factor):
    scale_factor = np.arange(1, max_scale_factor+1)
    j_length = np.array([int(np.round(len(one_dimension_ts) / scale, 0)) for scale in scale_factor])

    data = [(1/scale) * np.sum(one_dimension_ts[(j - 1)*scale:(j*scale)]) for scale, length in zip(scale_factor, j_length) for j in range(1, length+1)]
    data = [data[end - length:end] for length, end in zip(j_length, accumulate(j_length))]
    return data


def multiscale_entropy(coarse_grained_data, embedding_dimension, tolerance):
    sample_entropy = np.array([sampen(i, emb_dim=embedding_dimension, tolerance=tolerance) for i in tqdm(coarse_grained_data)])
    return sample_entropy


if __name__ == "__main__":
    # Fetch OHLC Data
    symbol = "EURUSD=X"  # ticker
    from_date = datetime(2000, 1, 1)
    to_date = datetime.now()
    drop_columns = ["Adj Close", "Volume"]

    df = fetch_data(symbol=symbol, from_date=from_date,
                    to_date=to_date, cols_to_drop=drop_columns)

    # Treat Outliers #
    df = outlier_treatment(df, lookback=10, n=2, method="linear")

    # Generate various time-series #

    # Rossler X1
    a = 0.2
    b = 0.2
    c = 5.7
    t = 0
    tf = 100
    h = 0.005

    x1, y, z = rossler_system(a, b, c, t, tf, h)
    x1 = np.array(x1).reshape(-1)

    # Logistic Map
    X_t = logistic_equation_orbit(0.6, r=3.9, n_iter=len(x1))

    # Henon Map
    steps = len(x1)
    X = np.zeros(steps + 1)
    Y = np.zeros(steps + 1)

    X[0], Y[0] = 0, 0
    for i in range(steps):
        x_next, y_next = henon_attractor(X[i], Y[i])
        X[i + 1] = x_next
        Y[i + 1] = y_next

    X_henon = X[:-1]

    # Ikeda Map
    X_ikeda, Y_ikeda = ikeda_map(seed=0.6, u=0.918, n_iter=len(x1))

    # Quadratic Map
    p = 1.7904
    quad_Xt = quadratic_map(seed=0.1, p=p, n_iter=len(x1))

    # White Noise
    wn = np.random.normal(0, 1, len(x1))

    # 1/f / Pink Noise
    pn = powerlaw_psd_gaussian(exponent=1, size=len(x1))

    # Fractional Brownian Noise
    fbmn = fbm(n=len(x1)-1, hurst=0.7)

    # Sine Wave
    sine = np.array([np.sin(i) for i in range(len(x1))])

    # Coarse Grain the Time-Series #
    rossler_coarse_grained_x1 = coarse_grain_timeseries(x1, max_scale_factor=20)
    l_map_coarse_grained = coarse_grain_timeseries(X_t, max_scale_factor=20)
    henon_map_coarse_grained_x = coarse_grain_timeseries(X_henon, max_scale_factor=20)
    quad_map_coarse_grained = coarse_grain_timeseries(quad_Xt, max_scale_factor=20)
    wn_cg = coarse_grain_timeseries(wn, max_scale_factor=20)
    ikeda_map_cg = coarse_grain_timeseries(Y_ikeda, max_scale_factor=20)
    pn_cg = coarse_grain_timeseries(pn, max_scale_factor=20)
    close_price_cg = coarse_grain_timeseries(one_dimension_ts=np.array(df.Close), max_scale_factor=20)
    fbmn_cg = coarse_grain_timeseries(fbmn, max_scale_factor=20)
    sine_cg = coarse_grain_timeseries(sine, max_scale_factor=20)

    # ------------------------------------------------------------------------------------------------------------------
    # Perform Multiscale Sample Entropy #
    rossler_mse_x1 = multiscale_entropy(rossler_coarse_grained_x1, embedding_dimension=2, tolerance=0.15)
    l_map_mse = multiscale_entropy(l_map_coarse_grained, embedding_dimension=2, tolerance=0.15)
    henon_map_mse_x = multiscale_entropy(henon_map_coarse_grained_x, embedding_dimension=2, tolerance=0.15)
    quad_map_mse = multiscale_entropy(quad_map_coarse_grained, embedding_dimension=2, tolerance=0.15)
    wn_mse = multiscale_entropy(wn_cg, embedding_dimension=2, tolerance=0.15)
    ikeda_map_mse = multiscale_entropy(ikeda_map_cg, embedding_dimension=2, tolerance=0.15)
    pn_mse = multiscale_entropy(pn_cg, embedding_dimension=2, tolerance=0.15)
    close_price_mse = multiscale_entropy(close_price_cg, embedding_dimension=2, tolerance=0.15)
    fbmn_mse = multiscale_entropy(fbmn_cg, embedding_dimension=2, tolerance=0.15)
    sine_mse = multiscale_entropy(sine_cg, embedding_dimension=2, tolerance=0.15)

    # ------------------------------------------------------------------------------------------------------------------
    # Visualization of MSE #
    fig = plt.figure(dpi=50)
    plt.plot(np.arange(1, 21), rossler_mse_x1, c="blue", label="Rossler X")
    plt.plot(np.arange(1, 21), l_map_mse, c="orange", label="Logistic Map")
    plt.plot(np.arange(1, 21), henon_map_mse_x, c="pink", label="Henon Map X")
    plt.plot(np.arange(1, 21), quad_map_mse, c="black", label="Quadratic Map")
    plt.plot(np.arange(1, 21), wn_mse, c="yellow", label="White Noise")
    plt.plot(np.arange(1, 21), ikeda_map_mse, c="red", label="Ikeda Map")
    plt.plot(np.arange(1, 21), pn_mse, c="purple", label="Pink Noise")
    plt.plot(np.arange(1, 21), close_price_mse, c="green", label="Close Price")
    plt.plot(np.arange(1, 21), fbmn_mse, c="cyan", label="Fractional Brownian Noise")
    plt.plot(np.arange(1, 21), sine_mse, c="deepskyblue", label="Sine Wave")
    plt.legend(loc="upper right")
    plt.xticks(np.arange(1, 21))
    plt.show()