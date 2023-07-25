import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib as ta
from datetime import datetime
import seaborn as sns
import warnings
import yfinance as yf
from gtda.time_series import SingleTakensEmbedding
from pykalman import KalmanFilter
from nolds import corr_dim
from sklearn.manifold import SpectralEmbedding, LocallyLinearEmbedding, MDS, Isomap
from sklearn.decomposition import PCA, KernelPCA
from mpl_toolkits.mplot3d import Axes3D

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


def hull_moving_average(data, window_size):
    return ta.EMA(2*ta.EMA(data, int(window_size/2)) - ta.EMA(data, window_size),
                  int(np.sqrt(window_size)))


def kalman_filter(data, transition_covariance=0.01):
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=data[0],
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=transition_covariance)

    return kf.filter(data)[0]


def plot_3D(embedded_matrix, dim_red_technique, name):
    data = dim_red_technique.fit_transform(embedded_matrix)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{name} Attractor')
    plt.suptitle(f"{dim_red_technique}")
    plt.show()


if __name__ == "__main__":
    # Import data
    symbol = "AUDJPY=X"
    df = yf.download(symbol, datetime(2000, 1, 1),
                        datetime.now()).drop(columns=["Adj Close", "Volume"])

    # Handle Outliers
    df = dealing_with_outliers_ma_res_iqr(data=df, lookback=10, n=2, method="linear")

    # Create two more smoothed versions of close price
    close = df.Close
    hull_smoothed_close_10 = hull_moving_average(close, window_size=10)
    hull_smoothed_close_20 = hull_moving_average(close, window_size=20)
    kf_close_01 = pd.Series(np.squeeze(kalman_filter(close, transition_covariance=0.01)), index=close.index)
    kf_close_001 = pd.Series(np.squeeze(kalman_filter(close, transition_covariance=0.001)), index=close.index)

    # Visualize
    plt.plot(close, c="black")
    plt.plot(hull_smoothed_close_10, c="blue")
    plt.plot(hull_smoothed_close_20, c="purple")
    plt.plot(pd.Series(np.squeeze(kf_close_01), index=close.index), c="orange")
    plt.plot(pd.Series(np.squeeze(kf_close_001), index=close.index), c="red")
    plt.show()

    # Dataframe
    data = pd.DataFrame({
        'close': close,
        'HMA(10)': hull_smoothed_close_10,
        'HMA(20)': hull_smoothed_close_20,
        'kf_01': kf_close_01,
        'kf_001': kf_close_001
    }, index=close.index).dropna()
    print(data)

    # Taken's Embedding Matrix to form Strange Attractor
    taken = SingleTakensEmbedding(parameters_type='search', n_jobs=11)

    em_A = taken.fit_transform(data.close)
    em_B = taken.fit_transform(data["HMA(10)"])
    em_C = taken.fit_transform(data["HMA(20)"])
    em_D = taken.fit_transform(data["kf_01"])
    em_E = taken.fit_transform(data["kf_001"])
    attractors = [em_A, em_B, em_C, em_D, em_E]

    # Correlation Dimension
    print("Close Attractor - corr dimension: ", corr_dim(data.close, emb_dim=np.shape(em_A)[1]))
    print("HMA(10) Attractor - corr dimension: ", corr_dim(data["HMA(10)"], emb_dim=np.shape(em_B)[1]))
    print("HMA(20) Attractor - corr dimension: ", corr_dim(data["HMA(20)"], emb_dim=np.shape(em_C)[1]))
    print("KF_01 Attractor - corr dimension: ", corr_dim(data["kf_01"], emb_dim=np.shape(em_D)[1]))
    print("KF_001 Attractor - corr dimension: ", corr_dim(data["kf_001"], emb_dim=np.shape(em_E)[1]))

    # Dimensionality Reduction Techniques
    drs = [PCA(n_components=3), KernelPCA(n_components=3, kernel="rbf", n_jobs=11),
           SpectralEmbedding(n_components=3, n_jobs=11), LocallyLinearEmbedding(n_components=3, n_jobs=11),
           LocallyLinearEmbedding(n_components=3, n_jobs=11), MDS(n_components=3, n_jobs=11),
           Isomap(n_components=3, n_jobs=11)]

    # Visualize the attractors
    plot_3D(attractors[2], dim_red_technique=drs[1], name="KF_01")






