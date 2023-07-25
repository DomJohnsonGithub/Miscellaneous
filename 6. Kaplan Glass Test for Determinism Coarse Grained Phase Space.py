from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import talib as ta
from tqdm import tqdm
from gtda.time_series import SingleTakensEmbedding
import warnings
import yfinance as yf
from scipy.signal import savgol_filter
from sympy import symbols, Eq
from sympy.solvers import solve
from itertools import chain
import collections
from sklearn.manifold import SpectralEmbedding
from scipy.signal import argrelmin

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

    return reconstructed_signal


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


def get_box_boundaries(embedded_series, num_boxes):
    x_min, y_min = np.min(embedded_series, axis=0)
    x_max, y_max = np.max(embedded_series, axis=0)

    # Calculate box size
    box_size_x = (x_max - x_min) / np.sqrt(num_boxes)
    box_size_y = (y_max - y_min) / np.sqrt(num_boxes)

    # Calculate adjusted box boundaries
    adjusted_x_min = x_min - box_size_x
    adjusted_x_max = x_max + box_size_x
    adjusted_y_min = y_min - box_size_y
    adjusted_y_max = y_max + box_size_y

    # Create box boundaries
    box_boundaries_x = np.linspace(adjusted_x_min, adjusted_x_max, int(np.sqrt(num_boxes)) + 1)
    box_boundaries_y = np.linspace(adjusted_y_min, adjusted_y_max, int(np.sqrt(num_boxes)) + 1)

    return box_boundaries_x, box_boundaries_y


def visualize_boxes(embedded_series, box_boundaries_x, box_boundaries_y):
    # Create a scatter plot of the embedded series
    plt.scatter(embedded_series[:, 0], embedded_series[:, 1], color='blue', label='Embedded Series', s=5)

    # Plot the box boundaries
    for i in range(len(box_boundaries_x) - 1):
        for j in range(len(box_boundaries_y) - 1):
            x = [box_boundaries_x[i], box_boundaries_x[i + 1], box_boundaries_x[i + 1], box_boundaries_x[i],
                 box_boundaries_x[i]]
            y = [box_boundaries_y[j], box_boundaries_y[j], box_boundaries_y[j + 1], box_boundaries_y[j + 1],
                 box_boundaries_y[j]]
            plt.plot(x, y, color='red', alpha=0.5)

    plt.title('Embedded Series with Box Boundaries')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.show()


def determinism_value(embedding_matrix, box_boundaries_x, box_boundaries_y):
    trajectory_vectors = {}

    for i in range(len(box_boundaries_x) - 1):
        for j in range(len(box_boundaries_y) - 1):
            box_name = f"({i},{j})"
            entry_exit_vectors = []
            entry_point = None
            for k in range(len(embedding_matrix) - 1):
                x, y = embedding_matrix[k]
                x_next, y_next = embedding_matrix[k + 1]

                if box_boundaries_x[i] <= x <= box_boundaries_x[i + 1] and \
                        box_boundaries_y[j] <= y <= box_boundaries_y[j + 1]:
                    if entry_point is None:
                        entry_point = (x, y)
                elif entry_point is not None:
                    exit_point = (x, y)

                    print(entry_point)
                    print(exit_point)

                    dx = exit_point[0] - entry_point[0]
                    dy = exit_point[1] - entry_point[1]
                    trajectory_vector = [dx, dy]
                    entry_exit_vectors.append(trajectory_vector)
                    entry_point = None

            if entry_exit_vectors:
                norm_entry_exit_vectors = []
                for vector in entry_exit_vectors:
                    norm = (vector[0] ** 2 + vector[1] ** 2) ** 0.5
                    normalized_vector = [vector[0] / norm, vector[1] / norm]
                    norm_entry_exit_vectors.append(normalized_vector)

                trajectory_vectors[box_name] = norm_entry_exit_vectors

    # Print the resulting trajectory vectors for each box
    for box, vectors in trajectory_vectors.items():
        print(f"Box {box}:")
        for vector in vectors:
            print(vector)
        print()

    # Create a dictionary to store the averaged trajectory vectors for each box
    averaged_vectors = {}

    # Iterate over each box and its trajectory vectors
    for box, vectors in trajectory_vectors.items():
        # Initialize variables to store the sum of vectors
        sum_vector = [0, 0]

        # Iterate over each vector in the box
        for vector in vectors:
            # Add the vector components to the sum_vector
            sum_vector[0] += vector[0]
            sum_vector[1] += vector[1]

        # Calculate the average vector components
        avg_vector = [sum_vector[0] / len(vectors), sum_vector[1] / len(vectors)]

        # Store the averaged vector for the box
        averaged_vectors[box] = avg_vector

    # Print the resulting averaged trajectory vectors for each box
    print("-------------")
    for box, avg_vector in averaged_vectors.items():
        print(f"Box {box}:")
        print(avg_vector)
        print()

    # Singular Values of each box
    Vj = [(avg_vector[0] ** 2 + avg_vector[1] ** 2) ** 0.5 for box, avg_vector in averaged_vectors.items()]

    return np.mean(Vj)


if __name__ == "__main__":
    # Fetch OHLC Data
    symbol = "EURUSD=X"
    from_date = datetime(2000, 1, 1)
    to_date = datetime.now()
    drop_columns = ["Adj Close", "Volume"]

    df = fetch_data(symbol=symbol, from_date=from_date,
                    to_date=to_date, cols_to_drop=drop_columns)

    # Treat Outliers
    df = outlier_treatment(df, lookback=10, n=2, method="linear")

    # denoised time-series for experimentation
    close_price = lowpassfilter_WD(df.Close, thresh=0.05, wavelet="db5", mode="smooth")
    close_price1 = savgol_filter(close_price, window_length=101, polyorder=4, mode="mirror")

    # tau is our time delay
    ami, tau = ami_optimal_time_delay(close_price, lags=200)

    # Taken's Embedded Matrix Transform
    te = SingleTakensEmbedding(parameters_type="search", n_jobs=11, dimension=3, time_delay=int(tau))
    taken_matrix = te.fit_transform(close_price)

    # Making a 2D Phase Space
    sc = StandardScaler()

    # Apply Laplacian EigenMap to get the Reconstructed Phase Space in 2D
    taken_matrix = SpectralEmbedding(n_components=2, affinity="rbf", n_jobs=11).fit_transform(taken_matrix)

    # PCA to get in 2 dimensions
    # taken_matrix = PCA(n_components=2).fit_transform(taken_matrix)

    # Example usage
    embedding_matrix = taken_matrix
    num_boxes = 16 * 16

    box_boundaries_x, box_boundaries_y = get_box_boundaries(embedding_matrix, num_boxes=num_boxes)

    V = determinism_value(embedding_matrix, box_boundaries_x, box_boundaries_y)
    print(V)

    visualize_boxes(embedding_matrix, box_boundaries_x, box_boundaries_y)







