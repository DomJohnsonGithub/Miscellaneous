from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema, argrelmin, welch
import seaborn as sns
import talib as ta
from tqdm import tqdm
from gtda.time_series import SingleTakensEmbedding
import warnings
import yfinance as yf

from pyrqa.time_series import EmbeddedSeries, TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius, RadiusCorridor, Unthresholded
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.computation import RPComputation
from pyrqa.image_generator import ImageGenerator
from pyts.image import RecurrencePlot

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


if __name__ == "__main__":

    # Fetch OHLC Data
    symbol = "EURUSD=X"  # ticker
    from_date = datetime(2000, 1, 1)
    to_date = datetime.now()
    drop_columns = ["Adj Close", "Volume"]

    df = fetch_data(symbol=symbol, from_date=from_date,
                    to_date=to_date, cols_to_drop=drop_columns)

    # Deal with outliers
    df = outlier_treatment(df, lookback=10, n=2, method="linear")

    # Create Taken's Embedding Matrix
    te = SingleTakensEmbedding(parameters_type='search', n_jobs=11)
    takens = te.fit_transform(df.Close)

    # Recurrence Plots - display of the spatial correlation in an attractor in terms of time #
    """It gives insight into whether the data is periodic, deterministic or random. 
    When the graph displays isolated recurrent points the data is random and when a 
    repeating pattern occurs the data is periodic. When the graph displays diagonal 
    line segments the data is deterministic. The graph also gives insight into whether 
    the data originated from a stationary or non stationary process. Non-stationarity 
    is shown on the graph by the decreasing density away from the main diagonal line 
    segment."""

    embedded_series = EmbeddedSeries(takens, dtype=np.float64)
    radius = [0.05, 0.075, 0.1, 0.15]
    for i in radius:
        settings = Settings(embedded_series,
                            analysis_type=Classic,
                            neighbourhood=FixedRadius(i),
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=1)

        computation = RQAComputation.create(settings, verbose=True)
        result = computation.run()
        result.min_diagonal_line_length = 2
        result.min_vertical_line_length = 2
        result.min_white_vertical_line_length = 2
        print(result)

        computation = RPComputation.create(settings)
        result = computation.run()
        ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse, f"recurrence_plot{i}.png")

    rp = RecurrencePlot(threshold=0.05)
    taken_rp = rp.transform(takens.T)

    fig = plt.figure(dpi=50)
    plt.imshow(taken_rp[0], origin="lower", cmap="gist_yarg")
    plt.tight_layout()
    plt.show()
