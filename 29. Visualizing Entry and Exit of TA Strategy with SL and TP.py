import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
import talib as ta
import pandas_datareader.data as pdr
import matplotlib.gridspec as gridspec
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import warnings
from hurst import compute_Hc as he
import yfinance as yf

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")


# OUTLIER TREATMENT FUNCTION #

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


# RISK MANAGEMENT INDICATOR #

def exponential_ATR(data, lookback):
    data["TR"] = ta.TRANGE(data.High, data.Low, data.Close)
    data["ATR"] = ta.ATR(data.High, data.Low, data.Close, timeperiod=lookback)

    a = (2/(lookback + 1))

    eATR = []
    for i in range(len(data)):
        eATR.append((a * data.iloc[i, 4]) + ((1 - a) * data.iloc[i - 1, 5]))
    data["eATR"] = eATR

    data.dropna(inplace=True)
    data.drop(columns=["TR", "ATR"], inplace=True)

    return data


# INDICATORS #

def moving_averages(data, lookback1, lookback2):
    data["SHORT_MA"] = ta.EMA(data.Close, timeperiod=lookback1)
    data["LONG_MA"] = ta.EMA(data.Close, timeperiod=lookback2)

    data.dropna(inplace=True)

    return data


def rsi(data, lookback):
    data["RSI"] = ta.RSI(data.Close, timeperiod=lookback)
    data.dropna(inplace=True)

    return data


# SYSTEMATIC RULES - SIGNAL GENERATION #

def signal(data, buy_col, sell_col, indicator1_col, indicator2_col, indicator3_col):

    for i in range(len(data)):
        # Buy signal
        if data.iloc[i, indicator1_col] > data.iloc[i, indicator2_col] and \
                data.iloc[i - 1, indicator1_col] < data.iloc[i, indicator2_col] and \
                data.iloc[i - 2, indicator1_col] < data.iloc[i, indicator2_col] and \
                data.iloc[i - 3, indicator1_col] < data.iloc[i, indicator2_col]:
            data.iloc[i, buy_col] = 1

        # Sell signal
        if data.iloc[i, indicator1_col] < data.iloc[i, indicator2_col] and \
                data.iloc[i - 1, indicator1_col] > data.iloc[i, indicator2_col] and \
                data.iloc[i - 2, indicator1_col] > data.iloc[i, indicator2_col] and \
                data.iloc[i - 3, indicator1_col] > data.iloc[i, indicator2_col]:
            data.iloc[i, sell_col] = -1

    # Print the number of buys and sell
    print("\n----------------------------")
    print("No. of Buys:  ", np.sum(data["BUY"] == 1))
    print("No. of Sells: ", np.sum(data["SELL"] == -1))
    print("")
    print("Total possible trades: ", np.sum(data["BUY"] == 1) + np.sum(data["SELL"] == -1))
    print("----------------------------")
    print("")

    return data


# GENERATE STOP LOSSES AND TAKE PROFITS FOR LONGS AND SHORTS #
def sl_tp(data, buy_signal, sell_signal, risk_reward):
    buy_eATR = data[data["BUY"] == 1]["eATR"]
    buy_stop = buy_signal - (1 * buy_eATR)
    buy_tp = buy_signal + (risk_reward * buy_eATR)

    sell_eATR = data[data["SELL"] == -1]["eATR"]
    sell_stop = sell_signal + (1 * sell_eATR)
    sell_tp = sell_signal - (risk_reward * sell_eATR)

    return buy_tp, buy_stop, sell_tp, sell_stop


# VISUALIZATION FUNCTION FOR THE STRATEGY #

def visualize_strategy(data, buy_signal, sell_signal, buy_tp, buy_stop, sell_tp, sell_stop):
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(nrows=11, ncols=2, figure=fig)

    ax1 = fig.add_subplot(gs[0:8, :])
    ax1.set_ylim(np.min(data.Low) - 0.01, np.max(data.High) + 0.01)
    ax1.plot(data.Close, c="black", lw=1.)
    ax1.plot(data.SHORT_MA, c="orange", lw=0.8, label="short ma")
    ax1.plot(data.LONG_MA, c="blue", lw=0.8, label="long_ma")
    ax1.plot_date(x=buy_signal.index, y=buy_signal, c="green", ms=3)
    ax1.plot_date(x=sell_signal.index, y=sell_signal, c="red", ms=3)
    ax1.legend(loc="upper right")

    ax1.scatter(x=buy_tp.index, y=buy_tp, c="green",
                marker="_", s=30)
    ax1.scatter(x=sell_tp.index, y=sell_tp, c="green",
                marker="_", s=30)
    ax1.scatter(x=buy_stop.index, y=buy_stop, c="red",
                marker="_", s=30)
    ax1.scatter(x=sell_stop.index, y=sell_stop, c="red",
                marker="_", s=30)

    ax2 = fig.add_subplot(gs[8:11, :], sharex=ax1)
    ax2.plot(data.RSI, c="red", lw=0.8)
    ax2.axhline(y=70, lw=0.7, ls="--", c="black")
    ax2.axhline(y=30, lw=0.7, ls="--", c="black")
    ax2.axhline(y=60, lw=0.7, ls="--", c="black")
    ax2.axhline(y=50, lw=0.7, ls="--", c="black")
    ax2.axhline(y=40, lw=0.7, ls="--", c="black")

    plt.subplots_adjust(left=0.021, bottom=0.0, right=0.998, top=0.993, hspace=0.029)
    multi = MultiCursor(fig.canvas, fig.axes, useblit=True, horizOn=True, vertOn=True, color="royalblue", lw=0.5)
    plt.show()


if __name__ == "__main__":
    # Import data
    symbol = "AUDNZD=X"
    df = yf.download(symbol, datetime(2000, 1, 1),
                        datetime.now()).drop(columns=["Adj Close", "Volume"])

    # Remove outliers and interpolate between missing values
    df = dealing_with_outliers_ma_res_iqr(data=df, lookback=10, n=1.5, method="linear")

    # Create the risk management indicator here for later purposes
    df = exponential_ATR(data=df, lookback=14)

    # Create the indicators used in the strategy
    df = moving_averages(data=df, lookback1=20, lookback2=50)
    df = rsi(data=df, lookback=8)

    # Create buy and sell signals
    df["BUY"], df["SELL"] = 0, 0
    print(df.info())

    df = signal(data=df, buy_col=8, sell_col=9, indicator1_col=5, indicator2_col=6, indicator3_col=7)
    print(df)

    # Visualize the strategy with the buy and sell signals

    # buy and sell locations
    buy_signal = df[df["BUY"] == 1]["Close"]
    sell_signal = df[df["SELL"] == -1]["Close"]

    # Get the stop loss and take profit positions based on eATR
    buy_tp, buy_stop, sell_tp, sell_stop = sl_tp(
        data=df, buy_signal=buy_signal, sell_signal=sell_signal, risk_reward=2
    )

    # plot close price, strategy and stops with take profits
    visualize_strategy(data=df,
                       buy_signal=buy_signal,
                       sell_signal=sell_signal,
                       buy_tp=buy_tp,
                       buy_stop=buy_stop,
                       sell_tp=sell_tp,
                       sell_stop=sell_stop)

    buy_signal_sl_tp = pd.DataFrame([buy_signal.values, buy_stop.values, buy_tp.values]).T
    buy_signal_sl_tp.index = buy_signal.index
    buy_signal_sl_tp.columns = ["BUY_PRICE", "SL", "TP"]

    sell_signal_sl_tp = pd.DataFrame([sell_signal.values, sell_stop.values, sell_tp.values]).T
    sell_signal_sl_tp.index = sell_signal.index
    sell_signal_sl_tp.columns = ["SELL_PRICE", "SL", "TP"]

    print(buy_signal_sl_tp)
    print(sell_signal_sl_tp)










