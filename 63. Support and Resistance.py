import pandas as pd
import pandas_datareader.data as pdr
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

start = datetime(2014, 1, 1)
end = datetime(2018, 1, 1)
SRC_DATA_FILENAME = "goog_data.pkl"

try:
    goog_data2 = pd.read_pickle(SRC_DATA_FILENAME)
except FileNotFoundError:
    goog_data2 = yf.download("GOOG", start, end)
    goog_data2.to_pickle(SRC_DATA_FILENAME)

goog_data = goog_data2.tail(620)
lows = goog_data["Low"]
highs = goog_data["High"]

# Build a pandas DataFrame getting the same dimension as the DataFrame containing the data
goog_data_signal = pd.DataFrame(index=goog_data.index)

# Extract the "Adj Close" data to use as the price
goog_data_signal["price"] = goog_data["Adj Close"]

# Need a column, daily_difference, to store the difference between two consecutive days
# We'll use the diff function
goog_data_signal["daily_difference"] = goog_data_signal["price"].diff()

# Create a signal based on daily_difference column
# If the value is +ve, give a value of 1, if -ve, give the value 0
goog_data_signal["signal"] = 0.0
goog_data_signal["signal"] = np.where(goog_data_signal["daily_difference"] > 0, 1.0, 0.0)

# Limit the position on the market, it will be impossible to buy or sell
# more than one time consecutively. Therefore, we apply diff() fn to col. signal
goog_data_signal["positions"] = goog_data_signal["signal"].diff()


def plot_lows_highs():
    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='Google price in $')
    highs.plot(ax=ax1, color='c', lw=2.)
    lows.plot(ax=ax1, color='y', lw=2.)
    plt.hlines(highs.head(200).max(), lows.index.values[0], lows.index.values[-1]
               , linewidth=2, color='g')
    plt.hlines(lows.head(200).min(), lows.index.values[0], lows.index.values[-1],
               linewidth=2, color='r')
    plt.axvline(linewidth=2, color='b', x=lows.index.values[200], linestyle=':')
    plt.show()


def trading_support_resistance(data, bin_width=20):
    data['sup_tolerance'] = pd.Series(np.zeros(len(data)))
    data['res_tolerance'] = pd.Series(np.zeros(len(data)))
    data['sup_count'] = pd.Series(np.zeros(len(data)))
    data['res_count'] = pd.Series(np.zeros(len(data)))
    data['sup'] = pd.Series(np.zeros(len(data)))
    data['res'] = pd.Series(np.zeros(len(data)))
    data['positions'] = pd.Series(np.zeros(len(data)))
    data['signal'] = pd.Series(np.zeros(len(data)))
    in_support = 0
    in_resistance = 0
    for x in range((bin_width - 1) + bin_width, len(data)):
        data_section = data[x - bin_width:x + 1]
        support_level = min(data_section['price'])
        resistance_level = max(data_section['price'])
        range_level = resistance_level - support_level
        data['res'][x] = resistance_level
        data['sup'][x] = support_level
        data['sup_tolerance'][x] = support_level + 0.2 * range_level
        data['res_tolerance'][x] = resistance_level - 0.2 * range_level
        if data['price'][x] >= data['res_tolerance'][x] and \
                data['price'][x] <= data['res'][x]:
            in_resistance += 1
            data['res_count'][x] = in_resistance
        elif data['price'][x] <= data['sup_tolerance'][x] and \
                data['price'][x] >= data['sup'][x]:
            in_support += 1
            data['sup_count'][x] = in_support
        else:
            in_support = 0
            in_resistance = 0
        if in_resistance > 2:
            data['signal'][x] = 1
        elif in_support > 2:
            data['signal'][x] = 0
        else:
            data['signal'][x] = data['signal'][x - 1]
    data['positions'] = data['signal'].diff()


if __name__ == "__main__":
    plot_lows_highs()

    trading_support_resistance(goog_data_signal)
    plt.show()
