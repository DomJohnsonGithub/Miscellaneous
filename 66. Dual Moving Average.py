import numpy as np
import pandas as pd
from pandas_datareader import data
import talib as ta
import matplotlib.pyplot as plt
import sys
import yfinance as yf


def load_financial_data(start_date, end_date, output_file):
    try:
        df = pd.read_pickle(output_file)
        print('File data found...reading GOOG data')
    except FileNotFoundError:
        print('File not found...downloading the GOOG data')
        df = yf.download('GOOG', start_date, end_date)
        df.to_pickle(output_file)
    return df


goog_data = load_financial_data(start_date='2001-01-01',
                                end_date='2018-01-01',
                                output_file='goog_data_large.pkl')


def double_moving_average(financial_data, short_window, long_window):
    signals = pd.DataFrame(index=financial_data.index)
    signals["signal"] = 0.0
    signals["short_MA"] = ta.MA(financial_data["Close"], timeperiod=short_window)
    signals["long_MA"] = ta.MA(financial_data["Close"], timeperiod=long_window)
    signals["signal"][short_window:] = \
        np.where(signals["short_MA"][short_window:] >
                 signals["long_MA"][short_window:], 1.0, 0.0)
    signals["orders"] = signals["signal"].diff()
    return signals


ts = double_moving_average(goog_data, 20, 100)
ts.dropna(inplace=True)
print(ts)
print(goog_data)
# 1 = long, 0 = short

print(len(goog_data))
print(len(ts))


def plot_curve_orders(financial_data, ts):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='Google price in $')
    financial_data["Adj Close"][(len(financial_data) - len(ts)):].plot(ax=ax1, color='g', lw=.5)
    ts["short_MA"].plot(ax=ax1, color='r', lw=2.)
    ts["long_MA"].plot(ax=ax1, color='b', lw=2.)
    ax1.plot(ts.loc[ts.orders == 1.0].index,
             financial_data["Adj Close"][(len(financial_data) - len(ts)):][ts.orders == 1.0],
             '^', markersize=7, color='k')
    ax1.plot(ts.loc[ts.orders == -1.0].index,
             financial_data["Adj Close"][(len(financial_data) - len(ts)):][ts.orders == -1.0],
             'v', markersize=7, color='k')
    plt.legend(["Price", "Short mavg", "Long mavg", "Buy", "Sell"])
    plt.title("Double Moving Average Trading Strategy")
    plt.show()


plot_curve_orders(financial_data=goog_data, ts=ts)

# You are going to set your initial amount of money you want
# to invest --- here it is 10,000
initial_capital = float(10000.0)

# You are going to create a new dataframe positions
# Remember the index is still the same as signals
positions = pd.DataFrame(index=ts.index).fillna(0.0)

# You are going to buy 10 shares of MSFT when signal is 1
# You are going to sell 10 shares of MSFT when signal is -1
# You will assign these values to the column MSFT of the
# dataframe positions
positions['MSFT'] = 10 * ts['signal']

financial_data = goog_data

# You are now going to calculate the notional (quantity x price)
# for your portfolio. You will multiply Adj Close from
# the dataframe containing prices and the positions (10 shares)
# You will store it into the variable portfolio
portfolio = positions.multiply(financial_data['Adj Close'], axis=0)

# Add `holdings` to portfolio
portfolio['holdings'] = (positions.multiply(financial_data['Adj Close'], axis=0)).sum(axis=1)

# You will store positions.diff into pos_diff
pos_diff = positions.diff()
# You will now add a column cash in your dataframe portfolio
# which will calculate the amount of cash you have
# initial_capital - (the notional you use for your different buy/sell)
portfolio['cash'] = initial_capital - (pos_diff.multiply(financial_data['Adj Close'], axis=0)).sum(axis=1).cumsum()

# You will now add a column total to your portfolio calculating the part of holding
# and the part of cash
portfolio['total'] = portfolio['cash'] + portfolio['holdings']

# Add `returns` to portfolio
portfolio['returns'] = portfolio['total'].pct_change()

# Print the first lines of `portfolio`
print(portfolio)
