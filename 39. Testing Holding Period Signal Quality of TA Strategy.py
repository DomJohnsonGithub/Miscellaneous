import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
import talib as ta
import pandas_datareader.data as pdr
import matplotlib.gridspec as gridspec
from datetime import datetime
import seaborn as sns
import yfinance as yf

sns.set_style("darkgrid")

def stochastic_oscillator(data, fastk_period, slowk_period,
                          slowk_matype, slowd_period,
                          slowd_matype):
    """
    """
    slowk, slowd = ta.STOCH(data.High, data.Low, data.Close,
                            fastk_period=fastk_period,
                            slowk_period=slowk_period,
                            slowk_matype=slowk_matype,
                            slowd_period=slowd_period,
                            slowd_matype=slowd_matype)
    data["STOCH_OSC"] = slowk
    return data


def stochastic_bollinger_bands(data, deviation, period, ma_type):
    up_bb, _, lw_bb = ta.BBANDS(data.STOCH_OSC, timeperiod=period,
                                nbdevup=deviation, nbdevdn=deviation,
                                matype=ma_type)
    data["UP_BB"], data["LW_BB"] = up_bb, lw_bb
    return data


def net_stoch_bb(data):
    data["NET"] = 0
    for i in range(len(data)):
        if data.iloc[i, 5] - data.iloc[i, 4] < 0:
            data.iloc[i, 7] = 1
        elif data.iloc[i, 4] - data.iloc[i, 6] < 0:
            data.iloc[i, 7] = -1

    return data


def ma(data, lookback):
    data["MA"] = data.Close.rolling(window=lookback).mean()
    return data


def signal(data):
    data["BUY"], data["SELL"] = 0, 0

    for i in range(len(data)):

        if data.iloc[i, 4] < data.iloc[i, 6] \
                and data.iloc[i-1, 4] > data.iloc[i-1, 6] \
                and data.iloc[i, 8] <= data.iloc[i, 3]:
         data.iloc[i, 9] = 1

        if data.iloc[i, 4] > data.iloc[i, 5] \
                and data.iloc[i-1, 4] < data.iloc[i-1, 5] \
                and data.iloc[i, 8] >= data.iloc[i, 3]:
         data.iloc[i, 10] = -1

    return data


def visualize_signals(data, bull, bear):
    data1 = data.Close
    data2 = data.STOCH_OSC
    data3 = data.UP_BB
    data4 = data.LW_BB
    data5 = data.NET
    data6 = data.MA

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(nrows=5, ncols=2, figure=fig)

    ax1 = fig.add_subplot(gs[0:3, :])
    ax1.plot(data1, lw=0.8, c="black")
    ax1.plot(data6, lw=0.6, c="orange")
    ax1.set_ylim(np.min(data1)-0.01, np.max(data1)+0.01)
    ax1.scatter(x=bull.index, y=bull, marker="^", c="green")
    ax1.scatter(x=bear.index, y=bear, marker="v", c="red")
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[3, :], sharex=ax1)
    ax2.plot(data2, lw=0.8, c="black", label="STOCH OSC")
    ax2.plot(data3, lw=0.8, c="red")
    ax2.plot(data4, lw=0.8, c="red")
    ax2.axhline(y=80, lw=0.5, ls="--", c="grey")
    ax2.axhline(y=20, lw=0.5, ls="--", c="grey")
    ax2.legend(loc="upper left")

    ax3 = fig.add_subplot(gs[4, :], sharex=ax2)
    ax3.plot(data5, lw=0.8, c="black")
    ax3.axhline(y=1, ls="--", c="red", lw=0.5)
    ax3.axhline(y=-1, ls="--", c="red", lw=0.5)

    plt.subplots_adjust(left=0.033, bottom=0.03, right=0.995, top=0.993, hspace=0)
    multi = MultiCursor(fig.canvas, fig.axes, useblit=True, horizOn=True, vertOn=True, color="royalblue", lw=0.5)
    plt.show()



def signal_quality_metric(data, holding_periods):
    sq = pd.DataFrame(index=data.index, columns=[f"SQ_HP({i})" for i in holding_periods])

    for n, j in enumerate(holding_periods):
        for i in range(len(data)):
            try:
                if data.iloc[i, 6] == 1:
                    sq.iloc[i + j, n] = data.iloc[i + j, 3] - data.iloc[i, 3]
                if data.iloc[i, 7] == -1:
                    sq.iloc[i + j, n] = data.iloc[i, 3] - data.iloc[i + j, 3]
            except IndexError:
                pass

    for holding_period in holding_periods:
        positives = sq[sq.loc[:, f"SQ_HP({holding_period})"] > 0]
        negatives = sq[sq.loc[:, f"SQ_HP({holding_period})"] < 0]

        # Calc. Signal Quality
        signal_quality = len(positives) / (len(positives) + len(negatives))
        print(f"SIGNAL QUALITY METRIC - Holding Period ({holding_period}): ", np.round(signal_quality * 100, 2), "%")


def holding(data):
    """
    For simplicity, we can consider buying and selling closing prices.
    This means that when we get a signal from our strategy on close,
    we initiate the trade on the close until getting another signal
    where we exit and initiate the new trade.
    """
    # Create Buy and Sell Return Columns
    data["BUY_RETURNS"], data["SELL_RETURNS"] = 0, 0

    # Calculates the gross profits and losses from individual trades
    for i in range(len(data)):
        try:
            if data.iloc[i, 9] == 1:
                for a in range(i + 1, i + 1000):
                    if data.iloc[a, 9] != 0 or data.iloc[a, 10] != 0:
                        data.iloc[a, 11] = (data.iloc[a, 3] - data.iloc[i, 3])
                        break
                    else:
                        continue

            elif data.iloc[i, 10] == -1:
                for a in range(i + 1, i + 1000):
                    if data.iloc[a, 9] != 0 or data.iloc[a, 10] != 0:
                        data.iloc[a, 12] = (data.iloc[i, 3] - data.iloc[a, 3])
                        break
                    else:
                        continue
        except IndexError:
            pass

    return data


def indexer(data, expected_cost, lot_size, capital_invested):
    """
    Transform the buy and sell return columns into cumulative
    numbers to generate the Equity Curve.
    """
    # Charting portfolio evolution
    indexer = data.iloc[:, 11:13]

    # Creating a combined array for long and short returns
    z = np.zeros((len(data), 1), dtype=np.float)
    indexer = np.append(indexer, z, axis=1)

    # Combining Returns
    for i in range(len(indexer)):
        try:
            if indexer[i, 0] != 0:
                indexer[i, 2] = indexer[i, 0] - (expected_cost / lot_size)
            if indexer[i, 1] != 0:
                indexer[i, 2] = indexer[i, 1] - (expected_cost / lot_size)
        except IndexError:
            pass

    # Switching to monetary values
    indexer[:, 2] = indexer[:, 2] * lot_size

    # Creating a portfolio balance array
    indexer = np.append(indexer, z, axis=1)
    indexer[:, 3] = capital_invested

    # Adding returns to the balance
    for i in range(len(indexer)):
        indexer[i, 3] = indexer[i - 1, 3] + (indexer[i, 2])
    indexer = np.array(indexer)

    return np.array(indexer)


def equity_curve_viz(data, capital_investment, dates):
    fig, ax = plt.subplots(dpi=50)
    ax.plot(dates, data[:, 3], c="red", lw=0.9, label="Account Balance")
    ax.set_title("Equity Curve / Strategy Returns", fontsize=20)
    ax.legend("upper left")
    ax.axhline(y=capital_investment, c="black", lw=1)
    plt.show()


def performance(indexer, data, name):
    """
    Provides metrics to assess the quality of the trading strategy.
    """
    # Profitability index
    indexer = np.delete(indexer, 0, axis=1)
    indexer = np.delete(indexer, 0, axis=1)

    profits, losses = [], []
    np.count_nonzero(data.iloc[:, 9])
    np.count_nonzero(data.iloc[:, 10])

    for i in range(len(indexer)):

        if indexer[i, 0] > 0:
            value = indexer[i, 0]
            profits = np.append(profits, value)

        if indexer[i, 0] < 0:
            value = indexer[i, 0]
            losses = np.append(losses, value)

    # Hit ratio calculation
    hit_ratio = round((len(profits) / (len(profits) + len(losses))) * 100, 2)

    realized_risk_reward = round(abs(profits.mean() / losses.mean()), 2)

    # Expected and total profits / losses
    expected_profits = np.mean(profits)
    expected_losses = np.abs(np.mean(losses))
    total_profits = round(np.sum(profits), 3)
    total_losses = round(np.abs(np.sum(losses)), 3)

    # Expectancy
    expectancy = round((expected_profits * (hit_ratio / 100)) \
                       - (expected_losses * (1 - (hit_ratio / 100))), 2)

    # Largest Win and Largest Loss
    largest_win = round(max(profits), 2)
    largest_loss = round(min(losses), 2)

    # Total Return
    indexer = data.iloc[:, 11:13]

    # Creating a combined array for long and short returns
    z = np.zeros((len(data), 1), dtype=float)
    indexer = np.append(indexer, z, axis=1)

    # Combining Returns
    for i in range(len(indexer)):
        try:
            if indexer[i, 0] != 0:
                indexer[i, 2] = indexer[i, 0] - (expected_cost / lot)

            if indexer[i, 1] != 0:
                indexer[i, 2] = indexer[i, 1] - (expected_cost / lot)
        except IndexError:
            pass

    # Switching to monetary values
    indexer[:, 2] = indexer[:, 2] * lot

    # Creating a portfolio balance array
    indexer = np.append(indexer, z, axis=1)
    indexer[:, 3] = investment

    # Adding returns to the balance
    for i in range(len(indexer)):
        indexer[i, 3] = indexer[i - 1, 3] + (indexer[i, 2])
    indexer = np.array(indexer)

    total_return = (indexer[-1, 3] / indexer[0, 3]) - 1
    total_return = total_return * 100

    print("\n# ---------- Performance ---------- #", name, "#")
    print("Hit Ratio      = ", hit_ratio, "%")
    print("Net Profit     = ", "$", round(indexer[-1, 3] - indexer[0, 3], 2))
    print("Expectancy     = ", "$", expectancy, "per trade")
    print("Profit Factor  = ", round((total_profits / total_losses), 2))
    print("Total Return   = ", round(total_return, 2), "%")
    print("\nAverage Gain = ", "$", round((expected_profits), 2), "per trade")
    print("Average Loss   = ", "$", round((expected_losses * -1), 2), "per trade")
    print("Largest Gain   = ", "$", largest_win)
    print("Largest Loss   = ", "$", largest_loss)
    print("\nRealized RR  = ", realized_risk_reward)
    print("Minimum        = ", "$", round(min(indexer[:, 3]), 2))
    print("Maximum        = ", "$", round(max(indexer[:, 3]), 2))
    print("Trades         = ", len(profits) + len(losses))


if __name__ == "__main__":

    # Import data
    symbol = "EURUSD=X"
    df = yf.download(symbol, datetime(2000, 1, 1), datetime.now()).drop(columns=["Adj Close", "Volume"])

    # Stochastic Oscillator
    fastk_period = 5
    slowk_period = 3
    slowk_matype = 0
    slowd_period = 3
    slowd_matype = 0
    df = stochastic_oscillator(data=df,
                               fastk_period=fastk_period,
                               slowk_period=slowk_period,
                               slowk_matype=slowk_matype,
                               slowd_period=slowd_period,
                               slowd_matype=slowd_matype)

    # Stochastic Oscillator Bollinger Bands
    deviation = 2; period = 10; ma_type=0
    df = stochastic_bollinger_bands(data=df, deviation=deviation,
                                    period=period, ma_type=ma_type)

    # Alternative version of the indicator
    df = net_stoch_bb(data=df)

    # Generate a MA to use in strategy as a filter
    lookback = 20
    df = ma(data=df, lookback=lookback)

    # Generate buy and sell signals
    df = signal(data=df)

    # Assess the signal quality - this resembles a fixed holding period strategy
    holding_periods = [1, 2, 3, 4, 5, 10, 21, 63]
    signal_quality_metric(data=df, holding_periods=holding_periods)

    # Visualize the indicators with close price
    bull = df[df["BUY"] == 1]["Close"]
    bear = df[df["SELL"] == -1]["Close"]
    visualize_signals(data=df, bull=bull, bear=bear)

    # Get the gross profits and losses
    df = holding(data=df)

    print(df.info(verbose=True))

    # EQUITY CURVE #
    # using the function for a 0.1 lot strategy on $10,000 investment
    lot = 10000
    expected_cost = 0.5 * (lot / 10000)  # 0.5 pip spread
    investment = 10000

    equity_curve = indexer(data=df, expected_cost=expected_cost,
                           lot_size=lot, capital_invested=investment)

    # Visualize the equity curve
    equity_curve_viz(data=equity_curve, capital_investment=investment,
                     dates=df.index)

    # Obtain performance metrics
    performance(equity_curve, df, "EURUSD")
