import yfinance as yf
import mplfinance as mpf
from datetime import datetime
from matplotlib.widgets import MultiCursor

def download_stock_data(ticker, start, end):
    """
    Downloads stock data for the specified ticker and time range.

    Parameters:
    - ticker (str): Ticker symbol of the stock.
    - start (datetime): Start date of the data range.
    - end (datetime): End date of the data range.

    Returns:
    - DataFrame: Stock data for the specified ticker and time range.
    """
    df = yf.download(ticker, start=start, end=end).drop(columns=["Adj Close"])
    return df

def calculate_volatility_indicators(df):
    """
    Calculates volatility-related indicators for the given stock data DataFrame.

    Parameters:
    - df (DataFrame): Stock data DataFrame.

    Returns:
    - DataFrame: Stock data DataFrame with additional volatility indicators.
    """
    window = 14
    df["STD"] = df.Close.rolling(window=window).std()

    lookback = 14
    df["vol_of_vol"] = df["STD"].rolling(window=lookback).std()

    returns = (df.Close/df.Close.shift(1)) - 1
    returns_vol = returns.rolling(window=14).std()
    returns_vol_vol = returns_vol.rolling(window=14).std()

    df["returns_vol"] = returns_vol
    df["returns_vol_vol"] = returns_vol_vol

    return df

def plot_stock_data(df, ticker):
    """
    Plots the stock data along with the calculated volatility indicators.

    Parameters:
    - df (DataFrame): Stock data DataFrame.
    - ticker (str): Ticker symbol of the stock.

    Returns:
    - None
    """
    fig = mpf.figure(style="charles", figsize=(10, 6), dpi=60)
    ax0 = fig.add_subplot(5, 1, 1)
    ax1 = fig.add_subplot(5, 1, 2, sharex=ax0)
    ax2 = fig.add_subplot(5, 1, 3, sharex=ax1)
    ax3 = fig.add_subplot(5, 1, 4, sharex=ax2)
    ax4 = fig.add_subplot(5, 1, 5, sharex=ax3)

    std = mpf.make_addplot(df.STD, ax=ax1, color="blue")
    vol_of_vol = mpf.make_addplot(df.vol_of_vol, ax=ax2, color="red")
    ret_std = mpf.make_addplot(df.returns_vol, ax=ax3, color="green")
    ret_volvol = mpf.make_addplot(df.returns_vol_vol, ax=ax4, color="orange")

    mpf.plot(df.iloc[:, :4], ax=ax0, axtitle=f"{ticker}", type="candle", ylabel="Price ($)",
             addplot=[std, vol_of_vol, ret_std, ret_volvol],
             vlines=dict(vlines=['2002-10-09', "2007-02-27", "2007-10-11", "2008-09-16", "2009-11-27", "2010-04-27",
                                 "2010-05-06", "2011-08-01", "2015-06-12", "2015-08-18", "2018-09-20",
                                 "2020-02-24"], linewidths=(1), alpha=0.2, colors="orange"))

    mpf._mplwraps.plt.subplots_adjust(left=0.02, bottom=0.02, top=0.98, right=0.974, wspace=0, hspace=0)
    multi = MultiCursor(fig.canvas, fig.axes, useblit=True, horizOn=True, vertOn=True, color="red", lw=0.5)

    mpf.show()

# Specify the inputs
ticker = "GS"
start = datetime(2000, 1, 1)
end = datetime(2021, 1, 1)

# Download stock data
df = download_stock_data(ticker, start, end)

# Calculate volatility indicators
df = calculate_volatility_indicators(df)

# Plot stock data
plot_stock_data(df, ticker)
