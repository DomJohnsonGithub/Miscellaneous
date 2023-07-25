import pandas as pd
from pathlib import Path
import hurst as hst
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_hurst_exponent(data, start=None, end=None):
    if start and end:
        data = data.loc[start:end]
    H = hst.compute_Hc(data)
    return H[0]


def plot_stock_data(df, ticker):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, edgecolor="black", facecolor="white", figsize=(10, 8))

    ax1.plot(df["close"])
    ax1.set_title(f"Ticker: {ticker}, Adj Close", color="b")
    ax1.set_ylabel("Price", color="b")
    ax1.grid(color="grey", linestyle="--", linewidth=0.9, alpha=0.5)
    ax1.set_facecolor("white")

    ax2.plot(df["returns"])
    ax2.set_title(f"Ticker: {ticker}, Returns", color="b")
    ax2.set_ylabel("Returns", color="b")
    ax2.grid(color="grey", linestyle="--", linewidth=0.9, alpha=0.5)
    ax2.set_facecolor("white")

    ax3.hist(df["close"], histtype="step", bins="rice")
    ax3.set_title(f"Ticker: {ticker}, Adj Close Distribution", color="b")
    ax3.set_ylabel("Price", color="b")
    ax3.grid(color="grey", linestyle="--", linewidth=0.9, alpha=0.5)
    ax3.set_facecolor("white")

    ax4.hist(df["returns"], histtype="step", bins="rice")
    ax4.set_title(f"Ticker: {ticker}, Returns Distribution", color="b")
    ax4.set_ylabel("Price", color="b")
    ax4.grid(color="grey", linestyle="--", linewidth=0.9, alpha=0.5)
    ax4.set_facecolor("white")

    fig.tight_layout()
    plt.show()


# Path for retrieving stock DataFrame
DATA_STORE = Path("C:\\Users\\domin\\PycharmProjects\\Miscellaneous\\dow_stocks.h5")

# Retrieve the Stock Data
with pd.HDFStore(DATA_STORE, "r") as store:
    df = store.get("DOW/stocks")

print(df.info())
print(df)

# Check for Null/NaN Values
print("Null/NaN Value Counts:", df.isnull().values.any())

# Descriptive Statistics
tickers = df.index.get_level_values(0).unique().values
print("Tickers:", tickers)
print("No. of tickers:", len(tickers))
for ticker in tickers:
    print(f"Ticker: {ticker}")
    print(df.loc[ticker].describe())

# Hurst Exponent
for ticker in tickers:
    print(ticker)
    H = calculate_hurst_exponent(df.loc[ticker, "close"])
    print(f"Date: 2000:2019, Hurst Exponent = {round(H, 4)}")
    H = calculate_hurst_exponent(df.loc[ticker, "close"], start="2010")
    print(f"Date: 2010:2019, Hurst Exponent = {round(H, 4)}")
    H = calculate_hurst_exponent(df.loc[ticker, "close"], start="2015")
    print(f"Date: 2015:2019, Hurst Exponent = {round(H, 4)}")
    H = calculate_hurst_exponent(df.loc[ticker, "close"], start="2017")
    print(f"Date: 2017:2019, Hurst Exponent = {round(H, 4)}")
    print("----------------------------------------")

# Visualize Adj Close, Returns, Distributions
df["returns"] = df.groupby("ticker")["close"].diff(1)
df.dropna(inplace=True, how="any")
for ticker in tickers:
    plot_stock_data(df.loc[ticker], ticker)