import pandas as pd
import pandas_datareader.data as pdr
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import re
from itertools import chain
from datetime import datetime

# Path for saving data
DATA_STORE = Path("C:\\Users\\domin\\PycharmProjects\\Miscellaneous\\dow_stocks.h5")

# Website to scrape
res = requests.get("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average").content
soup = BeautifulSoup(res, "lxml")
table = soup.find("table", attrs={"class": "wikitable sortable"})

# Scrape DJIA components
tickers = []
for row in table.findAll("tr"):
    cells = row.find_all("a", {"rel": "nofollow"})
    tickers.append(cells)

tickers = list(chain(*tickers))

# Clean stock tickers
stocks = []
for stock in tickers:
    ticker = re.sub('<[^<]+?>', '', str(stock))
    stocks.append(ticker)

print(stocks)  # List of stocks
print(len(stocks))  # Number of stocks

# Retrieve Dow Jones stock data
start, end = datetime(2000, 1, 1), datetime(2019, 1, 1)
df = pdr.DataReader(stocks, "yahoo", start=start, end=end).stack().swaplevel(0, 1).sort_index(level=0)

# Build the DataFrame
df = df.rename_axis(["ticker", "date"])
df.columns = ["adj_close", "close", "high", "low", "open", "volume"]
df = df.filter(items=["adj_close", "high", "low", "open", "volume"]).rename(columns=lambda x: x.replace('adj_', ''))

print(df)
print(df.info(show_counts=True, verbose=0))

# Store the DataFrame locally in HDF5 format
with pd.HDFStore(DATA_STORE, "w") as store:
    store.put("DOW/stocks", df)
