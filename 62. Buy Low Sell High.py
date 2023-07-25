## Imports & Settings ##
import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf

# Set print display options
pd.set_option("display.max_colwidth", 1000)
pd.set_option("display.width", 1000)

## Preparing the data - signal ##

# Start and end date for data
start = datetime(2014, 1, 1)
end = datetime(2018, 1, 1)

# Read in the Google price data
goog_data = yf.download("GOOG", start, end)

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


## Signal visualization ##

# Define function to plot the strategy performance chart
def plot_strategy_performance():
    # define a fig. chart
    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel="Google price in $")
    # plot price
    goog_data_signal["price"].plot(ax=ax1, color="r", lw=2.)
    # buying a share
    ax1.plot(goog_data_signal.loc[goog_data_signal.positions == 1.0].index,
             goog_data_signal.price[goog_data_signal.positions == 1.0],
             "^", markersize=5, color="k")
    # selling a share
    ax1.plot(goog_data_signal.loc[goog_data_signal.positions == -1.0].index,
             goog_data_signal.price[goog_data_signal.positions == -1.0],
             "^", markersize=5, color="k")

    plt.show()


## Backtesting ##
"""
we will have a portfolio (grouping of financial assets such as bonds and stocks) composed 
of only one type of stock: Google (GOOG). We will start this portfolio with $1,000:
"""
initial_capital = float(1000.0)

# Create a DataFrame for the positions and the portfolio
positions = pd.DataFrame(index=goog_data_signal.index).fillna(0.0)
portfolio = pd.DataFrame(index=goog_data_signal.index).fillna(0.0)

# Store GOOG positions in the following df
positions["GOOG"] = goog_data_signal["signal"]

# Store the amount of GOOG positions for the portfolio in this one:
portfolio["positions"] = (positions.mul(goog_data_signal["price"], axis=0))

# Calculate the non-invested cash
portfolio["cash"] = initial_capital - \
                    (positions.diff().mul(goog_data_signal["price"], axis=0)).cumsum()

# The total investment will be calculated by summing the positions and the cash
portfolio["total"] = portfolio["positions"] + portfolio["cash"]


# Plot the profitability
def strat_profitability():
    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel="Google price in $")
    pf = portfolio
    ax1.plot(pf.positions)
    ax1.plot(pf.cash)
    ax1.plot(pf.total)
    plt.show()


if __name__ == "__main__":
    # Display Google price dataframe
    print(goog_data)

    # Display the new df
    print(goog_data_signal.head(10))

    # Strategy Performance
    plot_strategy_performance()

    # See positions and portfolio
    print(positions)
    print(portfolio)

    # Plot strat. profitability
    strat_profitability()
