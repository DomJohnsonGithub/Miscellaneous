import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib as ta


class MRVectorBacktester(object):
    """ Class for the vectorized backtesting of
    mean reversion-based trading strategies.

    Attributes
    ==========
    symbol: str
        RIC symbol with which to work
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval
    amount: int, float
        amount to be invested at the beginning
    tc: float
        proportional transaction costs (e.g., 0.5% = 0.005) per trade

    Methods
    =======
    get_data:
        retrieves and prepares the base data set
    run_strategy:
        runs the backtest for the mean reversion-based strategy
    plot_results:
        plots the performance of the strategy compared to the symbol
    """

    def __init__(self, symbol, start, end, amount, tc):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc

        self.results = None
        self.get_data()

    def get_data(self):
        """Retrieves and prepares the data.
        """
        raw = pd.read_csv("aiif_eikon_eod_data.csv", index_col=0, parse_dates=True, delimiter=",").dropna()
        raw = pd.DataFrame(raw[self.symbol])
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: "price"}, inplace=True)
        raw["return"] = np.log(raw / raw.shift(1))
        self.data = raw

    def run_strategy(self, SMA, threshold):
        """Backtests the trading strategy.
        """
        data = self.data.copy().dropna()
        data["sma"] = ta.SMA(data["price"], timeperiod=SMA)
        data["distance"] = data["price"] - data["sma"]
        data.dropna(inplace=True)

        # sell signals
        data["position"] = np.where(data["distance"] > threshold, -1, np.nan)
        # buy signals
        data["position"] = np.where(data["distance"] < -threshold, 1, data["position"])
        # crossing of current price and SMA (zero distance)
        data["position"] = np.where(data["distance"] * data["distance"].shift(1) < 0, 0, data["position"])
        data["position"] = data.position.ffill().fillna(0)

        data["strategy"] = data["position"].shift(1) * data["return"]

        # determine when a trade takes place
        trades = data.position.diff().fillna(0) != 0

        # subtract transaction costs from return when trade takes place
        data["strategy"][trades] -= self.tc
        data["creturns"] = self.amount * data["return"].cumsum().apply(np.exp)
        data["cstrategy"] = self.amount * data["strategy"].cumsum().apply(np.exp)

        self.results = data

        # absolute performance of the strategy
        aperf = self.results["cstrategy"].iloc[-1]
        # out -/underperformance of strategy
        operf = aperf - self.results["creturns"].iloc[-1]

        return round(aperf, 2), round(operf, 2)

    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to the symbol.
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        title = '%s | TC = %.4f' % (self.symbol, self.tc)
        self.results[['creturns', 'cstrategy']].plot(title=title,
                                                     figsize=(10, 6))
        plt.show()


if __name__ == "__main__":
    mrbt = MRVectorBacktester("GLD", "2010-1-1", "2020-12-31", 10000, 0.001)

    print(mrbt.run_strategy(SMA=43, threshold=7.5))
    print(mrbt.plot_results())

    print(mrbt.run_strategy(SMA=100, threshold=8))
    print(mrbt.plot_results())
