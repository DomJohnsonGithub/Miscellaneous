# Python Module with Class
# for Vectorized Backtesting
# of Momentum-Based Strategies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MomVectorBacktester(object):
    """ Class for the vectorized backtesting of
    momentum-based trading strategies.

    Attributes
    ==========
    symbol: str
        RIC (financial instrument) to work with
    start: str
        start date for data selection
    end: str
        end date for data selection
    amount: int, float
        amount to be invested at the beginning
    tc: float
        proportional transaction costs (e.g., 0.5% = 0.005) per trade

    Methods
    =======
    get_data:
        retrieves and prepares the base data set
    run_strategy:
        runs the backtest for the momentum-based strategy
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
        raw = pd.read_csv("aiif_eikon_eod_data.csv", index_col=0, parse_dates=True, delimiter=",")
        raw = pd.DataFrame(raw[self.symbol]).rename(columns={self.symbol : "price"})
        raw = raw.loc[self.start:self.end]
        raw["returns"] = np.log(raw/raw.shift(1))

        self.data = raw

    def run_strategy(self, momentum=1):
        """Backtests the trading strategy.
        """
        self.momentum = momentum
        data = self.data.copy().dropna()
        data["position"] = np.sign(data["returns"].rolling(momentum).mean())
        data["strategy"] = (data["position"].shift(1) * data["returns"])
        # determine when a trade takes place
        data.dropna(inplace=True)
        trades = data["position"].diff().fillna(0) != 0
        # subtract transaction costsv from return when trade takes place
        data["strategy"][trades] -= self.tc
        data["creturns"] = self.amount * data.returns.cumsum().apply(np.exp)
        data["cstrategy"] = self.amount * data["strategy"].cumsum().apply(np.exp)
        self.results = data
        # absolute performance of the strategy
        aperf = self.results["cstrategy"].iloc[-1]
        # out-/underperformance of the strategy
        operf = aperf - self.results["creturns"].iloc[-1]
        return round(aperf, 2), round(operf, 2)

    def plot_results(self):
        """Plots the cumulative performance of the trading
        strategy compared to the symbol.
        """
        if self.results is None:
            print("No results to plot yet. Run a strategy")
        title = "%s | TC = %.4f" % (self.symbol, self.tc)
        self.results[["creturns", "cstrategy"]].plot(title=title,
                                                     figsize=(10, 6))
        plt.show()


if __name__ == "__main__":
    mombt = MomVectorBacktester("XAU=", "2010-1-1", "2020-12-31",
                                10000, 0.0)
    print(mombt.run_strategy())
    print(mombt.plot_results())

    print(mombt.run_strategy(momentum=3))
    print(mombt.plot_results())

    mombt = MomVectorBacktester("XAU=", "2010-1-1", "2020-12-31",
                                10000, 0.001)  # assuming transaction cost of 0.1% per trade
    print(mombt.run_strategy(momentum=3))
    print(mombt.plot_results())