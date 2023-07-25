import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("darkgrid")
#  in addition to the hit ratio, the quality of the
# market timing matters.

class LRVectorizedBacktester(object):

    def __init__(self, symbol, start, end, amount, tc):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc

        self.results = None
        self.get_data()

    def get_data(self):
        raw = pd.read_csv("aiif_eikon_eod_data.csv", index_col=0, parse_dates=True, delimiter=",").dropna()
        raw = pd.DataFrame(raw[self.symbol]).rename(columns={self.symbol: "price"})
        raw = raw.loc[self.start:self.end]
        raw["returns"] = np.log(raw/raw.shift(1))
        raw.dropna(inplace=True)
        self.data = raw

    def select_data(self, start, end):
        data = self.data.loc[(self.data.index >= start) &
                             (self.data.index <= end)].copy()
        return data

    def prepare_lags(self, start, end):
        data = self.select_data(start, end)
        self.cols = []
        for lag in range(1, self.lags + 1):
            col = f"lag_{lag}"
            data[col] = data["returns"].shift(lag)
            self.cols.append(col)
        data.dropna(inplace=True)

        self.lagged_data = data

    def fit_model(self, start, end):
        self.prepare_lags(start, end)
        regression = np.linalg.lstsq(self.lagged_data[self.cols],
                                     np.sign(self.lagged_data["returns"]),
                                     rcond=None)[0]
        self.regression = regression

    def run_strategy(self, start_in, end_in, start_out, end_out, lags=5):
        self.lags = lags
        self.fit_model(start_in, end_in)
        self.results = self.select_data(start_out, end_out).iloc[lags:]
        self.prepare_lags(start_out, end_out)
        prediction = np.sign(np.dot(self.lagged_data[self.cols], self.regression))
        self.results['prediction'] = prediction
        self.results['strategy'] = self.results['prediction'] * \
                                   self.results['returns']
        # determine when a trade takes place
        trades = self.results['prediction'].diff().fillna(0) != 0
        # subtract transaction costs from return when trade takes place
        self.results['strategy'][trades] -= self.tc
        self.results['creturns'] = self.amount * \
                                   self.results['returns'].cumsum().apply(np.exp)
        self.results['cstrategy'] = self.amount * \
                                    self.results['strategy'].cumsum().apply(np.exp)
        # gross performance of the strategy
        aperf = self.results['cstrategy'].iloc[-1]
        # out-/underperformance of strategy
        operf = aperf - self.results['creturns'].iloc[-1]
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
    lrbt = LRVectorizedBacktester(symbol="GDX", start="2010-1-1", end="2019-12-31", amount=10000, tc=0.002)
    print(lrbt.run_strategy('2010-1-1', '2019-12-31',
                            '2010-1-1', '2019-12-31', lags=10))
    print(lrbt.run_strategy('2010-1-1', '2014-12-31',
                            '2015-1-1', '2019-12-31', lags=10))
    lrbt.plot_results()