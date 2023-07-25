import backtrader as bt
from datetime import datetime


class maCross(bt.Strategy):
    params = (
        ("sma1", 40),
        ("sma2", 200),
        ("oneplot", True)  # all data on same master plot
    )

    def __init__(self):

        self.inds = dict()
        for i, d in enumerate(self.datas):
            self.inds[d] = dict()
            self.inds[d]['sma1'] = bt.indicators.SimpleMovingAverage(
                d.close, period=self.params.sma1)
            self.inds[d]['sma2'] = bt.indicators.SimpleMovingAverage(
                d.close, period=self.params.sma2)
            self.inds[d]['cross'] = bt.indicators.CrossOver(self.inds[d]['sma1'], self.inds[d]['sma2'])

            if i > 0:  # Check we are not on the first loop of data feed:
                if self.p.oneplot:
                    d.plotinfo.plotmaster = self.datas[0]

    def next(self):
        for i, d in enumerate(self.datas):
            dt, dn = self.datetime.date(), d._name
            if not self.position:  # no market / no orders
                if self.inds[d]["cross"][0] == 1:
                    self.buy(data=d, size=1000)
                elif self.inds[d]["cross"][0] == -1:
                    self.sell(data=d, size=1000)
            else:
                if self.inds[d]["cross"][0] == 1:
                    self.close(data=d)
                    self.buy(data=d, size=1000)
                elif self.inds[d]["cross"][0] == -1:
                    self.close(data=d)
                    self.sell(data=d, size=1000)

    def notify_trade(self, trade):
        dt = self.data.datetime.date()
        if trade.isclosed:
            print(f"{dt} {trade.data._name} Closed: P/L Gross ${round(trade.pnl, 2)}, Net ${round(trade.pnlcomm, 2)}")


class MultipleData(bt.feeds.YahooFinanceCSVData):
    params = (
        ("nullvalue", float("NaN")),
        ("dtformat", "%Y-%m-%d"),
        ("datetime", 1),
        ("open", 2),
        ("high", 3),
        ("low", 4),
        ("adj_close", 5),
        ("volume", 6),
    )


# Create an instance of Cerebro
cerebro = bt.Cerebro()

# Create our datalist
datalist = [
    ("cur_EURUSD=X.csv", "EURUSD"),
    ("cur_AUDNZD=X.csv", "AUDNZD"),
    ("cur_GBPCAD=X.csv", "GBPCAD"),
    ("cur_JPYCHF=X.csv", "JPYCHF"),
]

# Loop through the list adding to cerebro
for i in range(len(datalist)):
    data = MultipleData(dataname=datalist[i][0])
    cerebro.adddata(data, name=datalist[i][1])

# Set our desired cash start
starting_capital = 1000000
cerebro.broker.setcash(starting_capital)

# Run over everything
cerebro.run()

# Get final portfolio Value
portvalue = cerebro.broker.getvalue()
pnl = portvalue - starting_capital

# Print out the final result
print('Final Portfolio Value: ${}'.format(portvalue))
print('P/L: ${}'.format(pnl))

# Finally plot the end results
cerebro.plot(style='candlestick')
