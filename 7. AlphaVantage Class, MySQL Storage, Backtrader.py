import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import time
import pymysql
from sqlalchemy import create_engine
import talib as ta
from alpha_vantage.timeseries import TimeSeries
import backtrader as bt
from backtrader.feeds import PandasData

sns.set_style("darkgrid")
idx = pd.IndexSlice


class ALPHA_VANTAGE_DATA:
    def __init__(self, symbols: List[str], api_key: str):
        """
        Initialize function for retrieving asset data from alpha vantage
        :param symbols: list of asset symbols
        :param api_key: your API Key from Alpha Vantage
        """
        self.symbols = symbols
        self._ts = None
        self.dfs = []
        self.__api_key = api_key
        self._set_api_key()

    def _set_api_key(self):
        """
        This function is called during initialization
        to so that you can access Alpha Vantage data.
        """
        self._ts = TimeSeries(key=self.__api_key, output_format="pandas", indexing_type="date")

    def get_data(self, symbol: str) -> pd.DataFrame:
        """
        Retrieve single asset dataframe from alpha vantage
        :param symbol: ticker of a stock or other asset class
        :return: pandas dataframe
        """
        return self._ts.get_daily_adjusted(symbol, outputsize="full")[0]

    @staticmethod
    def transform_column(data: pd.DataFrame) -> pd.DataFrame:
        """
        Function to transform an alpha vantage dataframe
        :param data: alpha vantage dataframe
        :return: transformed dataframe
        """
        data.index.names = ["Date"]
        data.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "null0", "null1"]
        return data.drop(columns=["Adj Close", "null0", "null1"])

    def retrieve_data(self):
        """
        Retrieves asset data in batches of 5 and
        waits for 1 minute due to a limit on API calls.
        """
        batch_size = 5
        num_symbols = len(self.symbols)
        num_batches = (num_symbols + batch_size - 1) // batch_size

        try:
            for i in range(num_batches):
                start_index = i * batch_size
                end_index = min(start_index + batch_size, num_symbols)
                batch_symbols = self.symbols[start_index:end_index]

                # Execute function for the current batch of symbols
                for symbol in batch_symbols:
                    self.dfs.append(ALPHA_VANTAGE_DATA.transform_column(self.get_data(symbol)))

                if i < num_batches - 1:
                    time.sleep(60)
        except TimeoutError as e:
            print("Error occurred:", str(e), "Connection could not be established")
        except ConnectionError as e:
            print("Error occurred:", str(e), "Invalid API key")
        except ValueError as e:
            print("Error occurred:", str(e), "Asset class name does not exist")
        else:
            print("Data acquired successfully from Alpha Vantage")

    def combine_dataframes(self) -> pd.DataFrame:
        """
        This function combines individual dataframe into one.
        N.B.: dataframes differ in lenth and so some rows are
        dropped from the larger frames, so to as ensure we can
        concatenate the dataframe into a multiIndex.
        :return: multiIndex dataframe of assets OHLCV data.
        """
        combined_data = pd.concat([asset_data.sort_index().stack() for asset_data in self.dfs],
                                  axis="columns", keys=self.symbols, join="inner"
                                  ).unstack().stack(0).swaplevel(0).sort_index()
        combined_data.index.names = ["Ticker", "Date"]
        return combined_data

# Some FTSE 100 Tickers
tickers = sorted(
    {"TSCO.L", "JD.L", "GSK.L", "RR.L", "BATS.L", "BA.L", "UU.L", "SSE.L", "SPX.L", "SMT.L", "NXT.L", "SDR.L", "AAL.L",
     "ABF.L", "BARC.L", "VOD.L", "UTG.L", "STAN.L", "SMIN.L", "SGRO.L", "SBRY.L", "BKG.L", "BP.L", "HSX.L", "SGE.L",
     "RIO.L", "REL.L", "RKT.L", "PRU.L", "PSN.L", "NWG.L", "INF.L", "KGF.L", "PSON.L", "WPP.L", "WTB.L", "WEIR.L",
     "STJ.L", "SN.L", "SVT.L", "RTO.L", "NG.L", "LLOY.L", "LGEN.L", "LAND.L", "KGF.L", "JMAT.L", "INF.L", "IMB.L",
     "HSBA.L", "HLMA.L", "DGE.L", "DCC.L", "CRDA.L", "CPG.L", "CNA.L", "BNZL.L", "BP.L", "BKG.L", "BDEV.L", "BARC.L",
     "AZN.L", "ABF.L", "AHT.L", "ANTO.L", "III.L"})

# For later preprocessing to emulate CFD contracts
contract_size_100 = sorted(["NWG.L", "BARC.L", "III.L", "BA.L", "BP.L", "BDEV.L", "SBRY.L", "STAN.L",
                            "TSCO.L", "VOD.L", "RR.L", "GSK.L", "HSBA.L", "LGEN.L", "LLOY.L", "NXT.L", "PRU.L"])
contract_size_10 = sorted(list(set(tickers) - set(contract_size_100)))

four_dec_places = ["BP.L", "BARC.L", "HSBA.L", "LLOY.L", "TSCO.L", "VOD.L"]
three_dec_places = ["BA.L", "BDEV.L", "GSK.L", "LGEN.L", "PRU.L", "RR.L",
                    "SBRY.L", "III.L", "STAN.L", "RIO.L", "NG.L",
                    "WEIR.L", "SPX.L", "HSX.L", "HLMA.L", "NWG.L", "BKG.L",
                    "KGF.L", "CNA.L", "ANTO.L", "CPG.L", "REL.L", "BATS.L", "UTG.L", "JD.L", "DGE.L"]
two_dec_places = ["NXT.L", "AHT.L", "BNZL.L", "CRDA.L", "DCC.L", "INF.L", "PSON.L", "PSN.L", "RTO.L",
                  "SSE.L", "SGE.L", "UU.L", "SMIN.L", "SN.L", "SVT.L", "SMT.L", "RKT.L", "LAND.L",
                  "JMAT.L", "IMB.L", "ABF.L", "AAL.L", "WTB.L", "WPP.L", "SDR.L", "STJ.L", "AZN.L",
                  "SGRO.L"]

# MySQL Information
user = "your_user"
password = "your_password"
host = 'your_host'
db = 'av_dataframe'

ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_api_key"

# # Data Retrieval
# data_obj = ALPHA_VANTAGE_DATA(symbols=tickers, api_key=ALPHA_VANTAGE_API_KEY)
# data_obj.retrieve_data()
# combined_df = data_obj.combine_dataframes()
#
# # Manipulate MultiIndex for MySQL
# df = combined_df.dropna().swaplevel(0).reset_index()

# Store in MySQL
connection = pymysql.connect(host=host, user=user, password=password, db=db)
engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{db}')
# df.to_sql(name='av_ftse_stocks', con=engine, index=True, if_exists='replace')
df = pd.read_sql("SELECT * FROM av_ftse_stocks", connection).set_index(["Ticker", "Date"]).drop(columns="index")
connection.close()

# Change values for CFD Trading
df.loc[contract_size_100 + contract_size_10, ["Open", "High", "Low", "Close"]] /= 100
df.loc[contract_size_100 + contract_size_10, "Volume"] /= 10e5
df.loc[idx[four_dec_places, :], ["Open", "High", "Low", "Close"]] = df.loc[idx[four_dec_places, :], ["Open", "High", "Low", "Close"]].round(4)
df.loc[idx[three_dec_places, :], ["Open", "High", "Low", "Close"]] = df.loc[idx[three_dec_places, :], ["Open", "High", "Low", "Close"]].round(3)
df.loc[idx[two_dec_places, :], ["Open", "High", "Low", "Close"]] = df.loc[idx[two_dec_places, :], ["Open", "High", "Low", "Close"]].round(2)


# Engineer Features for Strategy
def atr(data):
    return ta.ATR(data.High, data.Low, data.Close, timeperiod=10)


def rolling_zscore(data):
    return (data - data.rolling(100).mean())/data.rolling(100).std()


def ema_50(data):
    return ta.EMA(data, timeperiod=50)


def ema_100(data):
    return ta.EMA(data, timeperiod=100)


def average_directional_index(data):
    return ta.ADX(data.High, data.Low, data.Close, timeperiod=10)


# Estimate Mid-Price
df["ATR"] = df.groupby("Ticker", group_keys=False).apply(atr)
df["Bid"] = df.Open
df["Ask"] = ((df.Open + df.ATR / 100) + (df.Bid * 1.005)) / 2
df["Mid_Price"] = (df.Bid + df.Ask) / 2

# Features for Trend Following Strategy
df["z_score"] = df.groupby("Ticker").Close.apply(rolling_zscore)
df["fast_ema"] = df.groupby("Ticker").Close.apply(ema_50)
df["slow_ema"] = df.groupby("Ticker").Close.apply(ema_100)
df["divergence"] = df["fast_ema"]/df["slow_ema"]
df["div_ema"] = df.groupby("Ticker").divergence.apply(ema_100)

# Mean Reversion Filtering Features
df["adx"] = df.groupby("Ticker", group_keys=False).apply(average_directional_index)
df["adx_ma"] = df.groupby("Ticker").adx.apply(ema_50)

# Changing up columns
df["Open"] = df.Mid_Price
df = df.drop(columns=["ATR", "Bid", "Ask","fast_ema", "slow_ema", "Mid_Price"])

df.index.names = ["Ticker", "datetime"]
df = df.dropna()

# close = df.loc[idx["BARC.L", :], ["Close", "Open", "High", "Low"]].droplevel(0).reset_index()
# close["rets"] = close.Close.pct_change()
# close["rets_t_1"] = close.rets.shift(1)
# close.dropna(inplace=True)
#
# cov = close.iloc[:, -2:].rolling(2).cov().drop(columns=["rets_t_1"]).drop('rets', level=1).values
# close["roll"] = 2 * np.sqrt(np.abs(cov))
# close["ask"] = close["Open"] + close.roll
#
# print(close[["Close", "Open", "High", "Low", "ask"]])


class CustomPandasData(PandasData):
    columns = df.columns[-5:]
    lines = tuple(columns)
    params = (
                 ('nullvalue', float('NaN')),
                 ('timeframe', bt.TimeFrame.Days),
                 ("dtformat", "%Y-%m-%d"),
                 ('datetime', 0),
                 ('open', 4),
                 ('high', 2),
                 ('low', 3),
                 ('close', 1),
                 ('volume', 5),
                 ('openinterest', None),
             ) + tuple([(i, int(j)) for i, j in
                        zip(columns,
                            list(np.ones(len(columns)) * -1))])


class TestStrategy(bt.Strategy):
    params = (
        ("exitbars", 5),
        ("stop_loss", 0.03),
        ("trail", False)
    )

    def log(self, txt):
        """ Logging function for this strategy"""
        dt = self.datas[0].datetime.date(0)
        print(f"{dt}, {txt}")  # print date and close

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries

        self.dataopen = self.datas[0].open
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low

        self.divergence = self.datas[0].divergence
        self.z_score = self.datas[0].z_score
        self.div_ema = self.datas[0].div_ema
        self.adx = self.datas[0].adx
        self.adx_ma = self.datas[0].adx_ma

        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.sellprice = None
        self.sellcomm = None
        self.opsize = None
        self.bar_executed = None

        self.log_pnl = []
        self.trades = []

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price:{order.executed.price}, Cost:{order.executed.value}, '
                    f'Comm:{order.executed.comm}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.opsize = order.executed.size

            elif order.issell():
                self.log(
                    f'SELL EXECUTED, Price:{order.executed.price}, Cost:{order.executed.value}, '
                    f'Comm:{order.executed.comm}')
                self.sellprice = order.executed.price
                self.sellcomm = order.executed.comm
                self.opsize = order.executed.size

            self.bar_executed = len(self)

        elif order.status in [order.Canceled]:
            self.log("Order Cancelled!")
            pass

        elif order.status in [order.Margin]:
            self.log("Order Margin!")
            pass

        elif order.status in [order.Rejected]:
            self.log("Order Rejected!")
            pass

        elif order.status in [order.Expired]:
            self.log("Order Expired!")
            pass

        if not order.alive():
            # Write down: no pending order
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'OPERATION PROFIT, GROSS: {trade.pnl}, NET: {trade.pnlcomm}')
            self.trades.append(trade.pnlcomm)

        elif trade.justopened:
            self.log('TRADE OPENED, SIZE %2d' % trade.size)

    def next(self):
        self.log(f"Open: {np.round(self.dataopen[0], 2)}, Close:  {np.round(self.dataclose[0], 2)}, "
                 f"High:  {np.round(self.datahigh[0], 2)}, Low:  {np.round(self.datalow[0], 2)}")

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # BUY LOGIC
            if self.divergence[-1] < self.div_ema[-1] and self.divergence[0] > self.div_ema[0]:
                self.log(f'BUY CREATE: {self.dataclose[0]}')
                self.buy(exectype=bt.Order.Market)

            # SELL LOGIC
            elif self.divergence[-1] > self.div_ema[-1] and self.divergence[0] < self.div_ema[0]:
                self.log(f'SELL CREATE: {self.dataclose[0]}')
                self.sell(exectype=bt.Order.Market)

        # EXIT MARKET
        else:
            # if in a buy position, create exit conditions and close position
            if self.position.size > 0:
                if self.divergence[-1] > self.div_ema[-1] and self.divergence[0] < self.div_ema[0]:  # exit after 5 time periods: len(self) >= (self.bar_executed + self.params.exitbars)
                    self.log(f'CLOSE CREATE {self.dataclose[0]}')
                    self.order = self.close(exectype=bt.Order.Market)

            # if in a sell position, create exit conditions and close position
            elif self.position.size < 0:
                if self.divergence[-1] < self.div_ema[-1] and self.divergence[0] > self.div_ema[0]:  # exit after 5 time period: len(self) >= (self.bar_executed + self.params.exitbars)
                    self.log(f'CLOSE CREATE {self.dataclose[0]}')
                    self.order = self.close(exectype=bt.Order.Market)

    def stop(self):
        self.log(f'Broker Ending Value {self.broker.getvalue()}')
        self.log(f"No. of Trades: {len(self.trades)}")
        unique_trades = np.unique(np.sign(self.trades), return_counts=True)[1]
        self.log(f"Win rate: {(unique_trades[1]/(unique_trades[0] + unique_trades[1]))  * 100}%")


class TestStrategy1(bt.Strategy):
    params = (
        ("exitbars", 5),
        ("stop_loss", 0.03),
        ("trail", False),
        ("account_risk", 0.01)
    )

    def log(self, txt):
        """ Logging function for this strategy"""
        dt = self.datas[0].datetime.date(0)
        if not self.stop_initiated:
            print(f"{dt}, {txt}")
        else:
            print(f"{txt}")

    def __init__(self):
        self.dataopen = self.datas[0].open
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low

        self.divergence = self.datas[0].divergence
        self.z_score = self.datas[0].z_score
        self.div_ema = self.datas[0].div_ema
        self.adx = self.datas[0].adx
        self.adx_ma = self.datas[0].adx_ma

        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.sellprice = None
        self.sellcomm = None
        self.opsize = None
        self.bar_executed = None

        self.start_portfolio_value = self.broker.getvalue()
        self.stop_initiated = False

        self.log_pnl = []
        self.trades = []

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price:{order.executed.price}, Cost:{order.executed.value}, '
                    f'Comm:{order.executed.comm}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.opsize = order.executed.size

            elif order.issell():
                self.log(
                    f'SELL EXECUTED, Price:{order.executed.price}, Cost:{order.executed.value}, '
                    f'Comm:{order.executed.comm}')
                self.sellprice = order.executed.price
                self.sellcomm = order.executed.comm
                self.opsize = order.executed.size

            self.bar_executed = len(self)

        elif order.status in [order.Canceled]:
            self.log("Order Cancelled!")
            pass

        elif order.status in [order.Margin]:
            self.log("Order Margin!")
            pass

        elif order.status in [order.Rejected]:
            self.log("Order Rejected!")
            pass

        elif order.status in [order.Expired]:
            self.log("Order Expired!")
            pass

        if not order.alive():
            # Write down: no pending order
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'OPERATION PROFIT, GROSS: {trade.pnl}, NET: {trade.pnlcomm}')
            self.log(f"BALANCE: {self.broker.getvalue():.2f}")
            self.trades.append(trade.pnlcomm)

        elif trade.justopened:
            self.log('TRADE OPENED, SIZE: %2d' % trade.size)

    def next(self):
        self.log(f"Open: {np.round(self.dataopen[0], 2)}, Close: {np.round(self.dataclose[0], 2)}, "
                 f"High: {np.round(self.datahigh[0], 2)}, Low: {np.round(self.datalow[0], 2)}, "
                 f"Equity: {self.broker.get_cash():,.2f}")

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position and len(self) != 1:

            # BUY LOGIC
            if self.divergence[0] > self.div_ema[0] and self.divergence[-1] < self.div_ema[-1]:
                self.log(f"BUY CREATE: {self.dataclose[0]}")
                self.log(f"BALANCE: {self.broker.getvalue():.2f}")
                self.buy(exectype=bt.Order.Market)

            # SELL LOGIC
            elif self.divergence[0] < self.div_ema[0] and self.divergence[-1] > self.div_ema[-1]:
                self.log(f'SELL CREATE: {self.dataclose[0]}')
                self.log(f"BALANCE: {self.broker.getvalue():.2f}")
                self.sell(exectype=bt.Order.Market)

        # EXIT MARKET
        else:
            # if in a buy position, create exit conditions and close position
            if self.position.size > 0:
                # exit after 5 time periods: len(self) >= (self.bar_executed + self.params.exitbars)
                if self.divergence[-1] > self.div_ema[-1] and self.divergence[0] < self.div_ema[0]:
                    self.log(f'CLOSE CREATE {self.dataclose[0]}')
                    self.order = self.close(exectype=bt.Order.Market)

            # if in a sell position, create exit conditions and close position
            # exit after 5 time period: len(self) >= (self.bar_executed + self.params.exitbars)
            elif self.position.size < 0:
                if self.divergence[-1] < self.div_ema[-1] and self.divergence[0] > self.div_ema[0]:
                    self.log(f'CLOSE CREATE {self.dataclose[0]}')
                    self.order = self.close(exectype=bt.Order.Market)

    def stop(self):
        self.stop_initiated = True
        self.log(f"")
        self.log(f"Starting Portfolio Value: {self.start_portfolio_value:,.2f}")
        self.log(f"Final Portfolio Value {self.broker.getvalue():,.2f}")
        self.log(f"PnL: {self.broker.getvalue() - self.start_portfolio_value:,.2f}")
        self.log(f"Broker Value (with open trade): {self.broker.get_cash():,.2f}")
        self.log(f"No. of Trades: {len(self.trades)}")
        unique_trades = np.unique(np.sign(self.trades), return_counts=True)[1]
        self.log(f"Win rate: {(unique_trades[1]/(unique_trades[0] + unique_trades[1]))  * 100:.2f}%")


cerebro = bt.Cerebro()

data = df.loc[idx["BARC.L", :], :].droplevel(0).reset_index()
feed = CustomPandasData(dataname=data, name="BARC.L")
cerebro.adddata(feed)

cerebro.addstrategy(TestStrategy1)

cerebro.broker.setcash(500_000.0)

cerebro.addsizer(bt.sizers.SizerFix, stake=5555)

results = cerebro.run(maxcpus=10, stdstats=True)

cerebro.plot()





