import backtrader as bt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import talib as ta
import quantstats
import math
from typing import Union

sns.set_style("darkgrid")
pd.set_option("display.max_columns", None)


class MT5_Trading_Platform:
    """
    This Class connects us to a broker in order to retrieve
    account data and can access OHLC and Tick Data (Bid/Ask).
    :param account_number: this is the account number/code
    :param account_password: password to access the account
    :param server_name: server name is the name of the broker/account
    """

    def __init__(self, account_number: int, account_password: str, server_name: str):
        self.ac_number = account_number
        self.ac_password = account_password
        self.server = server_name
        self.symbol = str
        self.timeframe = None
        self.first_datetime_index = datetime
        self.end_datetime_index = datetime

    @staticmethod
    def _establish_connection():
        """
        This method attempt to initialize a
        connection to the MT5 trading platform.
        """
        print("Establishing a connection to MetaTrader5...")

        if not mt5.initialize():
            print("initialize() method failed, error code =", mt5.last_error())
            quit()

        print("Initialization to MetaTrader5 successful!")
        print("\nMetaTrader5 package author: ", mt5.__author__)
        print("MetaTrader5 package version: ", mt5.__version__)

    def _login(self):
        """
        This method will use your account information
        to login into your specific account.
        """
        try:
            self.login = mt5.login(login=self.ac_number,
                                   password=self.ac_password,
                                   server=self.server)
        except ConnectionError as e:
            print(f"{e}")
            print("Invalid account information! Please ensure you enter"
                  "the correct login details.")
            print("It is possible your account no longer exists if you"
                  "are using a demo account.")

    @staticmethod
    def _shutdown_connection():
        """
        This method will shut down the connection
        with the MetaTrader5 platform.
        """
        mt5.shutdown()

    def _authorizing_account(self):
        """
        This method confirms the connection to the trading platform and
        broker you are using. It also provides your account information.
        """
        if self.login:
            print("\n-------------------------------------------")
            print(f"Connected to account MT5 Client #{self.ac_number}")
            print("\n-------------------------------------------")
            print("Account Information: ")

            account_info_dict = mt5.account_info()._asdict()
            for prop in account_info_dict:
                print(f"- {prop} = {account_info_dict[prop]}")
            print("-------------------------------------------")
        else:
            print(f"Failed to connect at account #{self.ac_number}, error code: {mt5.last_error()}")

    @staticmethod
    def _terminal_information():
        """
        This method will obtain the terminal information for the user.
        """
        terminal_info = mt5.terminal_info()
        if terminal_info != None:
            print("\n-------------------------------------------")
            print("Terminal Information")
            terminal_info_dict = mt5.terminal_info()._asdict()
            for prop in terminal_info_dict:
                print(f"- {prop} = {terminal_info_dict[prop]}")
            print("-------------------------------------------")

    @staticmethod
    def _number_symbols():
        """
        This method reveals the number of tradable symbols
        """
        symbols = mt5.symbols_total()
        if symbols > 0:
            print("\nTotal symbols =", symbols)
        else:
            print("\nSymbols not found!")

    @staticmethod
    def _get_tradable_symbols():
        """
        This method retrieves all available symbols.
        """
        # get all symbols
        symbols = mt5.symbols_get()
        print("\nTradable symbols: ")
        print(np.array([s.name for s in symbols]))

    def _get_specific_symbol_info(self, symbol: str):
        """
        This method will obtain the information for a particular symbol.
        :param symbol: symbol of the financial instrument
        """
        try:
            self.symbol = symbol
            get_symbol_info = mt5.symbol_info(symbol)
            if get_symbol_info != None:
                print(f"\n{symbol}: spread =", get_symbol_info.spread,
                      ",  digits =", get_symbol_info.digits)

                symbol_info_dict = get_symbol_info._asdict()
                for prop in symbol_info_dict:
                    print(f"- {prop} = {symbol_info_dict[prop]}")
            else:
                raise ConnectionError("Cannot retrieve symbol information.")

        except ConnectionError as e:
            print(e)

    def _set_timeframe(self, timeframe):
        # Set the timeframe
        self.timeframe = timeframe

    def _get_symbol_data(self):
        """
        This method retrieves OHLC, Timestamp, Tick_Volume and Spread.
        :param time-frame: input a MT5 TimeFrame, ranging from 1 minute to 1 month.
        :return: rates dataframe.
        """
        global rates
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0,
                                            99_999)  # 0 is beginning index, 99_999 is max obvs.
            rates = pd.DataFrame(rates)
            rates.index = pd.to_datetime(rates["time"], unit="s") - timedelta(hours=2)  # time discrepancy resolved.
            rates.columns = ["Timestamp", "Open", "High", "Low", "Close", "Volume", "Spread", "Real_Volume"]
            rates.index.name = "Datetime"
            rates = rates.iloc[:-1, :]
            rates.drop(columns=["Real_Volume", "Timestamp"], inplace=True)
            self.first_datetime_index = rates.index[0]
            self.end_datetime_index = rates.index[-1]

        except ImportError as e:
            print(e, ": Not working as the MT5 Platform is not running. It must be open to import this data.")

        return rates

    def _get_symbol_ticks(self):
        """
        This method returns tick data.
        :return: tick dataframe.
        """
        global ticks
        try:
            ticks = mt5.copy_ticks_from(self.symbol, self.first_datetime_index, 20_000_000, mt5.COPY_TICKS_ALL)
            ticks = pd.DataFrame(ticks)
            ticks.index = pd.to_datetime(ticks["time"], unit="s") - timedelta(hours=2)

            ticks = ticks.loc[self.first_datetime_index:self.end_datetime_index, ["bid", "ask"]]
            ticks.columns = ["Bid", "Ask"]
            conversion = {"Bid": "min",  # first
                          "Ask": "max"}  # note you're taking the first value of 1 min bid/ask, could take min and max or time ordered (nth)
            ticks = ticks.resample("min").agg(conversion, axis=1)
            ticks["Bid_Ask"] = ticks.Ask - ticks.Bid

        except ImportError as e:
            print(e, ": Not working as the MT5 Platform is not running. It must be open to import this data.")

        return ticks

    def _get_symbol_data_from_MT5(self, remove_erratic_bidask_times=True):
        """
        This method calls upon the get rates and ticks methods to
        merge the data into a singular dataframe.
        :return: tick and rates dataframe.
        """
        rates = self._get_symbol_data()
        ticks = self._get_symbol_ticks()
        df = rates.merge(ticks, how="inner", left_index=True, right_index=True)

        if remove_erratic_bidask_times == True:
            df = df.loc[np.array([str(i) for i in df.index if not "21:58" < str(i)[11:16] <= "23:01"]), :]

        return df


def keep_OHLCV_dataframe(data):
    data = data.drop(columns=["Bid", "Ask", "Spread", "Bid_Ask"])

    return data


def create_signal(data):
    slow_ma = ta.SMA(data.Close, timeperiod=20)
    fast_ma = ta.SMA(data.Close, timeperiod=10)
    rsi = ta.RSI(data.Close, timeperiod=8)
    data["slow_ma"] = slow_ma
    data["fast_ma"] = fast_ma
    data["rsi"] = rsi
    data["Signal"] = 0
    for i in range(len(data)):
        if (
                (fast_ma[i] > slow_ma[i]) and
                (fast_ma[i - 1] < slow_ma[i - 1]) and
                (rsi[i] > 30.0 > rsi[i - 1])
        ):
            data.iloc[i, -1] = 1

        elif (
                (fast_ma[i] < slow_ma[i]) and
                (fast_ma[i - 1] > slow_ma[i - 1]) and
                (rsi[i] < 70.0 < rsi[i - 1])
        ):
            data.iloc[i, -1] = -1

    return data


def preprocess_dataframe_for_backtest(data):
    dataframe = data.copy()
    dataframe["Open"] = data.Bid
    dataframe["Close"] = data.Ask
    dataframe.dropna(inplace=True)

    return dataframe


def manipulate_bid_with_ask(data):
    for i in range(len(data)):
        try:
            if data.iloc[i, -1] == 1:
                data.iloc[i, 0] = data.iloc[i, 3]

            elif data.iloc[i, -1] == -1:
                data.iloc[i + 7, 0] = data.iloc[i + 7, 3]
        except:
            continue

    return data


class CustomDataFrameSignal(bt.feeds.PandasData):
    lines = ("Signal",)

    params = (
        ('nullvalue', float('NaN')),
        ('dtformat', '%Y-%m-%d %H:%M:%S'),
        ('tmformat', '%H:%M:%S'),
        ("timeframe", bt.TimeFrame.Minutes),
        ("compression", 1),
        ("datetime", None),
        ("time", None),
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
        ("openinterest", None),
        ("Signal", -1),
    )


class MyBuySell(bt.observers.BuySell):
    plotlines = dict(
        buy=dict(marker="^", markersize=10.0, color="blue", fillstyle="full",
                 ls="", markeredgecolor="black", linewidth=0.5),
        sell=dict(marker="v", markersize=10.0, color="maroon", fillstyle="full",
                  ls="", markeredgecolor="black", linewidth=0.5),
    )


class MyTrades(bt.observers.Trades):
    plotlines = dict(
        pnlplus=dict(_name='Positive',
                     ls='', marker='x', color='blue',
                     markersize=4.0, fillstyle='full'),
        pnlminus=dict(_name='Negative',
                      ls='', marker='x', color='red',
                      markersize=4.0, fillstyle='full'),
    )


class BidLine(bt.Indicator):
    lines = ("bidline",)
    plotinfo = dict(bidline=dict(plot=True, subplot=False))
    plotlines = dict(bidline=dict(ls="solid", color="darkblue"))

    def __init__(self):
        self.lines.bidline = self.data.Bid


class AskLine(bt.Indicator):
    lines = ("askline",)
    plotinfo = dict(askline=dict(plot=True, subplot=False))
    plotlines = dict(askline=dict(ls="solid", color="orangered"))

    def __init__(self):
        self.lines.askline = self.data.Ask


class FixedAmountSizer(bt.Sizer):
    params = (('size', 1),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        return self.p.size


class MaxRiskToleranceSizer(bt.Sizer):
    """
    Returns the number of shares rounded down that can be purchased for the
    max risk tolerance
    """
    params = (('risk', 0.02),)

    def __init__(self):
        if self.p.risk > 1 or self.p.risk < 0:
            raise ValueError('The risk parameter is a percentage which must be'
                             'entered as a float. e.g. 0.5')

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy == True:
            size = math.floor((cash * self.p.risk) / data[0])
        else:
            size = math.floor((cash * self.p.risk) / data[0]) * -1
        return size


def printTradeAnalysis(analyzer):
    """
    Function to print the Technical Analysis results in a nice format.
    """
    # Get the results we are interested in
    total_open = analyzer.total.open
    total_closed = analyzer.total.closed
    total_won = analyzer.won.total
    total_lost = analyzer.lost.total
    win_streak = analyzer.streak.won.longest
    lose_streak = analyzer.streak.lost.longest
    pnl_net = round(analyzer.pnl.net.total, 2)
    strike_rate = (total_won / total_closed) * 100
    # Designate the rows
    h1 = ['Total Open', 'Total Closed', 'Total Won', 'Total Lost']
    h2 = ['Hit Rate', 'Win Streak', 'Losing Streak', 'PnL Net']
    r1 = [total_open, total_closed, total_won, total_lost]
    r2 = [f"{round(strike_rate, 3)}%", win_streak, lose_streak, pnl_net]
    # Check which set of headers is the longest.
    if len(h1) > len(h2):
        header_length = len(h1)
    else:
        header_length = len(h2)
    # Print the rows
    print_list = [h1, r1, h2, r2]
    row_format = "{:<15}" * (header_length + 1)
    print("Trade Analysis Results:")
    for row in print_list:
        print(row_format.format('', *row))


def printSQN(analyzer):
    sqn = round(analyzer.sqn, 2)
    print('SQN: {}'.format(sqn))


def printSharpeRatio(returns: Union[np.array, pd.Series], rf=0.0):
    print("Sharpe Ratio: ", round((np.mean(returns) - rf) / np.std(returns), 2))


def printInformationCoefficient(analyzer):
    total_won = analyzer.won.total
    total_lost = analyzer.lost.total
    total = total_won + total_lost
    print("Information Coefficient: ", round((2 * total_won / total) - 1, 4))


if __name__ == "__main__":

    df = pd.read_pickle("gbp_chf.pkl")

    ohlcv_df = keep_OHLCV_dataframe(data=df)  # for purposes of charting

    df = create_signal(data=df)  # generate signal using predefined indicators
    df = preprocess_dataframe_for_backtest(data=df)  # simple preprocessing, open/close become bid/ask
    data0 = manipulate_bid_with_ask(data=df)  # position the ask price at the bid to model spread

    print(data0)

    # Create a Strategy
    class TestStrategy(bt.Strategy):

        def log(self, txt, dt=None):
            """Logging function for this strategy"""
            dt = dt or self.datas[0].datetime.datetime(0)
            print(f"{str(dt)},   {str(txt)}")

        def __init__(self):
            # Keep a reference to our columns
            self.bid = self.datas[0].open
            self.ask = self.datas[0].close
            self.signal = self.datas[0].Signal
            self.holding_period = 5
            self.bar_executed = None
            # self.entered = None

            # Order variable will contain ongoing order details/status
            self.order = None

        def notify_order(self, order):
            if order.status in [order.Submitted, order.Accepted]:
                # An active Buy/Sell order has been submitted/accepted - Nothing to do
                return

            if order.status in [order.Expired]:
                self.log('ORDER EXPIRED')

            # Check if an order has been completed
            # Attention: broker could reject order if not enough cash
            if order.status in [order.Completed]:
                if order.isbuy():
                    self.log(f"BUY EXECUTED, Price: {order.executed.price:.5f}, "
                             f"Cost: {order.executed.value:.5f}, Commission: {order.executed.comm:.2f}")
                    self.log(f"MARGIN ------------------- {order.executed.margin}")
                    self.log(f"SIZE ------------------ {order.executed.size}")

                    self.buyprice = order.executed.price
                    self.buycomm = order.executed.comm
                    self.buyopsize = order.executed.size

                else:
                    self.log(f"SELL EXECUTED, Price: {order.executed.price:.5f}, "
                             f"Cost: {order.executed.value:.5f}, Commission: {order.executed.comm:.2f}")
                    self.log(f"MARGIN ------------------- {order.executed.margin}")
                    self.log(f"SIZE ------------------ {order.executed.size}")

                    self.sellprice = order.executed.price
                    self.sellcomm = order.executed.comm
                    self.sellopsize = order.executed.size

                    # gross_pnl = (order.executed.price - self.buyprice) * \
                    #             self.opsize
                    #
                    # if margin:
                    #     gross_pnl *= mult
                    #
                    # net_pnl = gross_pnl - self.buycomm - order.executed.comm
                    # self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                    #          (gross_pnl, net_pnl))


                # This variable is useful for when closing trades
                self.bar_executed = len(self)

            elif order.status in [order.Canceled]:
                self.log("Order Canceled.")
            elif order.status in [order.Margin]:
                self.log("Margin issue! Not enough cash.")
            elif order.status in [order.Rejected]:
                self.log("Order Rejected")

            # Reset orders
            self.order = None

        def notify_trade(self, trade):
            if trade.justopened:
                print("----TRADE OPENED----")
                print(f"Size: {trade.size}")
            elif trade.isclosed:
                print("----TRADE CLOSED----")
                print(f"PnL: Gross: {round(trade.pnl, 2)} / Net: {round(trade.pnlcomm, 2)}\n")
            else:
                return

        # def notify_cashvalue(self, cash, value):
        #     self.log(f"CASH: {cash}, PORTFOLIO: {value}")

        def next(self):

            self.log(f"Bid: {self.bid[0]:.5f}, Ask: {self.ask[0]:.5f}, Signal: {self.signal[0]}")

            # Check for open orders - if open then do nothing, only want to be in one position at a time
            if self.order:
                return

            # Check if we are in the market
            if not self.position:
                if self.signal[0] == 1:
                    self.log(f'BUY SIGNAL CREATED {self.ask[0]:.5f}')
                    self.datas[0].open[1] = self.datas[0].close[1]
                    self.order = self.buy(data=self.datas[0], size=None, exectype=bt.Order.Market, valid=None)

                elif self.signal[0] == -1:
                    self.log(f'SELL SIGNAL CREATED {self.bid[0]:.5f}')
                    self.order = self.sell(data=self.datas[0], size=None, exectype=bt.Order.Market, valid=None)

            else:
                if len(self) >= (self.bar_executed + self.holding_period):

                    if self.position.size > 0:
                        self.log(f'CLOSE SIGNAL CREATED {self.bid[0]:.5f}')
                        self.datas[0].open[1] = self.datas[0].close[1]
                        self.order = self.close(data=self.datas[0], size=None, exectype=bt.Order.Market, valid=None)

                    elif self.position.size < 0:
                        self.log(f'CLOSE SIGNAL CREATED {self.ask[0]:.5f}')
                        self.order = self.close(data=self.datas[0], size=None, exectype=bt.Order.Market, valid=None)

        def start(self):
            print('Backtesting is about to start...\n')

        def stop(self):
            print('\nBacktesting is finished!')


    # ----- Instantiate Cerebro ----- #
    cerebro = bt.Cerebro()

    # ----- Broker ----- #
    # broker = MyBroker()
    # cerebro.broker = broker
    initial_capital = 10_000.00
    commission_scheme = 0.0000
    leverage = 30.0
    multiplier = 100_000.00

    cerebro.broker.setcash(cash=initial_capital)

    # class ForexCommissionScheme(bt.CommissionInfo):
    #
    #     params = (
    #         ("commission", 0),
    #         ("leverage", 30),
    #         ("")
    #     )

    class ForexSizer():
            pass


    class FixedSize(bt.Sizer):
        params = (('stake', 1),)

        def _getsizing(self, comminfo, cash, data, isbuy):
            return self.params.stake


    class LongOnly(bt.Sizer):
        params = (('stake', 1),)

        def _getsizing(self, comminfo, cash, data, isbuy):
            if isbuy:
                return self.p.stake

            # Sell situation
            position = self.broker.getposition(data)
            if not position.size:
                return 0  # do not sell if nothing is open

            return self.p.stake


    class ForexCommissionScheme(bt.comms.CommInfoBase):

        params = (
            ("stocklike", False),
            ("JPY_pair", False),
            ("a/c_curr_quote_curr", False),
            ("commtype", bt.comms.CommInfoBase.COMM_FIXED)
        )

        def _getcommission(self, size, price, pseudoexec):

            return size * price * self.p.commission * self.p.mult



    forex_commision_scheme = bt.CommissionInfo(commission=commission_scheme, leverage=leverage, mult=multiplier, margin=leverage/multiplier, stocklike=False)
    cerebro.broker.addcommissioninfo(comminfo=forex_commision_scheme)

    # ----- Sizer ----- #
    # cerebro.addsizer(MaxRiskToleranceSizer, risk=0.02)  ########################################################################################################

    # ----- Data ----- #
    dataframe0 = CustomDataFrameSignal(dataname=data0)
    cerebro.adddata(data=dataframe0)

    # ----- Strategy ----- #
    cerebro.addstrategy(TestStrategy)  # here can change the params=() part

    # ----- Analyzers ----- #
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="Pyfolio")
    cerebro.addanalyzer(bt.analyzers.SQN, _name="SQN")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")

    # ----- Writer ----- #
    cerebro.addwriter(bt.WriterFile, csv=True, out='log.csv')

    # ----- Observer ----- #
    cerebro.addobserver(MyBuySell, barplot=True, bardist=0.0001)  # buy and sell locations
    cerebro.addobserver(bt.observers.Broker)
    cerebro.addobserver(MyTrades, pnlcomm=True)
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.Value)

    # ----- Perform Backtest ----- #
    # Run Cerebro Engine
    start_portfolio_value = cerebro.broker.getvalue()

    results = cerebro.run(writer=False, runonce=False, stdstats=False)

    end_portfolio_value = cerebro.broker.getvalue()
    pnl = end_portfolio_value - start_portfolio_value
    print(f'\nStarting Portfolio Value: {start_portfolio_value:2f}')
    print(f'Final Portfolio Value: {end_portfolio_value:2f}')
    print(f'PnL: {pnl:.2f}')

    # ----- Plotting ----- #
    cerebro.plot(style="candlestick", barup="lime", bardown="red", volup="seagreen", voldown="red", hlinescolor="black",
                 plotdist=0.1, linevalues=False, valuetags=False, start=0, end=len(df))

    # ----- Results ----- #
    strat = results[0]

    portfolio_stats = strat.analyzers.getbyname("Pyfolio")
    returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
    returns.index = returns.index.tz_convert(None)

    plt.plot(np.cumprod(1 + returns))
    plt.show()

    print("-----------------------------------------------------------------")
    print(positions)
    print("-----------------------------------------------------------------")
    print(transactions)
    print("")

    printSharpeRatio(returns=returns, rf=0.0)
    printSQN(strat.analyzers.SQN.get_analysis())
    printInformationCoefficient(strat.analyzers.ta.get_analysis())
    printTradeAnalysis(strat.analyzers.ta.get_analysis())
