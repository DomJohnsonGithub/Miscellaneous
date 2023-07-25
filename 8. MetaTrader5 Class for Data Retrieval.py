import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from itertools import permutations
from typing import List

# Two Variations

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

    def _set_timeframe(self, timeframe):
        # Set the timeframe
        self.timeframe = timeframe

    @staticmethod
    def cut_df_start(df, n_obs):
        n_rows = df.shape[0]
        final_df = df.iloc[n_rows - n_obs:]
        return final_df

    def _get_rates_and_tick_data(self, symbol, remove_erratic_bidask_times):
        """
        This function imports data from MT5 for a specific trade-able instrument.
        :param symbol: symbol of the instrument to get data on
        :param remove_erratic_bidask_times: removes erratic data
        :return: dataframe with OHLCV and BidAsk data
        """
        try:
            # Rates data manipulation
            rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0,
                                            99_999)  # 0 is beginning index, 99_999 is max obvs.
            rates = pd.DataFrame(rates)
            rates["time"] = pd.to_datetime(rates["time"], unit="s") - timedelta(hours=2)  # time discrepancy resolved.
            rates = rates.set_index("time")
            rates.columns = ["Open", "High", "Low", "Close", "Volume", "Spread", "Real_Volume"]
            rates.index.name = "Datetime"
            rates.drop(columns=["Real_Volume"], inplace=True)
            rates = self.cut_df_start(rates, 45_000)
            self.first_datetime_index = rates.index[0]
            self.end_datetime_index = rates.index[-1]

            # Tick data manipulation
            # ticks = mt5.copy_ticks_from(symbol, self.first_datetime_index, 30_000_000, mt5.COPY_TICKS_ALL)
            ticks = mt5.copy_ticks_range(symbol, self.first_datetime_index, self.end_datetime_index, mt5.COPY_TICKS_ALL)
            ticks = pd.DataFrame(ticks)

            if ticks.empty:
                print("\nEmpty Tick Dataframe!")

            if not ticks.empty:
                ticks.index = pd.to_datetime(ticks["time"], unit="s") - timedelta(hours=2)
                ticks = ticks.loc[self.first_datetime_index:self.end_datetime_index, ["bid", "ask"]]
                ticks.columns = ["Bid", "Ask"]

                bid_ask = ticks.copy()
                all_ticks = []
                ticks = pl.DataFrame(ticks)
                ticks = ticks.with_column(pl.Series(name="time", values=bid_ask.index))
                for i in ["mean", "median", ["min", "max"], [*set(list(permutations([0.25, 0.25, 0.75, 0.75], r=2)))]]:
                    if isinstance(i, str):
                        if i == "mean":
                            data0 = ticks.groupby_dynamic("time", every=f"{self.timeframe}m").agg([pl.col("Bid").mean(),
                                                                                                   pl.col(
                                                                                                       "Ask").mean()])
                            bid = pl.Series(np.round(data0["Bid"].to_numpy(), 5))
                            ask = pl.Series(np.round(data0["Ask"].to_numpy(), 5))
                            data0 = data0.with_columns([pl.Series(name=f"Bid_{i}", values=bid)])
                            data0 = data0.with_column(pl.Series(name=f"Ask_{i}", values=ask)).drop(["Bid", "Ask"])
                            data0 = data0.with_column(
                                pl.Series(name=f"Bid_Ask_{i}", values=data0[f"Ask_{i}"] - data0[f"Bid_{i}"]))
                            all_ticks.append(data0.to_pandas().set_index(keys="time"))
                        else:
                            data1 = ticks.groupby_dynamic("time", every=f"{self.timeframe}m").agg(
                                [pl.col("Bid").median(),
                                 pl.col("Ask").median()])
                            bid = pl.Series(np.round(data1["Bid"].to_numpy(), 5))
                            ask = pl.Series(np.round(data1["Ask"].to_numpy(), 5))
                            data1 = data1.with_column(pl.Series(name=f"Bid_{i}", values=bid))
                            data1 = data1.with_column(pl.Series(name=f"Ask_{i}", values=ask)).drop(["Bid", "Ask"])
                            data1 = data1.with_column(
                                pl.Series(name=f"Bid_Ask_{i}", values=data1[f"Ask_{i}"] - data1[f"Bid_{i}"]))
                            all_ticks.append(data1.to_pandas().set_index(keys="time"))
                    elif isinstance(i, list):
                        if type(i[0]) is str:
                            data2 = ticks.groupby_dynamic("time", every=f"{self.timeframe}m").agg([pl.col("Bid").min(),
                                                                                                   pl.col("Ask").max()])
                            bid = pl.Series(np.round(data2["Bid"].to_numpy(), 5))
                            ask = pl.Series(np.round(data2["Ask"].to_numpy(), 5))
                            data2 = data2.with_column(pl.Series(name=f"Bid_{i[0]}", values=bid))
                            data2 = data2.with_column(pl.Series(name=f"Ask_{i[1]}", values=ask)).drop(["Bid", "Ask"])
                            data2 = data2.with_column(
                                pl.Series(name=f"Bid_Ask_{i[0]}{i[1]}",
                                          values=data2[f"Ask_{i[1]}"] - data2[f"Bid_{i[0]}"]))
                            all_ticks.append(data2.to_pandas().set_index(keys="time"))
                        else:
                            for j in i:
                                data3 = ticks.groupby_dynamic("time", every=f"{self.timeframe}m").agg(
                                    [pl.col("Bid").quantile(j[0]),
                                     pl.col("Ask").quantile(j[1])])
                                bid = pl.Series(np.round(data3["Bid"].to_numpy(), 5))
                                ask = pl.Series(np.round(data3["Ask"].to_numpy(), 5))
                                data3 = data3.with_column(pl.Series(name=f"Bid_quantile({j[0]})", values=bid))
                                data3 = data3.with_column(pl.Series(name=f"Ask_quantile({j[1]})", values=ask)).drop(
                                    ["Bid", "Ask"])
                                data3 = data3.with_column(pl.Series(name=f"Bid_Ask_quantile({j[0]})({j[1]})",
                                                                    values=data3[f"Ask_quantile({j[1]})"] - data3[
                                                                        f"Bid_quantile({j[0]})"]))
                                all_ticks.append(data3.to_pandas().set_index(keys="time"))

                ticks = pd.concat(all_ticks, ignore_index=False, axis=1)
                ticks = ticks.loc[:, ~ticks.columns.duplicated()].copy()
                ticks["avg_bid"] = ticks.loc[:, ["Bid_mean", "Bid_median", "Bid_min",
                                                 "Bid_quantile(0.25)", "Bid_quantile(0.75)"]].mean(axis=1)
                ticks["avg_ask"] = ticks.loc[:, ["Ask_mean", "Ask_median", "Ask_max",
                                                 "Ask_quantile(0.25)", "Ask_quantile(0.75)"]].mean(axis=1)

                df = rates.merge(ticks, how="inner", left_index=True, right_index=True)

                if remove_erratic_bidask_times:
                    df = df.loc[np.array([str(i) for i in df.index if not "21:45" < str(i)[11:16] <= "00:00"]), :]

                return df

            else:
                return rates

        except ImportError as e:
            print(e, ": Not working as the MT5 Platform is not running. It must be open to import this data.")


class MT5_Trading_Platform_Handler:

    def __init__(self, data_frequency, account_number, account_password, server, groupby):
        """
        This Class connects us to a broker in order to retrieve
        account data and can access OHLC and Tick Data (Bid/Ask).
        :param account_number: this is the account number/code
        :param account_password: password to access the account
        :param server: server name is the name of the broker/account
        """
        self.account_number = account_number
        self.account_password = account_password
        self.server = server
        self.data_freq = data_frequency
        self.data_frames = {}
        self.groupby = groupby

    @staticmethod
    def _cut_df_start(df: pd.DataFrame, n_obs: int):
        n_rows = df.shape[0]
        final_df = df.iloc[n_rows - n_obs:]
        return final_df

    def retrieve_data(self, ticker: str):
        """
        This function imports data from MT5 for a specific trade-able instrument.
        :param ticker: symbol of the instrument to get data on
        :return: dataframe with OHLCV and BidAsk data
        """
        try:
            # Rates data manipulation
            rates = mt5.copy_rates_from_pos(ticker, self.data_freq, 0,
                                            99_999)  # 0 is beginning index, 99_999 is max obvs.
            rates = pd.DataFrame(rates)
            if self.data_freq == mt5.TIMEFRAME_D1:
                rates["time"] = pd.to_datetime(rates["time"], unit="s")
            else:
                rates["time"] = pd.to_datetime(rates["time"], unit="s") - timedelta(hours=2) # time discrepancy resolved.

            rates = rates.set_index("time")
            rates.columns = ["Open", "High", "Low", "Close", "Volume", "Spread", "Real_Volume"]
            rates.index.name = "Datetime"
            rates.drop(columns=["Real_Volume"], inplace=True)
            rates = self._cut_df_start(rates, 45_000)
            first_datetime_index = rates.index[0]
            end_datetime_index = rates.index[-1]

            # Tick data manipulation
            # ticks = mt5.copy_ticks_from(ticker, self.first_datetime_index, 30_000_000, mt5.COPY_TICKS_ALL)
            ticks = mt5.copy_ticks_range(ticker, first_datetime_index, end_datetime_index, mt5.COPY_TICKS_ALL)
            ticks = pd.DataFrame(ticks)

            if ticks.empty:
                print("\nEmpty Tick Dataframe!")

            if not ticks.empty:
                ticks.index = pd.to_datetime(ticks["time"], unit="s") - timedelta(hours=2)
                ticks = ticks.loc[first_datetime_index:end_datetime_index, ["bid", "ask"]]
                ticks.columns = ["Bid", "Ask"]

                bid_ask = ticks.copy()
                all_ticks = []
                ticks = pl.DataFrame(ticks).with_columns(pl.Series(name="time", values=bid_ask.index))
                for i in ["mean", "median", ["min", "max"], [*set(list(permutations([0.25, 0.25, 0.75, 0.75], r=2)))]]:
                    if isinstance(i, str):
                        if i == "mean":
                            data0 = ticks.groupby_dynamic("time", every=f"{self.groupby}").agg([pl.col("Bid").mean(),
                                                                                                   pl.col("Ask").mean()])
                            bid = pl.Series(np.round(data0["Bid"].to_numpy(), 5))
                            ask = pl.Series(np.round(data0["Ask"].to_numpy(), 5))
                            data0 = data0.with_columns([pl.Series(name=f"Bid_{i}", values=bid)])
                            data0 = data0.with_columns(pl.Series(name=f"Ask_{i}", values=ask)).drop(["Bid", "Ask"])
                            data0 = data0.with_columns(
                                pl.Series(name=f"Bid_Ask_{i}", values=data0[f"Ask_{i}"] - data0[f"Bid_{i}"]))
                            all_ticks.append(data0.to_pandas().set_index(keys="time"))
                        else:
                            data1 = ticks.groupby_dynamic("time", every=f"{self.groupby}").agg(
                                [pl.col("Bid").median(),
                                 pl.col("Ask").median()])
                            bid = pl.Series(np.round(data1["Bid"].to_numpy(), 5))
                            ask = pl.Series(np.round(data1["Ask"].to_numpy(), 5))
                            data1 = data1.with_columns(pl.Series(name=f"Bid_{i}", values=bid))
                            data1 = data1.with_columns(pl.Series(name=f"Ask_{i}", values=ask)).drop(["Bid", "Ask"])
                            data1 = data1.with_columns(
                                pl.Series(name=f"Bid_Ask_{i}", values=data1[f"Ask_{i}"] - data1[f"Bid_{i}"]))
                            all_ticks.append(data1.to_pandas().set_index(keys="time"))
                    elif isinstance(i, list):
                        if type(i[0]) is str:
                            data2 = ticks.groupby_dynamic("time", every=f"{self.groupby}").agg([pl.col("Bid").min(),
                                                                                                   pl.col("Ask").max()])
                            bid = pl.Series(np.round(data2["Bid"].to_numpy(), 5))
                            ask = pl.Series(np.round(data2["Ask"].to_numpy(), 5))
                            data2 = data2.with_columns(pl.Series(name=f"Bid_{i[0]}", values=bid))
                            data2 = data2.with_columns(pl.Series(name=f"Ask_{i[1]}", values=ask)).drop(["Bid", "Ask"])
                            data2 = data2.with_columns(
                                pl.Series(name=f"Bid_Ask_{i[0]}{i[1]}",
                                          values=data2[f"Ask_{i[1]}"] - data2[f"Bid_{i[0]}"]))
                            all_ticks.append(data2.to_pandas().set_index(keys="time"))
                        else:
                            for j in i:
                                data3 = ticks.groupby_dynamic("time", every=f"{self.groupby}").agg(
                                    [pl.col("Bid").quantile(j[0]),
                                     pl.col("Ask").quantile(j[1])])
                                bid = pl.Series(np.round(data3["Bid"].to_numpy(), 5))
                                ask = pl.Series(np.round(data3["Ask"].to_numpy(), 5))
                                data3 = data3.with_columns(pl.Series(name=f"Bid_quantile({j[0]})", values=bid))
                                data3 = data3.with_columns(pl.Series(name=f"Ask_quantile({j[1]})", values=ask)).drop(
                                    ["Bid", "Ask"])
                                data3 = data3.with_columns(pl.Series(name=f"Bid_Ask_quantile({j[0]})({j[1]})",
                                                                     values=data3[f"Ask_quantile({j[1]})"] - data3[
                                                                        f"Bid_quantile({j[0]})"]))
                                all_ticks.append(data3.to_pandas().set_index(keys="time"))

                ticks = pd.concat(all_ticks, ignore_index=False, axis=1)
                ticks = ticks.loc[:, ~ticks.columns.duplicated()].copy()
                ticks["avg_bid"] = ticks.loc[:, ["Bid_mean", "Bid_median", "Bid_min",
                                                 "Bid_quantile(0.25)", "Bid_quantile(0.75)"]].mean(axis=1)
                ticks["avg_ask"] = ticks.loc[:, ["Ask_mean", "Ask_median", "Ask_max",
                                                 "Ask_quantile(0.25)", "Ask_quantile(0.75)"]].mean(axis=1)

                df = rates.merge(ticks, how="inner", left_index=True, right_index=True)

                return df

            else:
                return rates

        except ImportError as e:
            print(e, ": Not working as the MT5 Platform is not running. It must be open to import this data.")

    def import_data(self, tickers: List[str]):
        """
        Method to import data from MetaTrader 5 using their Python API.
        :param tickers: list of symbols/tickers to import
        :return: dictionary containing OHLCV and Bid/Ask data
        """
        # Connect to MetaTrader5
        mt5.initialize()
        # Set the desired account details
        try:
            login = mt5.login(login=self.account_number,
                              password=self.account_password,
                              server=self.server)
        except ConnectionError as e:
            print(f"{e}")
            print("Invalid account information! Please ensure you enter"
                  "the correct login details.")
            print("It is possible your account no longer exists if you"
                  "are using a demo account.")

        dfs = [self.retrieve_data(ticker) for ticker in tickers]
        # counts = Counter(np.concatenate([df.index.values for df in dfs]).ravel())
        # dates = pd.to_datetime([value for value, count in counts.items() if count == len(tickers)])
        # dfs = [df[df.index.floor("5min").isin(dates)] for df in dfs]
        for key, df in zip(tickers, dfs):
            self.data_frames[key] = df

        mt5.shutdown()

        return self.data_frames
