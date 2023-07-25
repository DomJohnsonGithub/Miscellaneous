import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import talib as ta
from timeseriescv.cross_validation import CombPurgedKFoldCV, PurgedWalkForwardCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from skopt.searchcv import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from xgboost import XGBClassifier
from sklearn.metrics import (balanced_accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score)
import pickle
import sklearn.metrics as skm
import multiprocessing
import MetaTrader5 as mt5
from itertools import combinations

sns.set_style("darkgrid")


def split_train_test(data, size=0.8):
    return data.iloc[:int(len(data) * size), :], data.iloc[int(len(data) * size):, :]


def feature_engineering(train, test):
    dataframe = [train, test]

    for data in dataframe:
        data["rsi"] = ta.RSI(data.Close, timeperiod=14)
        data["returns"] = data.Close.pct_change()

        lags = np.arange(1, 6)
        for i in lags:
            data[f"Close_{i}d"] = data.Close.shift(i)

        data["target"] = data["Close"].shift(-1)

        data.dropna(inplace=True)

    return dataframe[0], dataframe[1]


class BlockingTimeSeriesPurgedSplit:
    def __init__(self, n_splits, purge_gap, train_percentage):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.train_pct = train_percentage

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

    def get_purge_gap(self, X, y, groups=None):
        return self.purge_gap

    def get_train_percentage(self, X, y, groups=None):
        return self.train_pct

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(self.train_pct * (stop - start)) + start

            yield indices[start: mid], indices[mid + self.purge_gap: stop]


class ReduceVIF(BaseEstimator, TransformerMixin):

    def __init__(self, thresh=5.0, columns=None):
        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh
        self.columns = columns

        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, because they are all set together
        if hasattr(self, 'dropped_cols_'):
            del self.dropped_cols_
            del self.max_vifs_

    def fit(self, X, y=None):
        self._reset()
        print('ReduceVIF fit')
        columns = self.columns if self.columns is not None else X.columns.tolist()
        X_i = pd.DataFrame(X, columns=columns)
        self.dropped_cols_, self.max_vifs_ = ReduceVIF.calculate_vif(X_i, self.thresh)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        X_df = pd.DataFrame(X, columns=self.columns) if self.columns is not None else X
        return X_df.drop(self.dropped_cols_, axis=1)

    @staticmethod
    def calculate_vif(X_o, thresh=5.0):
        X = X_o.copy()
        max_vifs = []
        dropped_cols = []

        dropped = True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]

            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)

                drop_col = X.columns.tolist()[maxloc]
                print(f'Dropping {drop_col} with vif={max_vif}')
                X = X.drop([drop_col], axis=1)

                max_vifs.append(max_vif)
                dropped_cols.append(drop_col)
                dropped = True

        return dropped_cols, max_vifs


class AnotherStandardScaler(StandardScaler):
    def fit(self, X, y=None, **kwargs):
        self.feature_names_ = X.columns
        return super().fit(X, y, **kwargs)

    def transform(self, X, **kwargs):
        return pd.DataFrame(data=super().transform(X, **kwargs),
                            columns=self.feature_names_)


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
        This method will shutdown the connection
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
        :param timeframe: input a MT5 TimeFrame, ranging from 1 minute to 1 month.
        :return: rates dataframe.
        """
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
        try:
            ticks = mt5.copy_ticks_from(self.symbol, self.first_datetime_index, 20_000_000, mt5.COPY_TICKS_ALL)
            ticks = pd.DataFrame(ticks)
            ticks.index = pd.to_datetime(ticks["time"], unit="s") - timedelta(hours=2)
            ticks = ticks.groupby(ticks.index).mean()[
                    self.first_datetime_index:]  # average tick data with same datetime
            ticks = ticks.resample("T").mean()
            ticks = ticks.loc[:self.end_datetime_index, :]
            ticks.columns = ["Timestamp", "Bid", "Ask", "Last", "Volume", "Time_msc", "Flags", "Volume_Real"]
            ticks.drop(columns=["Volume", "Time_msc", "Flags", "Volume_Real", "Last", "Timestamp"], inplace=True)
            ticks["Bid-Ask"] = ticks.Ask - ticks.Bid
        except ImportError as e:
            print(e, ": Not working as the MT5 Platform is not running. It must be open to import this data.")

        return ticks

    @staticmethod
    def _get_symbol_data_from_MT5():
        """
        This method calls upon the get rates and ticks methods to
        merge the data into a singular dataframe.
        :return: tick and rates dataframe.
        """
        rates = MT5_Trading_Platform._get_symbol_data()
        ticks = MT5_Trading_Platform._get_symbol_ticks()
        df = rates.merge(ticks, how="inner", left_index=True, right_index=True)

        return df

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None

    # account_number = 6675778386  <- not mine
    # account_password = "dffdfhhh" <- not mine
    # server = "MetaQuotes-Demo"
    #
    # MT5 = MT5_Trading_Platform(account_number, account_password, server)
    # MT5._establish_connection()
    # MT5._login()
    # MT5._authorizing_account()
    # MT5._terminal_information()
    # MT5._number_symbols()
    # MT5._get_tradable_symbols()
    #
    # symbol = "GBPCHF"
    # MT5._get_specific_symbol_info(symbol)
    #
    # timeframe = mt5.TIMEFRAME_M1  # Minutely data
    # MT5._set_timeframe(timeframe=timeframe)
    # df = MT5._get_symbol_data_from_MT5()
    # # ---------------------------------
    # df["Returns"] = df.Close.pct_change()
    # for i in np.arange(2, 11):
    #     df[f"rets{i}"] = df.Close.pct_change(i)
    #
    # for i in np.arange(1, 11):
    #     df[f"lag_rets{i}"] = df.Returns.shift(i)
    #
    # for i in np.arange(2, 11):
    #     df[f"momentum1_{i}"] = df[f"rets{i}"] - df.Returns
    #
    # for i in [10, 9, 8, 7, 6, 5, 4, 3, 2]:
    #     for j in np.arange(2, 11):
    #         if i > j:
    #             df[f"momentum2_{i}-{j}"] = df[f"rets{i}"] - df[f"rets{j}"]
    #
    # df[f"Kyle_{chr(955)}"] = np.abs(df.Returns)/df.Volume
    # df["Target"] = df.Returns.shift(-1)
    # df["Forward_Returns"] = df.Target
    #
    #
    # df.drop(columns=["Open", "High", "Low", "Close", "Volume", "Bid", "Ask"], inplace=True)
    # df.dropna(inplace=True)
    #
    # df["Target"] = np.where(df.Target >= 0, 1, -1)
    #
    # df.to_pickle("my_data_fx.pkl")

    df = pd.read_pickle("my_data_fx.pkl")

    # print(df.Target.value_counts())
    # print(df.Returns.describe())

    # initial_capital = 10_000
    # signal = np.array([1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1])
    # rets = np.array([0.00016, -0.00025, -0.00017, 0.0003, 0.00012, 0.00014, -0.00002, 0.00032,
    #                  -0.0002, 0.00051, 0.00013, -0.00022, -0.00013, 0.0001, 0.0003])
    #
    #
    # print(np.cumprod(1 + (signal * rets)))
    # plt.plot(np.cumprod(1 + (signal * rets)))
    # plt.show()

    # print(sfsg)

    rets = df.Returns[int(0.8 * len(df)):]
    fwd_rets = df.Forward_Returns[int(0.8 * len(df)):]
    X = df.drop(columns=["Target", "Forward_Returns"])
    y = df.Target

    X_train, X_test = X.iloc[:int(0.8 * len(X)), :], X.iloc[int(0.8 * len(X)):, :]
    y_train, y_test = y.iloc[:int(0.8 * len(X))], y.iloc[int(0.8 * len(X)):]

    print(skm.get_scorer_names())

    sc = StandardScaler()
    cv = BlockingTimeSeriesPurgedSplit(n_splits=5, purge_gap=2, train_percentage=0.8)

    # -- Naive Bayes -- #  1
    # nb_model = GaussianNB()
    # pipe_nb = Pipeline([
    #     ("sc", sc),
    #     ("nb", nb_model)
    # ])
    # pipe_nb.fit(X_train, y_train)

    # filename1 = 'nb_model.sav'
    # pickle.dump(pipe_nb, open(filename1, 'wb'))
    # pipe_nb = pickle.load(open(filename1, 'rb'))
    #
    # yhat_nb = pipe_nb.predict(X_test)
    # yhat_nb = pd.Series(yhat_nb, index=X_test.index)
    #
    # yprob_nb_0 = pipe_nb.predict_proba(X_test)[:, 0]
    # yprob_nb_1 = pipe_nb.predict_proba(X_test)[:, 1]
    #
    # with open('nb_probs.npy', 'wb') as f:
    #     np.save(f, yprob_nb_0)
    #     np.save(f, yprob_nb_1)

    with open('nb_probs.npy', 'rb') as f:
        yprob_nb_0 = np.load(f)
        yprob_nb_1 = np.load(f)
    #
    # yhat_nb.to_pickle("yhat_nb.pkl")
    yhat_nb = pd.read_pickle("yhat_nb.pkl")

    print("Naive Bayes:")
    print(accuracy_score(y_test, yhat_nb))
    print(balanced_accuracy_score(y_test, yhat_nb))
    print(roc_auc_score(y_test, yhat_nb, average="macro", labels=[-1, 1]))
    print(confusion_matrix(y_test, yhat_nb, labels=[-1, 1]))
    print(classification_report(y_test, yhat_nb, labels=[-1, 1]))

    # plot_confusion_matrix(pipe_nb, X_test, y_test, labels=[-1, 1])
    # plt.show()

    # -- Logistic Regression -- #  2
    # lr = LogisticRegression(penalty="elasticnet", fit_intercept=True, class_weight="balanced",
    #                         max_iter=1000, n_jobs=-1, solver="saga")
    #
    # pipe_lr = Pipeline([
    #     ("sc", sc),
    #     ("lr", lr)
    # ])
    #
    # param_grid_lr = {
    #     "lr__C": np.logspace(-5, 5, 10),
    #     "lr__l1_ratio": np.linspace(0, 1, 11)
    # }
    #
    # lr_model = GridSearchCV(estimator=pipe_lr, param_grid=param_grid_lr, scoring="balanced_accuracy",
    #                           refit=True, cv=cv, return_train_score=True, verbose=2)
    # lr_model.fit(X_train, y_train)

    filename2 = 'lr_model.sav'
    # pickle.dump(lr_model, open(filename2, 'wb'))
    lr_model = pickle.load(open(filename2, 'rb'))
    #
    # yhat_lr = lr_model.predict(X_test)
    # yhat_lr = pd.Series(yhat_lr, index=X_test.index)
    #
    # yprob_lr_0 = lr_model.predict_proba(X_test)[:, 0]
    # yprob_lr_1 = lr_model.predict_proba(X_test)[:, 1]
    #
    # with open('lr_probs.npy', 'wb') as f:
    #     np.save(f, yprob_lr_0)
    #     np.save(f, yprob_lr_1)

    with open('lr_probs.npy', 'rb') as f:
        yprob_lr_0 = np.load(f)
        yprob_lr_1 = np.load(f)

    # yhat_lr.to_pickle("yhat_lr.pkl")
    yhat_lr = pd.read_pickle("yhat_lr.pkl")

    print("Logistic Regression:")
    print(lr_model.best_score_)
    print(lr_model.best_params_)
    print(accuracy_score(y_test, yhat_lr))
    print(balanced_accuracy_score(y_test, yhat_lr))
    print(roc_auc_score(y_test, yhat_lr, average="macro", labels=[-1, 1]))
    print(confusion_matrix(y_test, yhat_lr, labels=[-1, 1]))
    print(classification_report(y_test, yhat_lr, labels=[-1, 1]))

    # plot_confusion_matrix(lr_model.best_estimator_, X_test, y_test, labels=[-1, 1])
    # plt.show()

    # -- Support Vector Machine -- #  3
    # svm = SVC(probability=True, class_weight="balanced", max_iter=-1, shrinking=True, gamma="auto",
    #           kernel="rbf")
    #
    # pipe_svm = Pipeline([
    #     ("sc", sc),
    #     ("svm", svm)
    # ])
    #
    # param_grid_svm = {
    #     "svm__C": [0.001, 0.01, 0.1, 1, 10, 100]
    # }
    #
    # svm_model = GridSearchCV(estimator=pipe_svm, param_grid=param_grid_svm, scoring="balanced_accuracy",
    #                           refit=True, cv=cv, return_train_score=True, verbose=2)
    # svm_model.fit(X_train, y_train)

    filename3 = 'svm_model.sav'
    # pickle.dump(svm_model, open(filename3, 'wb'))
    svm_model = pickle.load(open(filename3, 'rb'))

    # yhat_svm = svm_model.predict(X_test)
    # yhat_svm = pd.Series(yhat_svm, index=X_test.index)
    #
    # yprob_svm_0 = svm_model.predict_proba(X_test)[:, 0]
    # yprob_svm_1 = svm_model.predict_proba(X_test)[:, 1]
    #
    # with open('svm_probs.npy', 'wb') as f:
    #     np.save(f, yprob_svm_0)
    #     np.save(f, yprob_svm_1)

    with open('svm_probs.npy', 'rb') as f:
        yprob_svm_0 = np.load(f)
        yprob_svm_1 = np.load(f)

    # yhat_svm.to_pickle("yhat_svm.pkl")
    yhat_svm = pd.read_pickle("yhat_svm.pkl")

    print("Support Vector Machine:")
    print(svm_model.best_score_)
    print(svm_model.best_params_)
    print(accuracy_score(y_test, yhat_svm))
    print(balanced_accuracy_score(y_test, yhat_svm))
    print(roc_auc_score(y_test, yhat_svm, average="macro", labels=[-1, 1]))
    print(confusion_matrix(y_test, yhat_svm, labels=[-1, 1]))
    print(classification_report(y_test, yhat_svm, labels=[-1, 1]))

    # plot_confusion_matrix(svm_model.best_estimator_, X_test, y_test, labels=[-1, 1])
    # plt.show()

    # -- Quadratic Discriminant Analysis -- #  4
    # qda = QuadraticDiscriminantAnalysis(store_covariance=False)
    #
    # pipe_qda = Pipeline([
    #     ("sc", sc),
    #     ("qda", qda)
    # ])
    #
    # param_grid_qda = {
    #     "qda__reg_param": [0.00001, 0.0001, 0.001, 0.01, 0.1],
    #     "qda__tol": [0.0001, 0.001, 0.01, 0.1]
    # }
    #
    # qda_model = GridSearchCV(estimator=pipe_qda, param_grid=param_grid_qda, scoring="balanced_accuracy",
    #                          refit=True, cv=cv, return_train_score=True, verbose=2, error_score="raise")
    #
    # qda_model.fit(X_train, y_train)

    filename4 = 'qda_model.sav'
    # pickle.dump(qda_model, open(filename4, 'wb'))
    qda_model = pickle.load(open(filename4, 'rb'))
    # qda_model = pickle.load(open(filename4, 'rb'))
    #
    # yhat_qda = qda_model.predict(X_test)
    # yhat_qda = pd.Series(yhat_qda, index=X_test.index)
    #
    # yprob_qda_0 = qda_model.predict_proba(X_test)[:, 0]
    # yprob_qda_1 = qda_model.predict_proba(X_test)[:, 1]
    #
    # with open('qda_probs.npy', 'wb') as f:
    #     np.save(f, yprob_qda_0)
    #     np.save(f, yprob_qda_1)

    with open('qda_probs.npy', 'rb') as f:
        yprob_qda_0 = np.load(f)
        yprob_qda_1 = np.load(f)

    # yhat_qda.to_pickle("yhat_qda.pkl")
    yhat_qda = pd.read_pickle("yhat_qda.pkl")

    print("Quadratic Component Analysis:")
    print(qda_model.best_score_)
    print(qda_model.best_params_)
    print(accuracy_score(y_test, yhat_qda))
    print(balanced_accuracy_score(y_test, yhat_qda))
    print(roc_auc_score(y_test, yhat_qda, average="macro", labels=[-1, 1]))
    print(confusion_matrix(y_test, yhat_qda, labels=[-1, 1]))
    print(classification_report(y_test, yhat_qda, labels=[-1, 1]))

    # plot_confusion_matrix(qda_model.best_estimator_, X_test, y_test, labels=[-1, 1])
    # plt.show()

    # -- Random Forest -- #  5
    n_jobs = multiprocessing.cpu_count() - 1

    # rf = RandomForestClassifier(criterion="gini", bootstrap=True, n_jobs=n_jobs, class_weight="balanced_subsample",
    #                             max_features="sqrt")
    #
    # pipe_rf = Pipeline([
    #     ("rf", rf)
    # ])
    #
    # param_grid_rf = {
    #     "rf__n_estimators": Integer(100, 2000, "log-uniform"),
    #     "rf__max_depth": Integer(2, 20, "uniform"),
    #     "rf__max_samples": Real(0.1, 1.0, "uniform"),
    #     "rf__min_samples_split": Integer(2, 11, "uniform"),
    #     "rf__min_samples_leaf": Integer(1, 11, "uniform")
    # }
    #
    # rf_model = BayesSearchCV(estimator=pipe_rf, search_spaces=param_grid_rf, n_iter=100, n_points=10, n_jobs=-1, cv=cv,
    #                          verbose=2, return_train_score=True, refit=True, scoring="balanced_accuracy",
    #                          optimizer_kwargs={"base_estimator": "GP", "n_initial_points": 10,
    #                                            "acq_func": "gp_hedge","n_jobs": 11, "acq_optimizer": "auto",
    #                                             "initial_point_generator": "hammersly", "random_state": None})
    # rf_model.fit(X_train, y_train)

    filename5 = 'rf_model.sav'
    # pickle.dump(rf_model, open(filename5, 'wb'))
    rf_model = pickle.load(open(filename5, 'rb'))
    #
    # yhat_rf = rf_model.predict(X_test)
    # yhat_rf = pd.Series(yhat_rf, index=X_test.index)
    #
    # yprob_rf_0 = rf_model.predict_proba(X_test)[:, 0]
    # yprob_rf_1 = rf_model.predict_proba(X_test)[:, 1]
    #
    # with open('rf_probs.npy', 'wb') as f:
    #     np.save(f, yprob_rf_0)
    #     np.save(f, yprob_rf_1)

    with open('rf_probs.npy', 'rb') as f:
        yprob_rf_0 = np.load(f)
        yprob_rf_1 = np.load(f)

    # yhat_rf.to_pickle("yhat_rf.pkl")
    yhat_rf = pd.read_pickle("yhat_rf.pkl")

    print("Random Forest:")
    print(rf_model.best_score_)
    print(rf_model.best_params_)
    print(accuracy_score(y_test, yhat_rf))
    print(balanced_accuracy_score(y_test, yhat_rf))
    print(roc_auc_score(y_test, yhat_rf, average="macro", labels=[-1, 1]))
    print(confusion_matrix(y_test, yhat_rf, labels=[-1, 1]))
    print(classification_report(y_test, yhat_rf, labels=[-1, 1]))

    # plot_confusion_matrix(rf_model.best_estimator_, X_test, y_test, labels=[-1, 1])
    # plt.show()

    # -- K-Nearest Neighbours -- #  6
    # knn = KNeighborsClassifier(algorithm="auto", n_jobs=-1)
    #
    # pipe_knn = Pipeline([
    #     ("sc", sc),
    #     ("knn", knn)
    # ])
    #
    # param_grid_knn = {
    #     "knn__n_neighbors": np.arange(5, 31, 1),
    #     "knn__weights": ["uniform", "distance"],
    #     "knn__metric": ["manhattan", "euclidean"]
    # }
    #
    # knn_model = GridSearchCV(estimator=pipe_knn, param_grid=param_grid_knn, scoring="balanced_accuracy",
    #                           refit=True, cv=cv, return_train_score=True, verbose=2)
    # knn_model.fit(X_train, y_train)

    filename6 = 'knn_model.sav'
    # pickle.dump(knn_model, open(filename6, 'wb'))
    knn_model = pickle.load(open(filename6, 'rb'))

    # yhat_knn = knn_model.predict(X_test)
    # yhat_knn = pd.Series(yhat_knn, index=X_test.index)
    #
    # yprob_knn_0 = knn_model.predict_proba(X_test)[:, 0]
    # yprob_knn_1 = knn_model.predict_proba(X_test)[:, 1]

    # with open('knn_probs.npy', 'wb') as f:
    #     np.save(f, yprob_knn_0)
    #     np.save(f, yprob_knn_1)

    with open('knn_probs.npy', 'rb') as f:
        yprob_knn_0 = np.load(f)
        yprob_knn_1 = np.load(f)

    # yhat_knn.to_pickle("yhat_knn.pkl")
    yhat_knn = pd.read_pickle("yhat_knn.pkl")

    print("K-Nearest Neighbours:")
    print(knn_model.best_score_)
    print(knn_model.best_params_)
    print(accuracy_score(y_test, yhat_knn))
    print(balanced_accuracy_score(y_test, yhat_knn))
    print(roc_auc_score(y_test, yhat_knn, average="macro", labels=[-1, 1]))
    print(confusion_matrix(y_test, yhat_knn, labels=[-1, 1]))
    print(classification_report(y_test, yhat_knn, labels=[-1, 1]))

    # plot_confusion_matrix(knn_model.best_estimator_, X_test, y_test, labels=[-1, 1])
    # plt.show()

    # -- XGBoost -- #  7
    # xgboost = XGBClassifier(booster="gbtree", tree_method="gpu_hist", learning_rate=0.1,
    #                         objective="binary:logistic", predictor="gpu_predictor",
    #                         gpu_id=-1, verbosity=2, single_precision_histogram=True, random_state=None)
    #
    # xgboost_pipe = Pipeline([
    #     ("xgb", xgboost)])
    #
    # search_space = {
    #     "xgb__n_estimators": Integer(10, 1000, prior="uniform"),
    #     "xgb__reg_alpha": Real(0, 10, prior="uniform"),
    #     "xgb__reg_lambda": Real(0, 20, prior="uniform"),
    #     "xgb__max_depth": Integer(2, 8),
    #     "xgb__gamma": Real(0, 0.5, prior="uniform"),
    #     "xgb__min_child_weight": Integer(1, 20),
    #     "xgb__max_delta_step": Integer(0, 10),
    #     "xgb__subsample": Real(0.3, 0.9, prior="uniform"),
    #     "xgb__colsample_bytree": Real(0.5, 0.9, prior="uniform"),
    #     "xgb__colsample_bylevel": Real(0.5, 0.9, prior="uniform"),
    #     "xgb__colsample_bynode": Real(0.5, 0.9, prior="uniform"),
    #     "xgb__scale_pos_weight": Real(0.9, 1.1, prior="uniform")
    # }
    #
    # bayes_xgboost = BayesSearchCV(estimator=xgboost_pipe, search_spaces=search_space, return_train_score=True, refit=True,
    #                               n_jobs=n_jobs, cv=cv, verbose=2, scoring="roc_auc", n_iter=200, n_points=20,
    #                               optimizer_kwargs={"base_estimator": "GP", "n_initial_points": 10,
    #                                                 "acq_func": "gp_hedge",
    #                                                 "n_jobs": 11, "acq_optimizer": "auto",
    #                                                 "initial_point_generator": "hammersly",
    #                                                 "random_state": None},
    #                               random_state=None)

    y_train_xgb = np.where(y_train == -1, 0, 1)
    y_test_xgb = np.where(y_test == -1, 0, 1)

    # bayes_xgboost.fit(X_train, y_train_xgb)

    filename7 = 'xgboost_model.sav'
    # pickle.dump(bayes_xgboost, open(filename7, 'wb'))
    bayes_xgboost = pickle.load(open(filename7, 'rb'))
    #
    # yhat_xgboost = bayes_xgboost.predict(X_test)
    # yhat_xgboost = pd.Series(yhat_xgboost, index=X_test.index)
    #
    # yprob_xgboost_0 = bayes_xgboost.predict_proba(X_test)[:, 0]
    # yprob_xgboost_1 = bayes_xgboost.predict_proba(X_test)[:, 1]
    #
    # with open('xgboost_probs.npy', 'wb') as f:
    #     np.save(f, yprob_xgboost_0)
    #     np.save(f, yprob_xgboost_1)

    with open('xgboost_probs.npy', 'rb') as f:
        yprob_xgboost_0 = np.load(f)
        yprob_xgboost_1 = np.load(f)

    # yhat_xgboost.to_pickle("yhat_xgboost.pkl")
    yhat_xgboost = pd.read_pickle("yhat_xgboost.pkl")

    print("XGBoost: ")
    print(bayes_xgboost.best_score_)
    print(bayes_xgboost.best_params_)
    print(accuracy_score(y_test_xgb, yhat_xgboost))
    print(balanced_accuracy_score(y_test_xgb, yhat_xgboost))
    print(roc_auc_score(y_test_xgb, yhat_xgboost, average="macro", labels=[0, 1]))
    print(confusion_matrix(y_test_xgb, yhat_xgboost, labels=[0, 1]))
    print(classification_report(y_test_xgb, yhat_xgboost, labels=[0, 1]))

    # plot_confusion_matrix(bayes_xgboost.best_estimator_, X_test, y_test_xgb, labels=[0, 1])
    # plt.show()

    # -- Hard Vote Ensemble -- #
    yhat_all = np.vstack([yhat_nb.values, yhat_lr.values, yhat_svm.values, yhat_qda.values,
                          yhat_rf.values, yhat_knn.values, np.where(yhat_xgboost == 0, -1, 1)]).T
    yhat_hard_vote = pd.Series([np.squeeze(mode(i)[0]) for i in yhat_all], index=X_test.index, dtype=np.int32)

    print("Hard Vote (all): ")
    print(accuracy_score(y_test, yhat_hard_vote))
    print(balanced_accuracy_score(y_test, yhat_hard_vote))
    print(roc_auc_score(y_test, yhat_hard_vote, average="macro", labels=[-1, 1]))
    print(confusion_matrix(y_test, yhat_hard_vote, labels=[-1, 1]))
    print(classification_report(y_test, yhat_hard_vote, labels=[-1, 1]))

    model_names = ["NB", "LR", "SVM", "QDA", "RF", "KNN", "XGB"]
    model_predictions = pd.DataFrame(yhat_all, columns=model_names, index=X_test.index, dtype=np.int32)
    combo_three_names = [i for i in combinations(model_names, r=3)]
    combo_five_names = [i for i in combinations(model_names, r=5)]

    three_model_preds = []
    for i in range(len(combo_three_names)):
        three_model_preds.append(np.where(model_predictions[np.squeeze(combo_three_names[i])].sum(axis=1) >= 0, 1, -1))
    three_model_preds = np.array(three_model_preds).T
    three_model_preds = pd.DataFrame(three_model_preds, index=X_test.index,
                                     columns=["%s+%s+%s" % (i[0], i[1], i[2]) for i in combo_three_names])

    five_model_preds = []
    for i in range(len(combo_five_names)):
        five_model_preds.append(np.where(model_predictions[np.squeeze(combo_five_names[i])].sum(axis=1) >= 0, 1, -1))
    five_model_preds = np.array(five_model_preds).T
    five_model_preds = pd.DataFrame(five_model_preds, index=X_test.index,
                                    columns=["%s+%s+%s+%s+%s" % (i[0], i[1], i[2], i[3], i[4]) for i in
                                             combo_five_names])

    model_preds = pd.concat([three_model_preds, five_model_preds], axis=1, ignore_index=False)

    # -- Strategy Returns -- #
    buy_hold_returns = pd.DataFrame(rets[1:].values, index=rets[1:].index, columns=["Buy_Hold"])
    strategy_returns_nb = pd.DataFrame(yhat_nb.shift(1)[1:].values * rets[1:].values, index=rets[1:].index,
                                       columns=["NB"])
    strategy_returns_lr = pd.DataFrame(yhat_lr.shift(1)[1:].values * rets[1:].values, index=rets[1:].index,
                                       columns=["LR"])
    strategy_returns_svm = pd.DataFrame(yhat_svm.shift(1)[1:].values * rets[1:].values, index=rets[1:].index,
                                        columns=["SVM"])
    strategy_returns_qda = pd.DataFrame(yhat_qda.shift(1)[1:].values * rets[1:].values, index=rets[1:].index,
                                        columns=["QDA"])
    strategy_returns_rf = pd.DataFrame(yhat_rf.shift(1)[1:].values * rets[1:].values, index=rets[1:].index,
                                       columns=["RF"])
    strategy_returns_knn = pd.DataFrame(yhat_knn.shift(1)[1:].values * rets[1:].values, index=rets[1:].index,
                                        columns=["KNN"])
    strategy_returns_xgb = pd.DataFrame(yhat_xgboost.shift(1)[1:].values * rets[1:].values, index=rets[1:].index,
                                        columns=["XGB"])
    strategy_returns_hardv = pd.DataFrame(yhat_hard_vote.shift(1)[1:].values * rets[1:].values, index=rets[1:].index,
                                          columns=["Hard_Vote (all)"])
    ensemble_strategy_rets = (model_preds.shift(1)[1:].T * rets[1:]).T

    strategy_returns = pd.concat(
        [buy_hold_returns, strategy_returns_nb, strategy_returns_lr, strategy_returns_svm, strategy_returns_qda,
         strategy_returns_rf, strategy_returns_knn, strategy_returns_xgb, strategy_returns_hardv,
         ensemble_strategy_rets], axis=1, ignore_index=False)
    print(strategy_returns)
    cumulative_strategy_returns = np.cumprod(1 + strategy_returns)
    print(cumulative_strategy_returns)

    ax = cumulative_strategy_returns.plot()
    for line, name in zip(ax.lines, cumulative_strategy_returns.columns):
        y = line.get_ydata()[-1]
        ax.annotate(name, xy=(1, y), xytext=(6, 0), color=line.get_color(),
                    xycoords=ax.get_yaxis_transform(), textcoords="offset points",
                    size=5, va="center")
    ax.get_legend().remove()
    plt.subplots_adjust(left=0.03, bottom=0.082, right=0.905, top=0.985)
    plt.show()
