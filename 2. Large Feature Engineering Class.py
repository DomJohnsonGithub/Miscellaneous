import pandas as pd
import polars as pl
import numpy as np
import talib as ta
from time import time
from scipy.signal.windows import hann
from scipy.signal import welch
from scipy.spatial.distance import (jensenshannon, euclidean, cityblock,
                                    canberra, chebyshev, correlation, cosine)
from joblib import Parallel, delayed
from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc
from arch.unitroot.unitroot import VarianceRatio
import multiprocessing as mp
from nolds import lyap_r, corr_dim
from sklearn.cluster import KMeans
from scipy.stats import entropy, chi2_contingency
from typing import List


def timeit(method):
    """
    A decorator that times the execution of a method in a class.
    """

    def timed(*args, **kw):
        start_time = time()
        result = method(*args, **kw)
        end_time = time()
        print(f"Execution time of {method.__name__}: {end_time - start_time} seconds")
        return result

    return timed


class FeatureEngineering:
    def __init__(self, data: pd.DataFrame):
        self.df = data
        self.original_columns = self.df.columns.tolist()
        self.returns_continuous_columns = list
        self.technical_columns = list
        self.volatility_continuous_columns = list
        self.candlestick_column = str
        self.volume_continuous_columns = list
        self.volume_discrete_columns = list
        self.cyclical_columns = list
        self.statistical_discrete_columns = list
        self.statistical_continuous_columns = list
        self.other_continuous_columns = list
        self.other_discrete_columns = list
        self.lags = list
        self.windows = [3, 12, 72, 288]

    def target_variable(self, n_bars):
        self.df["target"] = np.where(self.df.Close.shift(-n_bars) > self.df.Close, 1, -1)
        self.df = self.df.iloc[:-n_bars, :]

    @timeit
    def cyclical_discrete(self):

        minute, hour = self.df.index.minute, self.df.index.hour
        sin_min = [np.sin(2 * np.pi * minute / (max(minute) / i)) for i in range(1, 13)]
        cos_min = [np.cos(2 * np.pi * minute / (max(minute) / i)) for i in range(1, 13)]
        sin_hr = [np.sin(2 * np.pi * hour / (max(hour) / i)) for i in range(1, 13)]
        cos_hr = [np.cos(2 * np.pi * hour / (max(hour) / i)) for i in range(1, 13)]
        sin_cos = np.concatenate([sin_min, cos_min, sin_hr, cos_hr])

        cols = [f"sin_min_{i}" for i in range(1, 13)] + [f"cos_min_{i}" for i in range(1, 13)] \
               + [f"sin_hr_{i}" for i in range(1, 13)] + [f"cos_hr_{i}" for i in range(1, 13)]

        self.cyclical_columns = cols
        self.df = pd.concat([self.df, pd.DataFrame(np.array(sin_cos).T, index=self.df.index, columns=cols)],
                            ignore_index=False, axis=1, join="outer")
        self.df = self.df.drop(columns=["sin_min_10", "sin_min_12", "sin_min_8", "sin_min_6",
                                        "sin_min_10", "sin_min_12", "sin_min_9", "sin_min_6",
                                        "sin_hr_12", "cos_hr_12"])

    @timeit
    def returns_continuous(self):
        cols_to_drop = self.df.columns.tolist()
        data = pl.DataFrame(self.df)
        data = data.with_column(pl.Series(name="datetime", values=self.df.index))
        lags = [1, 2, 3, 6, 12, 24, 72, 288]
        self.lags = lags

        # Returns
        data = data.with_column(pl.col("Close").pct_change().alias("returns"))

        # Lagged Returns & Momentum Returns
        for i in lags:
            data = data.with_column(pl.col("returns").shift(i).alias(f"lag_returns_{i}"))
            for j in lags:
                if i > j:
                    data = data.with_columns(
                        (pl.col(f"lag_returns_{i}") - pl.col(f"lag_returns_{j}")).alias(f"momentum_{i}_{j}"))

        for i in lags:
            data = data.drop(columns=[f"lag_returns_{i}"])

        # PSAR
        def psar() -> pl.Expr:
            return ta.SAR(data["High"], data["Low"], acceleration=0.02, maximum=0.2) / data["Close"] - 1

        data = data.select([pl.all(), pl.Series(psar()).alias("PSAR")])

        data = data.to_pandas().set_index("datetime")
        self.returns_continuous_columns = [
            [item for item in data.columns.tolist() if item not in cols_to_drop]]
        self.df = data

    @timeit
    def volatility_continuous(self):
        data = pl.DataFrame(self.df)  # Polars
        cols_to_drop = data.columns
        data = data.with_column(pl.Series(name="datetime", values=self.df.index))
        windows = self.windows

        # Polar Helper Functions
        def atr(timeperiod: int) -> pl.Expr:
            return ta.ATR(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)

        # Standard Deviation, Mean Absolute Deviation and ATR Combined
        for i, j in enumerate(windows):
            data = data.select([pl.all(), pl.Series(atr(j)).alias(f"ATR_{j}")])
            data = data.with_columns(data["returns"].rolling_apply(lambda x: pd.Series(x).std(), j).alias(f"STD_{j}"))
            data = data.with_column(
                data["returns"].rolling_apply(lambda x: self.mean_absolute_deviation(x), j).alias(f"MAD_{j}"))
            data = data.with_columns(
                pl.Series([np.nan] + list(np.diff((data[f"STD_{j}"] + data[f"MAD_{j}"] + data[f"ATR_{j}"]) / 3))).alias(
                    f"volatility_{j}"))
            data = data.drop(columns=[f"STD_{j}", f"MAD_{j}", f"ATR_{j}"])

        self.df = data.to_pandas().set_index("datetime")
        self.volatility_continuous_columns = [item for item in self.df.columns.tolist() if
                                              item not in cols_to_drop]

    @timeit
    def technical_analysis_continuous(self):
        data = pl.DataFrame(self.df)  # Polars
        cols_to_drop = data.columns
        data = data.with_column(pl.Series(name="datetime", values=self.df.index))
        windows = self.windows

        # Polar Helper Functions
        def adx(timeperiod: int) -> pl.Expr:
            return ta.ADX(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)

        def aroon(timeperiod: int) -> pl.Expr:
            return ta.AROONOSC(data["High"], data["Low"], timeperiod=timeperiod)

        def cci(timeperiod: int) -> pl.Expr:
            return ta.CCI(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)

        def mfi(timeperiod: int) -> pl.Expr:
            return ta.MFI(data["High"], data["Low"], data["Close"], data["Volume"], timeperiod=timeperiod)

        def rsi(timeperiod: int) -> pl.Expr:
            return ta.RSI(data["Close"], timeperiod=timeperiod)

        def willr(timeperiod: int) -> pl.Expr:
            return ta.WILLR(data["High"], data["Low"], data["Close"], timeperiod=timeperiod)

        def macd(slowperiod: int, fastperiod: int) -> pl.Expr:
            return ta.MACD(data["Close"], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=9)[0]

        def stoch(fastk_period: int, slowk_period: int) -> pl.Expr:
            return ta.STOCH(data["High"], data["Low"], data["Close"], fastk_period=fastk_period,
                            slowk_period=slowk_period, slowk_matype=0, slowd_period=3, slowd_matype=0)[0]

        def ultosc(timeperiod: int) -> pl.Expr:
            return ta.ULTOSC(data["High"], data["Low"], data["Close"],
                             timeperiod1=timeperiod, timeperiod2=timeperiod * 2, timeperiod3=timeperiod * 3)

        def bop() -> pl.Expr:
            return ta.BOP(data["Open"], data["High"], data["Low"], data["Close"])

        def adosc(slowperiod: int, fastperiod: int) -> pl.Expr:
            return ta.ADOSC(data["High"], data["Low"], data["Close"], data["Volume"], fastperiod=fastperiod,
                            slowperiod=slowperiod)

        def rsi_obv(timeperiod: int) -> pl.Expr:
            return ta.RSI(ta.OBV(data["Close"], data["Volume"]), timeperiod=timeperiod)

        # ----- MOMENTUM & VOLUME ----- #
        # ADX, AROONOSC, CCI, MFI, WILLR, MACD, ULTOSC, BOP, RSI, STOCH, ADOSC & RSI_OBV
        for i in windows:
            data = data.select(
                [pl.all(),
                 pl.Series(adx(i)).alias(f"ADX_{i}"),
                 pl.Series(aroon(i)).alias(f"AROON_{i}"),
                 pl.Series(cci(i)).alias(f"CCI_{i}"),
                 pl.Series(mfi(i)).alias(f"MFI_{i}"),
                 pl.Series(willr(i)).alias(f"WILLR_{i}"),
                 pl.Series(rsi(i)).alias(f"RSI_{i}"),
                 pl.Series(rsi_obv(i)).alias(f"RSI_OBV_{i}"),
                 pl.Series(ultosc(i)).alias(f"ULTOSC_{i}_{i * 2}_{i * 3}")])
            for j in windows:
                if i > j:
                    data = data.select([pl.all(), pl.Series(macd(i, j)).alias(f"MACD_{i}_{j}"),
                                        pl.Series(adosc(i, j)).alias(f"ADOSC_{i}_{j}"),
                                        pl.Series(stoch(i, j)).alias(f"STOCH_{i}_{j}")])

        data = data.rename({"MACD_12_3": "MACD_12"})
        data = data.rename({"ADOSC_12_3": "ADOSC_12"})
        data = data.rename({"STOCH_12_3": "STOCH_12"})
        data = data.select([pl.all(), pl.Series(bop()).alias("BOP"),
                            pl.Series((data["MACD_72_12"] + data["MACD_72_3"]) / 2).alias("MACD_72"),
                            pl.Series((data["MACD_288_72"] + data["MACD_288_12"] + data["MACD_288_3"]) / 3).alias(
                                "MACD_288"),
                            pl.Series((data["ADOSC_72_12"] + data["ADOSC_72_3"]) / 2).alias("ADOSC_72"),
                            pl.Series((data["ADOSC_288_72"] + data["ADOSC_288_12"] + data["ADOSC_288_3"]) / 3).alias(
                                "ADOSC_288"),
                            pl.Series((data["STOCH_72_12"] + data["STOCH_72_3"]) / 2).alias("STOCH_72"),
                            pl.Series((data["STOCH_288_72"] + data["STOCH_288_12"] + data["STOCH_288_3"]) / 3).alias(
                                "STOCH_288")])

        data = data.select([pl.all(),
                            pl.Series((data["WILLR_3"] + data["CCI_3"] + data["MFI_3"] + data["ULTOSC_3_6_9"] + data[
                                "AROON_3"] + data["RSI_OBV_3"] + data["RSI_3"]) / 7).alias("TA_3"),
                            pl.Series((data["WILLR_12"] + data["CCI_12"] + data["MFI_12"] + data["ULTOSC_12_24_36"] +
                                       data["MACD_12"] + data["AROON_12"] + data["RSI_OBV_12"] + data["RSI_12"] + data[
                                           "STOCH_12"]) / 9).alias("TA_12"),
                            pl.Series((data["WILLR_72"] + data["CCI_72"] + data["MFI_72"] + data["ULTOSC_72_144_216"] +
                                       data["MACD_72"] + data["AROON_72"] + data["RSI_OBV_72"] + data["RSI_72"] + data[
                                           "STOCH_72"]) / 9).alias("TA_72"),
                            pl.Series((data["WILLR_288"] + data["CCI_288"] + data["MFI_288"] + data[
                                "ULTOSC_288_576_864"] + data["MACD_288"] + data["AROON_288"] + data[
                                           "RSI_OBV_288"] + data["RSI_288"] + data["STOCH_288"]) / 9).alias("TA_288")])

        for i in windows:
            data = data.drop(
                [f"WILLR_{i}", f"CCI_{i}", f"MFI_{i}", f"RSI_OBV_{i}",
                 f"AROON_{i}", f"RSI_{i}", f"ULTOSC_{i}_{i * 2}_{i * 3}"])
            for j in windows:
                if i > j and i != 12:
                    data = data.drop([f"MACD_{i}_{j}", f"ADOSC_{i}_{j}", f"STOCH_{i}_{j}"])
        data = data.drop(["MACD_12", "MACD_72", "MACD_288", "STOCH_12", "STOCH_72", "STOCH_288"])

        self.df = data.to_pandas().set_index("datetime")
        self.technical_columns = [item for item in self.df.columns.tolist() if item not in cols_to_drop]

    @timeit
    def volume_features_continuous(self):
        data = pl.DataFrame(self.df)  # Polars
        cols_to_drop = data.columns
        data = data.with_column(pl.Series(name="datetime", values=self.df.index))

        # Volume: Percentage change and lags, Z-Scores, Amihud Illiquidity
        data = data.with_column(pl.col("Volume").pct_change().alias(f"volume_pct_change"))
        for i in self.lags:
            data = data.with_column(pl.col("volume_pct_change").shift(i).alias(f"volume_pct_change_t-{i}"))

        for i in self.windows:
            data = data.with_column(
                ((pl.col("Volume") - pl.col("Volume").rolling_mean(i)) / pl.col("Volume").rolling_std(i)).alias(
                    f"volume_z_scores_{i}"))

        data = data.with_column(pl.Series(np.abs(data["returns"]) / data["Volume"]).alias("amihud_illiquidity"))
        for i in self.windows:
            data = data.with_column(
                pl.Series(np.abs(data["returns"].rolling_mean(i)) / data["Volume"].rolling_mean(i)).alias(
                    f"amihud_illiquidity_{i}"))

        self.df = data.to_pandas().set_index("datetime")
        self.volume_continuous_columns = [item for item in self.df.columns.tolist() if item not in cols_to_drop]

    @timeit
    def statistical_continuous(self):
        df, data = self.df, pl.DataFrame(self.df)  # Pandas, Polars
        data = data.with_column(pl.Series(name="datetime", values=df.index))
        windows = self.windows
        cols_to_drop = df.columns.tolist()

        # Statistics: Max, Min, Skewness, Kurtosis:
        rets_range = np.array([df.returns.rolling(i).max() + df.returns.rolling(i).min() for i in windows]).T
        rets_skew = np.array([df.returns.rolling(i).skew().diff() for i in windows[1:]]).T
        rets_kurt = np.array([df.returns.rolling(i).kurt().diff() for i in windows[1:]]).T

        for i, j in enumerate(windows):
            data = data.with_column(pl.Series(rets_range[:, i]).alias(f"max_min_{j}"))

        for i, j in enumerate(windows[1:]):
            data = data.with_column(pl.Series(rets_skew[:, i]).alias(f"skew_{j}"))

        for i, j in enumerate(windows[1:]):
            data = data.with_column(pl.Series(rets_kurt[:, i]).alias(f"kurt_{j}"))

        # Statistics: Spectral Entropy
        fs = 1 / (5 * 60)  # sampling frequency (in Hz)
        spectral_entropies = np.array([df["returns"][1:].rolling(i).apply(
            lambda x: self.spectral_entropy(x, fs, hann(j)), raw=True).to_list() for i, j in
                                       zip(windows, windows)]).T

        for i, j in enumerate(windows):
            data = data.with_column(
                pl.Series(self.add_nan_row_top(spectral_entropies)[:, i]).alias(f"spectral_entropy_{j}"))
        data = data.drop(columns=["spectral_entropy_3"])

        # Statistics: Rolling High Max - Rolling Low Min, Log and Diff
        for i in windows:
            rolling_low_min = data["Low"].rolling_min(i)
            rolling_high_max = data["High"].rolling_max(i)
            diff_0 = np.log(rolling_high_max - rolling_low_min)
            data = data.with_column(
                pl.Series(diff_0).alias(f"log_max_min_{i}"))

        for i in windows:
            rolling_low_min = data["Low"].rolling_min(i)
            rolling_high_max = data["High"].rolling_max(i)
            diff_1 = np.array([np.nan] + list(np.log1p(np.diff(rolling_high_max - rolling_low_min))))
            data = data.with_column(pl.Series(diff_1).alias(f"diff_max_min_{i}"))

        # Statistics: JS Divergence, Mutual Information
        window = 288
        rets = df.returns[1:].values
        jsd_values = [jensenshannon(
            self.normalized_non_zero_probs(rets[i - window:i][:-1], bins=2, discretize=True, n_vals=2),
            self.normalized_non_zero_probs(rets[i - window:i][1:], bins=2, discretize=True, n_vals=2)) ** 2
                      for i in range(window, len(rets))]
        mi_values = [self.calc_MI(rets[i - window:i][:-1], rets[i - window:i][1:], bins=2)
                     for i in range(window, len(rets))]
        jsd_values = pd.Series((np.array([np.nan] * (window + 1) + jsd_values)) + 1)
        mi_values = pd.Series((1 + (np.array([np.nan] * (window + 1) + mi_values))))
        data = data.with_column(pl.Series([np.nan] + list(np.diff(jsd_values) * 100)).alias(f"js_div"))
        data = data.with_column(pl.Series([np.nan] + list(np.diff(mi_values))).alias(f"mutual_info"))

        # Distances
        window = 288
        distances = np.array(
            [self.calculate_distance(df["Close"][i - window:i].values) for i in range(window, len(df))])
        nan_array = np.full((window, 6), np.nan)
        distances = np.concatenate((nan_array, distances))
        for i in range(np.shape(distances)[1]):
            data = data.with_column(pl.Series(distances[:, i]).alias(f"distances_{i}").pct_change())

        data = data.with_column(pl.Series(
            (data["distances_0"] + data["distances_1"] + data["distances_2"] + data["distances_3"]) / 4).alias(
            "distances"))
        data = data.drop(columns=["distances_0", "distances_1", "distances_2", "distances_3", "distances_4"])

        self.df = data.to_pandas().set_index("datetime")
        self.statistical_continuous_columns = [item for item in self.df.columns.tolist() if
                                               item not in cols_to_drop]

    @timeit
    def other_continuous(self):
        df, data = self.df, pl.DataFrame(self.df)  # Pandas, Polars
        data = data.with_column(pl.Series(name="datetime", values=df.index))
        windows = self.windows
        cols_to_drop = df.columns.tolist()

        # Ranges
        data = data.with_column(pl.Series(data["Open"] - data["Close"]).alias("Open_Close"))
        data = data.with_column(pl.Series(data["Low"] - data["High"]).alias("Low_High"))
        data = data.with_column(
            pl.Series((data["High"] + data["Low"]) / 2 - (data["Open"] + data["Close"]) / 2).alias("HL_avg-OC_avg"))

        # Smart Money Index
        range = data["High"] - data["Low"]
        buy_pressure = (data["Close"] - data["Low"]) / range
        sell_pressure = (data["High"] - data["Close"]) / range
        for i in windows:
            data = data.with_column(
                pl.Series(buy_pressure.rolling_mean(i) - sell_pressure.rolling_mean(i)).alias(f"SMI_{i}"))

        # Pivot Points
        for i in windows:
            rolling_high, rolling_low = data["High"].rolling_max(i), data["Low"].rolling_min(i)
            rolling_close = data["Close"].rolling_apply(lambda x: x[-1], i)
            pivots = (rolling_high + rolling_low + rolling_close) / 3
            support_1, support_2 = (2 * pivots) - rolling_high, pivots - (rolling_high - rolling_low)
            resistance_1, resistance_2 = (2 * pivots) - rolling_low, pivots + (rolling_high - rolling_low)
            combined_1 = pivots - (support_1 + support_2) / 2
            combined_2 = (resistance_2 + resistance_1) / 2 - pivots
            data = data.with_column(
                pl.Series([np.nan] + list(np.diff((combined_1 + combined_2) / 2))).alias(f"pivot_points_comb_{i}"))

        # Lyapunov Exponent
        data = data.with_column(
            pl.Series(self.tau_rolling_parallelised(df.Close.values, period=288, lag=20, min_tstep=75)).alias(
                f"lyapunov_exp"))

        # Fractal Dimension
        data = data.with_column(
            pl.Series(self.correlation_dimension_parallel(df=df["Close"].values, window=288, emb_dim=3)).alias(
                "fractal_dimension"))

        # Hurst Exponent
        periods = [100, 200, 300]
        hurst_exps = np.column_stack([[np.nan] * i + self.compute_hurst(data=df.Close, window=i) for i in periods])
        for i in np.arange(len(periods)):
            data = data.with_column(pl.Series(hurst_exps[:, i]).alias(f"hurst_exp_{i}"))

        # Price and RSI Divergence
        for i in windows:
            close_max, close_min = data["Close"].rolling_max(i), data["Close"].rolling_min(i)
            rsi = ta.RSI(data["Close"], timeperiod=14)
            rsi_max, rsi_min = rsi.rolling_max(i), rsi.rolling_min(i)
            data = data.with_column(
                pl.Series(([np.nan] * (i + 1)) + list(
                    np.diff(np.log(((close_max[i:] - close_min[i:]) / (rsi_max[i:] - rsi_min[i:])) + 1)))).alias(
                    f"close_rsi_divergence_{i}"))
            for j in self.lags:
                data = data.with_column(data[f"close_rsi_divergence_{i}"].shift(j).alias(f"close_rsi_div_lag_{i}_{j}"))

        self.df = data.to_pandas().set_index("datetime")
        self.other_continuous_columns = [item for item in self.df.columns.tolist() if item not in cols_to_drop]

    @timeit
    def candlesticks_discrete(self):
        data = pl.DataFrame(self.df)
        data = data.with_column(pl.Series(name="datetime", values=self.df.index))

        o, h, l, c = data["Open"], data["High"], data["Low"], data["Close"]

        candle_patterns = [
            ta.CDL2CROWS(o, h, l, c), ta.CDL3BLACKCROWS(o, h, l, c), ta.CDL3INSIDE(o, h, l, c),
            ta.CDL3LINESTRIKE(o, h, l, c), ta.CDL3OUTSIDE(o, h, l, c), ta.CDL3STARSINSOUTH(o, h, l, c),
            ta.CDL3WHITESOLDIERS(o, h, l, c), ta.CDLABANDONEDBABY(o, h, l, c), ta.CDLADVANCEBLOCK(o, h, l, c),
            ta.CDLBELTHOLD(o, h, l, c), ta.CDLBREAKAWAY(o, h, l, c), ta.CDLCLOSINGMARUBOZU(o, h, l, c),
            ta.CDLCONCEALBABYSWALL(o, h, l, c), ta.CDLCOUNTERATTACK(o, h, l, c), ta.CDLDARKCLOUDCOVER(o, h, l, c),
            ta.CDLDOJI(o, h, l, c), ta.CDLDOJISTAR(o, h, l, c), ta.CDLDRAGONFLYDOJI(o, h, l, c),
            ta.CDLENGULFING(o, h, l, c), ta.CDLEVENINGDOJISTAR(o, h, l, c), ta.CDLEVENINGSTAR(o, h, l, c),
            ta.CDLGAPSIDESIDEWHITE(o, h, l, c), ta.CDLGRAVESTONEDOJI(o, h, l, c), ta.CDLHAMMER(o, h, l, c),
            ta.CDLHANGINGMAN(o, h, l, c), ta.CDLHARAMI(o, h, l, c), ta.CDLHARAMICROSS(o, h, l, c),
            ta.CDLHIGHWAVE(o, h, l, c), ta.CDLHIKKAKE(o, h, l, c), ta.CDLHIKKAKEMOD(o, h, l, c),
            ta.CDLHOMINGPIGEON(o, h, l, c), ta.CDLIDENTICAL3CROWS(o, h, l, c), ta.CDLINNECK(o, h, l, c),
            ta.CDLINVERTEDHAMMER(o, h, l, c), ta.CDLKICKING(o, h, l, c), ta.CDLKICKINGBYLENGTH(o, h, l, c),
            ta.CDLLADDERBOTTOM(o, h, l, c), ta.CDLLONGLEGGEDDOJI(o, h, l, c), ta.CDLLONGLINE(o, h, l, c),
            ta.CDLMARUBOZU(o, h, l, c), ta.CDLMATCHINGLOW(o, h, l, c), ta.CDLMATHOLD(o, h, l, c),
            ta.CDLMORNINGDOJISTAR(o, h, l, c), ta.CDLMORNINGSTAR(o, h, l, c), ta.CDLONNECK(o, h, l, c),
            ta.CDLPIERCING(o, h, l, c), ta.CDLRICKSHAWMAN(o, h, l, c), ta.CDLRISEFALL3METHODS(o, h, l, c),
            ta.CDLSEPARATINGLINES(o, h, l, c), ta.CDLSHOOTINGSTAR(o, h, l, c), ta.CDLSHORTLINE(o, h, l, c),
            ta.CDLSPINNINGTOP(o, h, l, c), ta.CDLSTALLEDPATTERN(o, h, l, c), ta.CDLSTICKSANDWICH(o, h, l, c),
            ta.CDLTAKURI(o, h, l, c), ta.CDLTASUKIGAP(o, h, l, c), ta.CDLTHRUSTING(o, h, l, c),
            ta.CDLTRISTAR(o, h, l, c, ), ta.CDLUNIQUE3RIVER(o, h, l, c), ta.CDLUPSIDEGAP2CROWS(o, h, l, c),
            ta.CDLXSIDEGAP3METHODS(o, h, l, c)]

        data = data.with_column(pl.Series(name="candlesticks", values=np.sign(np.sum(candle_patterns, axis=0))))
        for i in self.lags:
            data = data.with_column(data["candlesticks"].shift(i).alias(f"candlesticks_lag_{i}"))

        self.candlestick_column = ["candlesticks"] + [f"candlesticks_lag_{i}" for i in self.lags]
        self.df = data.to_pandas().set_index("datetime")

    @timeit
    def volume_features_discrete(self):
        df, data = self.df, pl.DataFrame(self.df)  # Pandas, Polars
        cols_to_drop = data.columns
        data = data.with_column(pl.Series(name="datetime", values=df.index))

        # Volume: Above MA, Binning, Sign, Plus-Minus
        data = data.with_column(pl.col("volume_pct_change").sign().alias("volume_sign"))

        for i in self.windows:
            data = data.with_column(
                pl.Series(np.where(data["Volume"] >= data[f"Volume"].rolling_mean(i), 1, 0)).alias(
                    f"volume_above_ma_{i}"))
            data = data.with_column(
                pl.col("volume_sign").rolling_apply(lambda x: x.sum(), i).alias(f"volume_plus_minus_{i}"))
            for j in self.lags:
                data = data.with_column(data[f"volume_above_ma_{i}"].shift(j).alias(f"volume_above_ma_{i}_{j}"))
                data = data.with_column(data[f"volume_plus_minus_{i}"].shift(j).alias(f"volume_plus_minus_{i}_{j}"))

        bins = np.array(
            [df.Volume.rolling(i).apply(lambda x: self.binning(x, n_bins=5), raw=True) for i in self.windows]).T
        for i, j in enumerate(self.windows):
            data = data.with_column(pl.Series(bins[:, i]).alias(f"volume_binning_{j}"))
            for k in self.lags:
                data = data.with_column(data[f"volume_binning_{j}"].shift(k).alias(f"volume_binning_{j}_{k}"))

        self.df = data.to_pandas().set_index("datetime")
        self.volume_discrete_columns = [item for item in self.df.columns.tolist() if item not in cols_to_drop]

    @timeit
    def statistical_discrete(self):
        df, data = self.df, pl.DataFrame(self.df)  # Pandas, Polars
        data = data.with_column(pl.Series(name="datetime", values=df.index))
        windows = self.windows
        cols_to_drop = df.columns.tolist()

        choices = [0, 1, 2, 3, 4, 5, 6, 7]
        conditions = [(data["returns"].to_numpy() < -0.003),
                      ((-0.003 <= data["returns"].to_numpy()) & (data["returns"].to_numpy() < -0.002)),
                      ((-0.002 <= data["returns"].to_numpy()) & (data["returns"].to_numpy() < -0.001)),
                      ((-0.001 <= data["returns"].to_numpy()) & (data["returns"].to_numpy() < 0.000)),
                      ((0.000 <= data["returns"].to_numpy()) & (data["returns"].to_numpy() < 0.001)),
                      ((0.001 <= data["returns"].to_numpy()) & (data["returns"].to_numpy() < 0.002)),
                      ((0.002 <= data["returns"].to_numpy()) & (data["returns"].to_numpy() < 0.003)),
                      (data["returns"].to_numpy() >= 0.003)]

        data = data.with_column(pl.Series(np.select(conditions, choices, default=np.nan)).alias(f"returns_threshold"))

        # Plus-Minus
        data = data.with_column(pl.col("returns").sign().alias("returns_sign"))
        for i in windows:
            data = data.with_column(
                pl.col("returns_sign").rolling_apply(lambda x: x.sum(), i).alias(f"returns_plus_minus_{i}"))
            for j in self.lags:
                data = data.with_column(data[f"returns_plus_minus_{i}"].shift(j).alias(f"returns_plus_minus_{i}_{j}"))

        # Binning
        bins = self.add_nan_row_top(
            np.array([df.returns[1:].rolling(i).apply(lambda x: self.binning(x), raw=True) for i in windows]).T)
        for i, j in enumerate(windows):
            data = data.with_column(pl.Series(bins[:, i]).alias(f"returns_binning_{j}"))
            for k in self.lags:
                data = data.with_column(data[f"returns_binning_{j}"].shift(k).alias(f"returns_binning_{j}_{k}"))

        # Statistics: Shannon Entropy
        def apply_rolling_entropy(window: int, discretize: bool, n_vals: int) -> pl.Expr:
            return FeatureEngineering.rolling_entropy(data["returns"][1:], window, discretize=discretize, n_vals=n_vals)

        for i, j in enumerate(windows):
            data = data.select([pl.all(), pl.Series(apply_rolling_entropy(j, discretize=True, n_vals=2)).alias(
                f"shannon_entropy_{j}")])
            for k in self.lags:
                data = data.with_column(data[f"shannon_entropy_{j}"].shift(j).alias(f"shannon_entropy_{j}_{k}"))

        self.df = data.to_pandas().set_index("datetime")
        self.statistical_discrete_columns = [item for item in self.df.columns.tolist() if item not in cols_to_drop]

    @timeit
    def others_discrete(self):
        df, data = self.df, pl.DataFrame(self.df)  # Pandas, Polars
        data = data.with_column(pl.Series(name="datetime", values=df.index))
        windows = self.windows
        cols_to_drop = df.columns.tolist()

        # Helper Functions
        def up_bb(timeperiod: int, deviation: int) -> pl.Expr:
            return ta.BBANDS(data["Close"], timeperiod=timeperiod, nbdevup=deviation, nbdevdn=deviation, matype=0)[0]

        def lw_bb(timeperiod: int, deviation: int) -> pl.Expr:
            return ta.BBANDS(data["Close"], timeperiod=timeperiod, nbdevup=deviation, nbdevdn=deviation, matype=0)[2]

        # Bollinger Bands - Price Inside or Outside
        choices = [0, 1, 0]
        for i in windows:
            for j in [1, 2]:
                data = data.select([pl.all(), pl.Series(up_bb(i, j)).alias(f"UP_BB_{i}_{j}")])
                data = data.select([pl.all(), pl.Series(lw_bb(i, j)).alias(f"LW_BB_{i}_{j}")])

                conditions = [data["Close"].to_numpy() - data[f"UP_BB_{i}_{j}"].to_numpy() > 0,
                              np.logical_and(data[f"LW_BB_{i}_{j}"].to_numpy() <= data["Close"].to_numpy(),
                                             data["Close"].to_numpy() <= data[f"UP_BB_{i}_{j}"].to_numpy()),
                              data["Close"].to_numpy() - data[f"LW_BB_{i}_{j}"].to_numpy() < 0]
                data = data.drop([f"UP_BB_{i}_{j}", f"LW_BB_{i}_{j}"])
                if i == 3 and j == 2:
                    pass
                else:
                    data = data.with_column(
                        pl.Series(np.select(conditions, choices, default=np.nan)).alias(f"outside_BB_{i}_{j}"))
                    for m in self.lags:
                        data = data.with_column(data[f"outside_BB_{i}_{j}"].shift(m).alias(f"outside_BB_{i}_{j}_{m}"))
                        for k in windows:
                            data = data.with_column(
                                pl.col(f"outside_BB_{i}_{j}").rolling_sum(k).alias(f"outside_BB_{i}_{j}_sum_{k}"))
                            data = data.with_column(
                                data[f"outside_BB_{i}_{j}_sum_{k}"].shift(m).alias(f"outside_BB_{i}_{j}_sum_{m}"))

        # Binary Volatile Candles
        range = data["High"] - data["Low"]
        stds = [1, 2, 3]
        for i in stds:
            for j in windows:
                up_bb = range.rolling_mean(j) + i * range.rolling_std(j)
                if i == 1:
                    data = data.with_column(pl.Series(j * [np.nan] + list(np.where(range[j:] > up_bb[j:], 1, 0))).alias(
                        f"volatile_candles_{i}_{j}"))
                    for k in self.lags:
                        data = data.with_column(
                            data[f"volatile_candles_{i}_{j}"].shift(k).alias(f"volatile_candles_{i}_{j}_{k}"))
                if i == 2 and j > 3:
                    data = data.with_column(pl.Series(j * [np.nan] + list(np.where(range[j:] > up_bb[j:], 1, 0))).alias(
                        f"volatile_candles_{i}_{j}"))
                    for k in self.lags:
                        data = data.with_column(
                            data[f"volatile_candles_{i}_{j}"].shift(k).alias(f"volatile_candles_{i}_{j}_{k}"))
                if i == 3 and j > 12:
                    data = data.with_column(pl.Series(j * [np.nan] + list(np.where(range[j:] > up_bb[j:], 1, 0))).alias(
                        f"volatile_candles_{i}_{j}"))
                    for k in self.lags:
                        data = data.with_column(
                            data[f"volatile_candles_{i}_{j}"].shift(k).alias(f"volatile_candles_{i}_{j}_{k}"))

        # Bank trading round numbers
        middle_series = df.Close
        for i in [0.001, 0.005]:
            upper_round = np.ceil(df.Close / i) * i
            lower_round = np.floor(df.Close / i) * i
            upper_middle = upper_round - middle_series
            middle_lower = middle_series - lower_round
            conditions = [upper_middle - middle_lower < 0, upper_middle - middle_lower == 0,
                          upper_middle - middle_lower > 0]
            choices = [0, 1, 2]
            data = data.with_column(pl.Series(np.select(conditions, choices)).alias(f"Bank_round_num_{i * 1000}"))
        for i in [0.001, 0.005]:
            for j in self.lags:
                data = data.with_column(
                    pl.col(f"Bank_round_num_{i * 1000}").shift(j).alias(f"Bank_round_num_{i * 1000}_{j}"))

        # Is price near recent max high, or recent min low
        for i in windows:
            rolling_max = data["High"].rolling_max(i)
            rolling_min = data["Low"].rolling_min(i)
            rolling_close = data["Close"].rolling_apply(lambda x: x[-1], i)
            upper_close = (rolling_max - rolling_close).to_numpy()
            close_lower = (rolling_close - rolling_min).to_numpy()
            conditions = [upper_close - close_lower < 0, upper_close - close_lower == 0,
                          upper_close - close_lower > 0]
            choices = [0, 1, 2]
            data = data.with_column(
                pl.Series(np.select(conditions, choices, default=np.nan)).alias(f"closer_min_max_{i}"))
            for j in self.lags:
                data = data.with_column(pl.col(f"closer_min_max_{i}").shift(j).alias(f"closer_min_max_{i}_{j}"))

        # Fibonacci Price Placement
        for i in windows:
            data = data.with_column(
                pl.Series(self.price_fibo_placement(df.High, df.Low, df.Close, period=i)).alias(f"fibo_{i}"))
            for j in self.lags:
                data = data.with_column(pl.col(f"fibo_{i}").shift(j).alias(f"fibo_{i}_{j}"))

        # Hurst Exponent Binary
        for i in [0, 1, 2]:
            data = data.with_column(
                pl.Series(np.where(data[f"hurst_exp_{i}"].to_numpy() >= 0.5, 1, 0)).alias(f"hurst_exp_{i}_binary"))

        # Lyapunov Exponent Binary
        data = data.with_column(
            pl.Series(np.where(data[f"lyapunov_exp"].to_numpy() > 0, 1, 0)).alias(f"lyapunov_exp_binary"))

        # K-Means Clustering
        data = data.with_column(
            pl.Series(self.parallel_compute_kmeans(df, window=288, n_clusters=3)).alias("kmeans_cluster"))

        #  Statistics: ADF Test, Variance Ratio Test - Mean-Reversion (Stationary vs Non-Stationary), Ergodicity/Stationarity
        for i in windows[1:]:
            data = data.with_column(
                pl.Series(self.adfuller_rolling_parallelised(data["Close"], i)).alias(f"adf_{i}"))
            data = data.with_column(pl.Series(self.variance_ratio_test_parallel(df["returns"][1:].values, i)).alias(
                f"variance_ratio_{i}"))

        self.df = data.to_pandas().set_index("datetime")
        self.other_discrete_columns = [item for item in self.df.columns.tolist() if item not in cols_to_drop]

    @staticmethod
    def variance_ratio_test(ts, window):
        return VarianceRatio(ts, window).vr

    @staticmethod
    def variance_ratio_test_parallel(ts, window):
        results = Parallel(n_jobs=-1)(
            delayed(FeatureEngineering.variance_ratio_test)(ts[i - window:i], window / 5) for i in
            range(window, len(ts)))
        return np.where(np.array([np.nan] * (window + 1) + results) >= 1, 1, 0)

    @staticmethod
    def kmeans_clustering(data, n_clusters):
        k_means = KMeans(n_clusters=n_clusters, max_iter=300)
        k_means.fit(data)
        return k_means.labels_[-1]

    @staticmethod
    def parallel_compute_kmeans(df, window, n_clusters):
        results = Parallel(n_jobs=mp.cpu_count())(delayed(FeatureEngineering.kmeans_clustering)(
            np.array([df["High"][i - window:i].values / df["Open"][i - window:i].values,
                      df["Low"][i - window:i].values / df["Open"][i - window:i].values,
                      df["Close"][i - window:i].values / df["Open"][i - window:i].values]).T, n_clusters) for i in
                                                  range(window, len(df)))
        return [np.nan] * window + results

    @staticmethod
    def hurst(data: pd.Series) -> float:
        return compute_Hc(data, kind="price", simplified=True)[0]

    @staticmethod
    def compute_hurst(data: pd.Series, window: int):
        return Parallel(n_jobs=-1, backend="loky")(
            delayed(FeatureEngineering.hurst)(data[i:i + window]) for i in range(len(data) - window))

    @staticmethod
    def correlation_dimension_parallel(df, window, emb_dim):
        results = Parallel(n_jobs=12)(
            delayed(corr_dim)(df[i - window:i], emb_dim) for i in range(window, len(df)))
        return [np.nan] * window + results

    @staticmethod
    def largest_lyapunov_exponent(x, lag, min_tstep):
        return lyap_r(x, emb_dim=3, lag=lag, min_tsep=min_tstep, tau=1)

    @staticmethod
    def tau_rolling_parallelised(ts, period, lag, min_tstep):
        results = Parallel(n_jobs=12)(
            delayed(FeatureEngineering.largest_lyapunov_exponent)(ts[i - period:i], lag, min_tstep) for i in
            range(period, len(ts)))
        return period * [np.nan] + results

    @staticmethod
    def fibonacci_retracement(high, low, period):
        # Calculate the range of prices over the specified period
        high_range = high.rolling(window=period).max()
        low_range = low.rolling(window=period).min()
        range = high_range - low_range
        # Calculate the Fibonacci levels
        levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        return [np.array(high_range - range * level) for level in levels]

    @staticmethod
    def price_fibo_placement(high, low, close, period):
        retracements = FeatureEngineering.fibonacci_retracement(high, low, period)
        high, middle_high, middle = retracements[0], retracements[1], retracements[2]
        middle_low, low, close = retracements[3], retracements[4], close.values
        choicelist = [0, 1, 2, 3, 4, 5, np.nan]
        condlist = [close > high,
                    np.logical_and(middle_high < close, close <= high),
                    np.logical_and(middle < close, close <= middle_high),
                    np.logical_and(middle_low < close, close <= middle),
                    np.logical_and(low < close, close <= middle_low),
                    close <= low, np.arange(len(close)) <= period - 1]
        return np.select(condlist, choicelist)

    @staticmethod
    def calculate_adf(data):
        return 0 if adfuller(data, regression="c")[1] <= 0.05 else 1

    @staticmethod
    def adfuller_rolling_parallelised(data, period):
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(FeatureEngineering.calculate_adf)(data[i:i + period]) for i in range(len(data) - period + 1))
        return np.array((period - 1) * [np.nan] + results)

    @staticmethod
    def mean_absolute_deviation(data):
        mean = data.mean()
        absolute_deviation = np.abs(data - mean)
        return absolute_deviation.mean()

    @staticmethod
    def jensen_shannon_divergence(p, q):
        m = 0.5 * (p + q)
        kl_p_m = entropy(p, m)
        kl_q_m = entropy(q, m)
        return 0.5 * kl_p_m + 0.5 * kl_q_m

    @staticmethod
    def calc_MI(x, y, bins):
        c_xy = np.histogram2d(x, y, bins)[0]
        g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
        return 0.5 * g / c_xy.sum()

    @staticmethod
    def normalized_non_zero_probs(observations, bins=10, discretize=True, n_vals=3):
        observations = np.copy(observations)
        if discretize and n_vals == 3:
            observations = FeatureEngineering.to_0_1_minus_1(observations, three_bins=True)
        elif discretize and n_vals == 2:
            observations = FeatureEngineering.to_0_1_minus_1(observations, three_bins=False)

        # Bin the observations into `bins` bins
        hist, bin_edges = np.histogram(observations, bins=bins)
        # Calculate the width of the bins
        bin_widths = np.diff(bin_edges)
        # Divide the histogram values by the total number of observations
        relative_frequencies = hist / len(observations)
        # Divide the relative frequencies by the bin widths to get the PMF
        pmf = relative_frequencies / bin_widths
        # Keep only non-zero probabilities
        non_zero = pmf != 0
        pmf = pmf[non_zero]
        # Normalize the PMF so the probabilities sum up to 1
        pmf /= pmf.sum()
        return pmf

    @staticmethod
    def rolling_entropy(observations, window_size=10, discretize=True, n_vals=3):
        # Get the normalized non-zero probabilities for each window of size `window_size`
        rolling_entropies = np.zeros(len(observations) - window_size + 1)
        for i in range(len(rolling_entropies)):
            window = observations[i:i + window_size]
            pmf = FeatureEngineering.normalized_non_zero_probs(window, discretize=discretize, n_vals=n_vals)
            rolling_entropies[i] = -np.sum(pmf * np.log2(pmf))  # entropy using normalized PMF
        return window_size * [np.nan] + list(rolling_entropies)

    @staticmethod
    def calculate_distance(x):
        manhattan_dist = cityblock(x[:-1], x[1:])
        euclidean_dist = euclidean(x[:-1], x[1:])
        canberra_dist = canberra(x[:-1], x[1:])
        cosine_dist = cosine(x[:-1], x[1:])
        chebyshev_dist = chebyshev(x[:-1], x[1:])
        correlation_dist = correlation(x[:-1], x[1:])
        return (manhattan_dist, euclidean_dist, canberra_dist,
                cosine_dist, chebyshev_dist, correlation_dist)

    @staticmethod
    def to_0_1_minus_1(arr, three_bins=True):
        if three_bins:
            arr = np.array(arr)
            arr[arr > 0] = 1
            arr[arr < 0] = -1
            arr[arr == 0] = 0
        else:
            arr[arr >= 0] = 1
            arr[arr < 0] = -1
        return arr

    @staticmethod
    def to_0_1(arr):
        arr[arr >= 0] = 1
        arr[arr < 0] = 0
        return arr

    @staticmethod
    def add_nan_row_top(arr):
        return np.vstack([np.full(arr.shape[1], np.nan), arr])

    @staticmethod
    def binning(data, n_bins=10):
        bucket = np.linspace(data.min(), data.max(), num=n_bins)
        dig = np.digitize(data, bins=bucket)
        return dig[-1]

    @staticmethod
    def spectral_entropy(x, fs, window):
        # Estimate the power spectral density using Welch's method
        freqs, psd = welch(x, fs=fs, window=window, nperseg=len(x))
        # Normalize the power spectral density
        psd /= np.sum(psd)
        return -np.sum(psd * np.log2(psd))  # spectral entropy

    def remove_nan_values(self):
        self.df = self.df.dropna()

    def drop_columns(self, columns: List):
        self.df = self.df.drop(columns=columns)

    def get_dataframe(self) -> pd.DataFrame:
        return self.df