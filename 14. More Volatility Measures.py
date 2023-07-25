import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
import talib as ta
import pandas_datareader.data as pdr
import matplotlib.gridspec as gridspec
from datetime import datetime
import seaborn as sns
import yfinance as yf

sns.set_style("darkgrid")


def volatility(data, period):
    data["Volatility(sigma)"] = data.Close.rolling(window=period).std()
    return data


def annualized_volatility(data, period):
    data["Annualized_Vol"] = data.Close.rolling(window=period).std() * np.sqrt(252)
    return data


def atr(data, period):
    data["ATR"] = ta.ATR(data.High, data.Low, data.Close, timeperiod=period)
    return data


def exponential_atr(data, period):
    """Calculates the Exponential Average True Range.
    """
    # Calculate True Range
    data["TR"] = ta.TRANGE(data.High, data.Low, data.Close)
    # Calculate the smoothing factor
    a = 2 / (period + 1)
    # Calculate the Average True Range
    data["ATR"] = ta.ATR(data.High, data.Low, data.Close, timeperiod=period)
    # Calculate the exponential ATR
    for i in range(len(data)):
        eATR = a * data.iloc[i, 4] + (1 - a) * data.iloc[i - 1, 5]
    data["eATR"] = eATR
    # remove unnecessary columns
    data.drop(columns=["TR", "ATR"], inplace=True)

    return data


def pure_pupil_volatility_indicator(data, period, ma_period, multiplier, bands=False):
    """
    PPVI indicator has two lines. They are correlated but they do make the assumption
    that upward volatility is different from when the market goes down. They can be
    used for either risk management or part of a strategy. In risk management, it
    can be used to set stops and take profits. In the strategy arena, we can
    """
    data["H_SD"] = data.Close.rolling(window=period).std()
    data["L_SD"] = data.Close.rolling(window=period).std()
    data["PPVI_HSD"] = data.H_SD.rolling(window=period).max()
    data["PPVI_LSD"] = data.L_SD.rolling(window=period).max()

    data["MA"] = data.Close.rolling(window=ma_period).mean()
    data["PPVI_UP_BAND"] = data.MA + (data["PPVI_HSD"] * multiplier)
    data["PPVI_LW_BAND"] = data.MA - (data["PPVI_LSD"] * multiplier)
    data.drop(columns=["H_SD", "L_SD", "MA"], inplace=True)

    if bands == True:
        data.drop(columns=["PPVI_HSD", "PPVI_LSD"], inplace=True)
    else:
        data.drop(columns=["PPVI_UP_BAND", "PPVI_LW_BAND"], inplace=True)
    return data


def bollinger_band_spread(data, period, deviation, ma_type):
    up_bb, _, lw_bb = ta.BBANDS(data.Close, timeperiod=period,
                                nbdevup=deviation, nbdevdn=deviation,
                                matype=ma_type)
    data["BB_SPREAD"] = up_bb - lw_bb
    return data


def augmented_bollinger_band_spread(data, ma_period, deviation, bands=False):
    data["HIGH_MA"] = data.High.rolling(window=ma_period).mean()
    data["LOW_MA"] = data.Low.rolling(window=ma_period).mean()

    high_std = data.High.rolling(window=ma_period).std()
    low_std = data.Low.rolling(window=ma_period).std()

    data["AUG_UP_BB"] = data.HIGH_MA + (deviation * high_std)
    data["AUG_LW_BB"] = data.LOW_MA - (deviation * low_std)

    data["AUG_BB_SPREAD"] = data["AUG_UP_BB"] - data["AUG_LW_BB"]

    if bands == True:
        data.drop(columns=["HIGH_MA", "LOW_MA", "AUG_BB_SPREAD"], inplace=True)
    else:
        data.drop(columns=["HIGH_MA", "LOW_MA", "AUG_UP_BB",
                           "AUG_LW_BB"], inplace=True)

    return data


def normalized_bollinger_band(data, deviation, period, ma_type):
    up_bb, _, lw_bb = ta.BBANDS(data.Close, timeperiod=period, nbdevup=deviation,
                                nbdevdn=deviation, matype=ma_type)
    data["%B"] = (data.Close - lw_bb) / (up_bb - lw_bb)
    return data


# Future Realized Volatility Estimators

def historical_high_low_volatility_parkinson(data):
    """
    Uses the daily range and can then capture the intraday move.
    However, it cannot handle trends and jumps (only appropriate
    for measuring volatility of a GBM process) and it systematically
    underestimates the volatility.
    """
    parkinson_volatility = []
    for i in range(len(data)):
        parkinson_volatility.append(np.sqrt((1 / (4 * len(data) * np.log(2))) \
                                            * np.sum(np.log(data.iloc[i, 0] / data.iloc[i, 1]) ** 2)))
    data["Parkinson_Vol"] = parkinson_volatility

    return data


def garman_klass_volatility(data):
    """
    This is an extension of the Parkinson volatility as it takes
    into account of the open and close prices. Since markets are
    most active during the opening and closing of a trading session,
    this is an non-negligible shortcoming. It incorporates some
    intraday information stored at daily frequencies. Is 8x more
    efficient than Parkinson's. However it is more biased than his.
    """
    garman_klass = []
    for i in range(len(data)):
        garman_klass.append(np.sqrt((252/len(data))*np.sum(0.5*np.log(data.iloc[i, 0]/data.iloc[i, 1])**2 \
                                - (2*np.log(2)-1)*np.log(data.iloc[i, 3]/data.iloc[i, 2])**2)))
    data["Garman_Klass_Vol"] = garman_klass

    return data


def roger_satchel_volatility_estimator(data):
    """
    This allows for thr presence of trends (non-zero drift),
    but it does not account for jumps
    """
    rs = []
    for i in range(len(data)):
        rs.append(
            (1/len(data)) * \
            np.sum(
                (np.log(data.iloc[i, 0]/data.iloc[i, 3]) * \
                 np.log(data.iloc[i, 0]/data.iloc[i, 2])) + \
                (np.log(data.iloc[i, 1]/data.iloc[i, 3]) * \
                 np.log(data.iloc[i, 1]/data.iloc[i, 2]))
            )
        )
    data["Roger_Satchel_Vol"] = rs

    return data


def yang_zhang_volatility(data):
    """
    Yang and Zhang extension of the Garman and Klass historical
    volatility estimator. This modification allows the volatility
    estimator to account for the opening jumps, but as the original
    function, it assumes that the underlying follows a Brownian
    motion with zero drift (the historical mean return should be
    equal to zero).

    The estimator tend to overestimate the volatility when the
    drift is different from zero, however, for a zero drift motion,
    this estimator has an efficiency of eight times the classic
    close-to-close estimator (standard deviation).


    """
    yz = []
    for i in range(len(data)):
        yz.append(
            np.sqrt(
                (1/len(data))*np.sum(
                    np.log(data.iloc[i, 2]/data.iloc[i-1, 3])**2
                ) + \
                (1/len(data))*np.sum(0.5 * \
                                     (np.log(data.iloc[i, 0]/
                                             data.iloc[i, 1]))**2) - \
                (1/len(data))*np.sum((2*np.log(2)-1)*(np.log(data.iloc[i, 3]/
                                                             data.iloc[i, 2])**2))
            )
        )
    data["Yang_Zhang_Vol"] = yz

    return data

