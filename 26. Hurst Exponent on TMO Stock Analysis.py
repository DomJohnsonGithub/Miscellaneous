import pandas as pd
import pandas_datareader.data as pdr
from datetime import datetime
import matplotlib.pyplot as plt
from hurst import compute_Hc as hc
import numpy as np
import matplotlib.gridspec as gridspec
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import probplot, moment, norm
import seaborn as sns
import warnings
import yfinance as yf

warnings.filterwarnings("ignore")

tickers = ["TMO", "DHR"]
prices = yf.download(tickers, start=datetime(2010, 1, 1), end=datetime(2020, 1, 1))["Close"].dropna()

prices["HURST_EXPONENT"] = 0
prices.drop(columns=["DHR"], inplace=True)
print(prices)

H = hc(prices["TMO"])[0]
print("HURST_EXPONENT (2010-2020) for TMO:", H)

for i in range(101, len(prices)):
    prices.HURST_EXPONENT.iloc[i] = hc(prices.TMO.iloc[i - 101:i - 1], kind="price")[0]
print(prices)

print(prices["HURST_EXPONENT"][101:])

fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=100, sharex=True)
axes[0].plot(prices["TMO"][101:], c="red", lw=1.2, label="TMO Daily Close ($)")
axes[0].set_title("Thermo Fisher Scientific Inc. Daily Close Price")
axes[0].grid(True)
axes[0].legend(loc="upper left")
axes[1].plot(prices["HURST_EXPONENT"][101:], c="green", label="Rolling Hurst_Exponent")
axes[1].axhline(y=0.5, linestyle="--", c="purple", label="H_E: 0.5 = Random Walk")
axes[1].legend(loc="best")
axes[1].grid(True)
plt.show()


# --------------------------------------------------------------------------------------
def correlogram(x, lags, title):
    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(nrows=3, ncols=2, figure=fig)

    # Log Returns
    ax1 = fig.add_subplot(gs[0, :])
    x.plot(ax=ax1, grid=True, color="red")
    q_p = np.max(q_stat(acf(x, lags), len(x))[1])
    stats = f'ADF:{adfuller(x)[1]:.2f},\nQ-Stat:{q_p:.2f}'
    ax1.text(x=.03, y=.85, s=stats, transform=ax1.transAxes)

    # ACF
    ax2 = fig.add_subplot(gs[1, 0])
    plot_acf(x, lags=lags, zero=False, ax=ax2, alpha=.05)

    # PACF
    ax3 = fig.add_subplot(gs[1, 1])
    plot_pacf(x, lags=lags, zero=False, ax=ax3)

    # qq plot
    ax4 = fig.add_subplot(gs[2, 0])
    probplot(x, plot=ax4)
    mean, variance, skewness, kurtosis = moment(x, moment=[1, 2, 3, 4])
    stats1 = f'Mean: {mean:.2f}\nSD: {np.sqrt(variance):.2f}\nSkew: {skewness:.2f}\nKurtosis:{kurtosis:.2f}'
    ax4.text(x=.02, y=.75, s=stats1, transform=ax4.transAxes)
    ax4.set_title("Normal Q-Q")

    # Histogram plus estimated density
    ax5 = fig.add_subplot(gs[2, 1])
    sns.distplot(a=x, hist=True, bins=40, rug=True, label="KDE", color="green",
                 kde=True, fit=norm, ax=ax5)
    ax5.set_title("Histogram plus Estimated Density")
    ax5.legend(loc="best")
    fig.suptitle(title, fontsize=20)

    plt.show()


def unit_root_test(df, title):
    result = adfuller(df)
    print("---------------------------------")
    print(title)
    print('ADF Statistic:%.4f' % result[0]),
    print('p-value: %.4f' % result[1]),
    print('\nCritical Values:'),
    for key, value in result[4].items():
        print('\t%s: %.4f' % (key, value))
    print("---------------------------------")


correlogram(prices["HURST_EXPONENT"][101:], lags=50, title="HURST EXPONENT")
unit_root_test(prices["HURST_EXPONENT"][101:], title="HURST EXPONENT")
