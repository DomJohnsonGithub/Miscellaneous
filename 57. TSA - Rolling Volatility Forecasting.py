import os
import sys
import warnings
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
from numpy.linalg import LinAlgError

warnings.filterwarnings('ignore')
import statsmodels.tsa.api as tsa  # time_series_models
import statsmodels.api as sm  # cross_sectional_models
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_acf, acf, plot_pacf, \
    pacf  # plots for autocorrelation and Partial autocorrelation function
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from scipy.stats import probplot, moment
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import yfinance as yf
from arch import arch_model
from arch.univariate import ConstantMean, GARCH, Normal

warnings.filterwarnings('ignore')

start, end = datetime(2010, 1, 1), datetime(2020, 1, 1)
tmo = yf.download("TMO", start, end).squeeze().drop(columns=["High", "Open", "Low", "Close"]).rename(columns={
    "Volume": "volume", "Adj Close": "close"})

tmo = pd.Series(np.log(tmo.close).diff().dropna().mul(100))  # rescale to facilitate optimization
print(tmo)


def correlogram(x, lags, title):
    fig = plt.figure()
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
    plt.tight_layout(pad=0.04, h_pad=0.5, w_pad=0.5)
    plt.show()


correlogram(tmo, lags=50, title="Time Out Group PLC Daily Log Returns")
correlogram(tmo.sub(tmo.mean()).pow(2), lags=100, title="TMO Daily Volatility")

"""
Order Selection: rolling out-of-sample forecasts:

1. The objective is to estimate the GARCH model that captures the linear relationship of past volatilities.
2. Rolling 10-year windows is the train data, p and q ranges from 1-4 to generate 1-step out-of-sample forecasts.
3. Compare the RMSE of the predicted volatility of GARCH with the actual squared deviation of the return from its mean
4. Winsorize the data to eliminate fat distributions
"""
# pilot code - identify the orders to minimize the RMSE
start = datetime(2019, 12, 10)
tmoStock = yf.download("TMO", start=start, end=end).squeeze().drop(
    columns=["High", "Low", "Close", "Open", "Volume"]).rename(
    columns={"Adj Close": "close"}).dropna()
log_tmo = np.log(tmoStock)
log_tmo_diff = np.log(tmoStock).diff().dropna().mul(100)

print(log_tmo_diff)

trainsize = 5
data = log_tmo_diff.clip(lower=log_tmo_diff.quantile(0.05),
                         upper=log_tmo_diff.quantile(0.95), axis=1)

T = len(data)
test_results = {}

for p in range(1, 5):
    for q in range(1, 5):
        print(f'{p} | {q}')
        result = []
        for s, t in enumerate(range(trainsize, T - 1)):
            print("s", s)
            print("t", t)
            train_set = data.iloc[s: t]
            print("train", train_set)
            test_set = data.iloc[t + 1]  # 1-step ahead forecast
            print("test", test_set)
            model = arch_model(y=train_set, p=p, q=q).fit(disp='off')
            forecast = model.forecast(horizon=1)
            mu = forecast.mean.iloc[-1, 0]
            print("mean", mu)
            var = forecast.variance.iloc[-1, 0]
            print("variance", var)
            result.append([(test_set - mu) ** 2, var])
        df = pd.DataFrame(result, columns=['y_true', 'y_pred'])
        test_results[(p, q)] = np.sqrt(mean_squared_error(df.y_true, df.y_pred))

s = pd.Series(test_results)
s.index.names = ['p', 'q']
s = s.unstack().sort_index(ascending=False)
sns.heatmap(s, cmap='coolwarm', annot=True, fmt='.4f')
plt.title('Out-of-Sample RMSE');
plt.show()

# ----- Final Code ----- #
print(tmo)
print(tmo.shape)

train_size = int(len(tmo) * 0.75)
final_data = tmo.clip(lower=tmo.quantile(0.05),
                      upper=tmo.quantile(0.95))
total = len(final_data)
test_results = {}
for p in range(1, 5):
    for q in range(1, 5):
        print(p, q)
        result = []
        for x1, x2 in enumerate(range(train_size, total - 1)):
            train_set = final_data.iloc[x1: x2]
            test_set = final_data.iloc[x2 + 1]  # 1-step ahead forecast
            model = arch_model(y=train_set, p=p, q=q).fit(disp='off')
            forecast = model.forecast(horizon=1)
            mu = forecast.mean.iloc[-1, 0]
            var = forecast.variance.iloc[-1, 0]
            result.append([(test_set - mu) ** 2, var])
        df = pd.DataFrame(result, columns=['y_true', 'y_pred'])
        test_results[(p, q)] = np.sqrt(mean_squared_error(df.y_true, df.y_pred))

series = pd.Series(test_results)
series.index.names = ['p', 'q']
series = s.unstack().sort_index(ascending=False)
sns.heatmap(s, cmap='coolwarm', annot=True, fmt='.4f')
plt.title('Out-of-Sample RMSE');
plt.show()

# Estimate GARCH(2, 3) Model
gm = ConstantMean(tmo.clip(lower=tmo.quantile(0.05),
                           upper=tmo.quantile(0.95)))
gm.volatility = GARCH(2, 0, 3)
gm.distribution = Normal()
model = gm.fit(update_freq=10)
print(model.summary())

fig = model.plot(annualize="D")
fig.set_size_inches(12, 8)
fig.tight_layout()
plt.show()

correlogram(model.resid.dropna(), lags=250, title="GARCH(2, 2) Residuals")

# Estimate GARCH(2, 1) Model
gm = ConstantMean(tmo.clip(lower=tmo.quantile(0.05),
                           upper=tmo.quantile(0.95)))
gm.volatility = GARCH(2, 0, 1)
gm.distribution = Normal()
model = gm.fit(update_freq=10)
print(model.summary())

fig = model.plot(annualize="D")
fig.set_size_inches(12, 8)
fig.tight_layout()
plt.show()

correlogram(model.resid.dropna(), lags=250, title="GARCH(2, 1) Residuals")
