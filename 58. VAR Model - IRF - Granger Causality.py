import os
import sys
import warnings
from datetime import date
import pandas as pd
import pandas_datareader.data as web
import numpy as np
from datetime import datetime
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.gridspec as gridspec
import seaborn as sns
import yfinance as yf
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.api import VAR, VARMAX
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import probplot, moment
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# ----- VAR MODEL FOR FORECASTS ----- #

# ----- Helper Function ----- #
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


def calculate_model_accuracy_metrics(actual, predicted):
    """
    Output model accuracy metrics, comparing predicted values
    to actual values.
    Arguments:
        actual: list. Time series of actual values.
        predicted: list. Time series of predicted values
    Outputs:
        Forecast bias metrics, mean absolute error, mean squared error,
        and root mean squared error in the console
    """
    # Calculate forecast bias
    forecast_errors = [actual[i] - predicted[i] for i in range(len(actual))]
    bias = sum(forecast_errors) * 1.0 / len(actual)
    print('Bias: %f' % bias)
    # Calculate mean absolute error
    mae = mean_absolute_error(actual, predicted)
    print('MAE: %f' % mae)
    # Calculate mean squared error and root mean squared error
    mse = mean_squared_error(actual, predicted)
    print('MSE: %f' % mse)
    rmse = np.sqrt(mse)
    print('RMSE: %f' % rmse)


def granger_causation_matrix(data, variables, maxlag, verbose):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c, r in zip(df.columns, df.index):
        test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=verbose)
        p_values = [round(test_result[i + 1][0]["ssr_chi2test"][1], 5) for i in range(maxlag)]
        if verbose: print(f"Y = {r}, X = {c}, P Values = {p_values}")
        min_p_value = np.min(p_values)
        df.loc[r, c] = min_p_value
    df.columns = [var + "_x" for var in variables]
    df.index = [var + "_y" for var in variables]
    return df


# ----- Get Data (TS) ----- #
start, end = datetime(2010, 1, 1), datetime(2020, 1, 1)
tickers = ["TMO", "DHR", "ILMN", "IDXX", "A"]
df = yf.download(tickers, start, end).squeeze().drop(
    columns=["High", "Open", "Low", "Close", "Volume"]).rename(columns={"Adj Close": "close"})
df = df.stack(level=1).swaplevel(i=0, j=1)
idx = pd.IndexSlice


# Log Returns
def create_adjclose_df(df, tickers):
    idx = pd.IndexSlice
    ts = [df.loc[idx[i, :], "close"].droplevel(0) for i in tickers]
    return ts


data = create_adjclose_df(df=df, tickers=tickers)
data = pd.DataFrame([data[0], data[1], data[2], data[3], data[4]]).T
data.columns = tickers
log_rets = np.log(data).diff().mul(100).dropna()

# Also Log Returns
dhr = np.log(df.loc[idx["DHR", :], "close"].droplevel(0)).diff().dropna().mul(100)  # rescale to facilitate optimization
tmo = np.log(df.loc[idx["TMO", :], "close"].droplevel(0)).diff().dropna().mul(100)  # rescale to facilitate optimization
ilmn = np.log(df.loc[idx["ILMN", :], "close"].droplevel(0)).diff().dropna().mul(
    100)  # rescale to facilitate optimization
idxx = np.log(df.loc[idx["IDXX", :], "close"].droplevel(0)).diff().dropna().mul(
    100)  # rescale to facilitate optimization
a = np.log(df.loc[idx["A", :], "close"].droplevel(0)).diff().dropna().mul(100)  # rescale to facilitate optimization

# Dict of log_returns for each price series
stocks = {"dhr": dhr,
          "tmo": tmo,
          "ilmn": ilmn,
          "idxx": idxx,
          "a": a}

# Visualize the Original Prices and their correlations
plt.figure()
for i in tickers:
    stock_prices = df.loc[idx[i, :], "close"].droplevel(0)
    plt.plot(stock_prices, label=i)
plt.legend(loc="best")
plt.title("Daily Prices of Each Price Series (Sector: Healthcare, Industry: Medical Diagnostics & Research)")
plt.grid(True)
plt.show()

# Spearman Rank Correlation
close = df.unstack(level=0).droplevel(0, axis=1)
close.columns = ["A", "DHR", "IDXX", "ILMN", "TMO"]
sprmn_corr = close.corr("spearman")
print("Spearman Rank Corr Coef's:")
print(sprmn_corr)
sns.heatmap(sprmn_corr, annot=True)
plt.title("Spearman Rank Correlation Coefficients")
plt.show()

# Rolling Correlation
tmo_a = close["A"].rolling(window=14).corr(close["TMO"], "spearman").dropna()
tmo_dhr = close["DHR"].rolling(window=14).corr(close["TMO"], "spearman").dropna()
tmo_idxx = close["IDXX"].rolling(window=14).corr(close["TMO"], "spearman").dropna()
tmo_ilmn = close["ILMN"].rolling(window=14).corr(close["TMO"], "spearman").dropna()

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
tmo_a.plot(ax=axes[0], lw=1., label="TMO & A")
axes[0].axhline(y=0.9, c="r", linestyle="--")
axes[0].axhline(y=0.5, c="r", linestyle="--")
axes[0].axhline(y=0.0, c="magenta", linestyle="-.")
axes[0].legend(loc="best")
tmo_dhr.plot(ax=axes[1], lw=1., label="TMO & DHR")
axes[1].axhline(y=0.9, c="r", linestyle="--")
axes[1].axhline(y=0.5, c="r", linestyle="--")
axes[1].axhline(y=0.0, c="magenta", linestyle="-.")
axes[1].legend(loc="best")
tmo_idxx.plot(ax=axes[2], lw=1., label="TMO & ILMN")
axes[2].axhline(y=0.9, c="r", linestyle="--")
axes[2].axhline(y=0.5, c="r", linestyle="--")
axes[2].axhline(y=0.0, c="magenta", linestyle="-.")
axes[2].legend(loc="best")
tmo_ilmn.plot(ax=axes[3], lw=1., label="TMO & IDXX")
axes[3].axhline(y=0.9, c="r", linestyle="--")
axes[3].axhline(y=0.5, c="r", linestyle="--")
axes[3].axhline(y=0.0, c="magenta", linestyle="-.")
axes[3].legend(loc="best")
plt.suptitle("Rolling Correlation")
plt.show()

from scipy.stats import norm

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), dpi=100)
sns.distplot(tmo_a.values, ax=axes[0, 0], bins=50, hist=True, kde=True, fit=norm, color="red", label="TMO & A")
axes[0, 0].legend(loc="best")
axes[0, 0].axvline(x=0.5, c="blue", linestyle="--")
sns.distplot(tmo_dhr.values, ax=axes[0, 1], bins=50, hist=True, kde=True, fit=norm, color="green", label="TMO & DHR")
axes[0, 1].legend(loc="best")
axes[0, 1].axvline(x=0.5, c="blue", linestyle="--")
sns.distplot(tmo_ilmn.values, ax=axes[1, 0], bins=50, hist=True, kde=True, fit=norm, color="orange", label="TMO & ILMN")
axes[1, 0].legend(loc="best")
axes[1, 0].axvline(x=0.5, c="blue", linestyle="--")
sns.distplot(tmo_idxx.values, ax=axes[1, 1], bins=50, hist=True, kde=True, fit=norm, color="purple", label="TMO & IDXX")
axes[1, 1].legend(loc="best")
axes[1, 1].axvline(x=0.5, c="blue", linestyle="--")
plt.suptitle("Distribution of the 14-Day Rolling Spearman Rank Correlation Coefficients")
plt.show()

from scipy.stats import spearmanr, kendalltau

tickers = ["TMO", "DHR", "ILMN", "IDXX", "A"]
for i in tickers[1:]:
    sprmn_coef, p = spearmanr(close[i], close["TMO"])
    kndll_coef, p1 = kendalltau(close[i], close["TMO"])
    alpha = 0.05
    print("----------------------------------------------------------------------")
    print("\nSpearmans Correlation Coefficient: %.3f" % sprmn_coef, f"({i} & TMO)")
    if p > alpha:
        print("Samples are uncorrelated (fail to reject H0) p=%.3f" % p)
    else:
        print("Samples are correlated (reject H0) p=%.3f" % p)

    print("Kendall's Tau Correlation Coefficient: %.3f" % kndll_coef, f"({i} & TMO)")
    if p1 > alpha:
        print("Samples are uncorrelated (fail to reject H0) p=%.3f" % p1)
    else:
        print("Samples are correlated (reject H0) p=%.3f" % p1)
    print("----------------------------------------------------------------------")

# ----- ADF Test for Stationarity & Plots of Data Characteristics ----- #
# Plot returns the series, acf, pacf, qqplot, density plot
for key, value in stocks.items():
    unit_root_test(value, title=f"({key.upper()}), Daily Log Returns")
    correlogram(value, lags=50, title=f"('{key.upper()}'), Daily Log Returns")

# ----- VAR MODEL ----- #
# Train, Test split
n_obs = 21
train = log_rets[:-n_obs]
test = log_rets[-n_obs:]

# Initiate the VAR Model
from statsmodels.tsa.vector_ar import var_model

model = var_model.VAR(endog=train)
res = model.select_order(20)
print(res.summary())


# Fit to a VAR Model
model_fit = model.fit(maxlags=0, ic="aic")
print(model_fit.summary())

# Residual Plots
residuals = model_fit.resid
fig, axes = plt.subplots(2, 1)
axes[0].plot(residuals["TMO"])
axes[1].plot(log_rets["TMO"])
plt.suptitle("RESIDUALS VS LOG_RETURNS")
plt.show()

y_fitted = model_fit.fittedvalues
plt.figure(figsize=(15, 5))
plt.plot(residuals, label='resid')
plt.plot(y_fitted, label='VAR prediction')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel('Residuals')
plt.legend()
plt.grid(True)
plt.show()

for i in tickers:
    correlogram(x=residuals.loc[:, i], lags=50, title=f"{i} Residual Plots")

# Durbin Watson Statistic
"""
Durbin-Watson Statistic is related to related to auto correlation.

The Durbin-Watson statistic will always have a value between 0 and 4. A value of 2.0 means that there is no 
auto-correlation detected in the sample. Values from 0 to less than 2 indicate positive auto-correlation and 
values from 2 to 4 indicate negative auto-correlation. A rule of thumb is that test statistic values in the 
range of 1.5 to 2.5 are relatively normal. Any value outside this range could be a cause for concern.

A stock price displaying positive auto-correlation would indicate that the price yesterday has a positive 
correlation on the price today — so if the stock fell yesterday, it is also likely that it falls today. 
A stock that has a negative auto-correlation, on the other hand, has a negative influence on itself over 
time — so that if it fell yesterday, there is a greater likelihood it will rise today.
"""
from statsmodels.stats.stattools import durbin_watson

out = durbin_watson(residuals)

for key, value in zip(log_rets.columns, out):
    print((key), ":", round(value, 3), "==> if DW spans (1.5, 2.5), rule of thumb says is relatively normal residuals,"
                                       "therefore can forecast now")

# ----- PREDICTION ----- #
# res.k_ar is the lag order
# Forecasting
print(res.selected_orders)
print(res.k_ar)

model_fit.plot_forecast(n_obs)
plt.show()

pred = res.forecast(train.values[-res.k_ar:], steps=n_obs)
pred = (pd.DataFrame(pred, index=test.index, columns=test.columns + "_pred"))
print(pred)


# Invert the transform
def invert_transform(train, pred):
    forecast = pred.copy()
    columns = train.columns
    for col in columns:
        forecast[str(col) + "_pred"] = train.iloc[-1] + forecast[str(col) + "_pred"].cumsum()
    return forecast


output = invert_transform(train, pred)
output = np.exp(output)
print(output)

# combining predicted and real data set
combine = pd.concat([output['TMO_pred'], test['TMO']], axis=1)
combine['accuracy'] = round(combine.apply(lambda row: row.TMO_pred / row.TMO * 100, axis=1), 2)
combine['accuracy'] = pd.Series(["{0:.2f}%".format(val) for val in combine['accuracy']], index=combine.index)
combine = combine.round(decimals=2)
combine = combine.reset_index()
combine = combine.sort_values(by='Date', ascending=False)

# Evaluation
# Forecast bias
"""
“Least squares parameter estimation of dynamic regression models is known to 
exhibit substantial bias in small samples when the data is fairly persistent”
"""
forecast_errors = [combine['TMO'][i] - combine['TMO_pred'][i] for i in range(len(combine['TMO']))]
bias = sum(forecast_errors) * 1.0 / len(combine['TMO'])
print('Bias: %f' % bias)
# Mean absolute error tells us how big of an error we can expect from the forecast on average.
print('Mean absolute error:', mean_absolute_error(combine['TMO'].values, combine['TMO_pred'].values))
print('Mean squared error:', mean_squared_error(combine['TMO'].values, combine['TMO_pred'].values))
print('Root mean squared error:', np.sqrt(mean_squared_error(combine['TMO'].values, combine['TMO_pred'].values)))

# ----- IMPULSE RESPONSE ANALYSIS ----- #
"""
1.The IRA quantifies the reaction of every single variable in the model on an exogenous shock to the model
2.IRA helps us to trace the transmission of a single shock within an otherwise noisy system of equations and 
  useful in assessment of economic policies
3.Cases of shocks
a) Single equation shock : we investigate forecast error impulse responses
b) Joint equation shock: shock mirrors the residual covariance structure, we investigate orthogonalized impulse responses.

Forecast error impulse response:

- Variables in a VAR model is a function of each other and individual coefficient estimates only 
  provide limited information on the reaction of the system to a shock. Hence, Impulse responses 
  are used to understand the dynamic behaviour of the model
- In order to get a better picture of the model’s dynamic behaviour, impulse responses (IR) are used.
- The departure point of every IRF for a linear VAR model is moving average (MA) representation
- It is also called as forecast error impulse response (FEIR) function.
"""

irf = model_fit.irf(10)
irf.plot(orth=False)
plt.show()

irf.plot(impulse="TMO")
plt.show()

irf.plot_cum_effects(orth=False)
plt.show()

# ----- FORECAST ERROR VARIANCE DECOMPOSITION ----- #
fevd = model_fit.fevd(5)
print(fevd.summary())

model_fit.fevd(20).plot()
plt.show()

# ----- STATISTICAL TESTS ----- #
# Granger Causality
print(model_fit.test_causality("TMO", ["DHR", "IDXX", "ILMN", "A"], kind="f"))

# Normality
print(model_fit.test_normality())

# Whiteness of residuals
print(model_fit.test_whiteness())
