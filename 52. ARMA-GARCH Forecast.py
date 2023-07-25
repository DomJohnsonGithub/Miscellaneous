from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import lag_plot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA

# ----- Get Data ----- #
DATA_STORE = Path("C:\\Users\\domin\\PycharmProjects\\Miscellaneous\\dow_stocks.h5")

with pd.HDFStore(DATA_STORE, "r") as store:
    df = store.get("DOW/stocks")

print(df.info())

idx = pd.IndexSlice
df = df.loc[idx[["CSCO"], :], "close"]
df.dropna(inplace=True)
df = df.droplevel(0)
print(df)

# ----- Plot Close Price ----- #
fig = plt.figure(figsize=(10, 6), dpi=120)
plt.plot(df, color="red")
plt.title("CISCO Daily Adj Close Price (2000-2019)")
plt.xlabel("Dates")
plt.ylabel("Adj Close Price")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Visualize the Train and Test Data ----- #
train, test = df[:int(0.8 * len(df))], df[int(0.8 * len(df)):]
fig = plt.figure(figsize=(10, 6), dpi=120)
plt.plot(train, color="red", label="Train data")
plt.plot(test, color="green", label="Test data")
plt.title("CISCO Daily Adj Close Price (2000-2019)")
plt.xlabel("Dates")
plt.ylabel("Adj Close Price")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----- Lag Plots (looking at Autocorrelation) ----- #
plt.rcParams.update({"ytick.left": False, "axes.titlepad": 10})

fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(10, 3), sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:15]):
    lag_plot(df, lag=i + 1, ax=ax, c="firebrick")
    ax.set_title("Lag " + str(i + 1))
    ax.grid(True)
fig.suptitle(
    "Lag Plots of CSCO Daily Adj Close (Points get wide and scattered with increasing lag -> lesser correlation))")
plt.show()

# ----- Checking Correlation by Calculating Covariance ----- #
dates = pd.Series(df.index, index=df.index)
data = pd.concat([df, dates], axis=1)
data["t"] = data.close
shift = [1, 5, 10, 30]
for i in shift:
    data[f"t+{i}"] = data.close.shift(i)
data.drop(columns=["date", "close"], inplace=True)
data.dropna(inplace=True)

result = data.corr()
print(result)


# ----- ARMA ----- #
# Two approaches for stationary:

# 1. Difference (removes underlying seasonal or cyclical patterns in the TS)
def test_stationarity(ts):
    rolmean = pd.Series.rolling(ts, window=12).mean()
    rolstd = pd.Series.rolling(ts, window=12).std()
    fig = plt.figure()
    fig.add_subplot()
    orig = plt.plot(ts, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='rolling mean')
    std = plt.plot(rolstd, color='black', label='Rolling standard deviation')

    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical value (%s)' % key] = value
    print(dfoutput)


ts_log = df
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

"""
Both mean and variance amplitudes of the data reduced against time axis after difference. 
Dickey Fuller test showed the data is stationary in 90% confidence level.
"""

# ----- Finding the p and q terms in ARMA by ACF and PACF ----- #
lag_acf = acf(ts_log_diff, nlags=50)
lag_pacf = pacf(ts_log_diff, nlags=50, method="ols")
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.show()

# ARMA process with data after difference
train, test = df[0:len(df) - 21], df[-21:]

history = [x for x in train]
y = test
predictions = list()
model = ARIMA(history, order=(1, 1, 1))
model_fit = model.fit()
yhat = model_fit.forecast()[0]
predictions.append(yhat)
history.append(y[0])

for i in range(1, len(y)):
    model = ARIMA(history, order=(1, 1, 1))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    obs = y[i]
    history.append(obs)

plt.figure(figsize=(14, 8))
plt.plot(df.index, df, color='green', label='Train Stock Price')
plt.plot(test.index, y, color='red', label='Real Stock Price')
plt.plot(test.index, predictions, color='blue', label='Predicted Stock Price')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 8))
plt.plot(df.index[-100:], df.tail(100), color='green', label='Train Stock Price')
plt.plot(test.index, y, color='red', label='Real Stock Price')
plt.plot(test.index, predictions, color='blue', label='Predicted Stock Price')
plt.legend()
plt.grid(True)
plt.show()

print('MSE: ' + str(mean_squared_error(y, predictions)))
print('MAE: ' + str(mean_absolute_error(y, predictions)))
print('RMSE: ' + str(np.sqrt(mean_squared_error(y, predictions))))

# 2. Seasonal decomposition
decomposition = seasonal_decompose(df[-1000:], model="additive", period=30)
plt.figure(figsize=(20, 10))
fig = decomposition.plot()
plt.show()

"""
Separate original data into three parts as shown in illustration. 
Residual is the rest of data which can be used in the model. 
It is stationary data after eliminating the other two parts.
"""
residual = decomposition.resid
residual.dropna(inplace=True)
test_stationarity(residual)

"""
As shown in the charts, mean and variance have low volatility. 
And Dickey Fuller test showed the data is stationary in 99% confidence level.
"""

# ARIMA process with data with seasonal decomposition
trend = decomposition.trend
seasonal = decomposition.seasonal
train_trend, test_trend = trend[0:(len(trend) - 21)], trend[-21:]
train_seasonal, test_seasonal = seasonal[:(len(seasonal) - 21)], seasonal[-21:]

ts_log = train_trend
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

train_arima = train_trend.dropna()
test_arima = test_trend.dropna()
history = [x for x in train_arima]
y = test_arima

predictions = list()
model = ARIMA(history, order=(2, 1, 2))
model_fit = model.fit()
yhat = model_fit.forecast()[0]
predictions.append(yhat)
history.append(y[0] + test_seasonal[0])

for i in range(1, len(y)):
    model = ARIMA(history, order=(2, 1, 2))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat + test_seasonal[i])
    obs = y[i]
    history.append(obs)

plt.figure(figsize=(14, 8))
plt.plot(df.index, df, color='green', label='Train Stock Price')
plt.plot(y.index, y, color='red', label='Real Stock Price')
plt.plot(y.index, predictions, color='blue', label='Predicted Stock Price')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 8))
plt.plot(df.index[-100:], df.tail(100), color='green', label='Train Stock Price')
plt.plot(df.index[-73:], df.tail(73), color='red', label='Real Stock Price')
plt.plot(y.index, predictions, color='blue', label='Predicted Stock Price')
plt.legend()
plt.grid(True)
plt.show()

print('MSE: ' + str(mean_squared_error(y, predictions)))
print('MAE: ' + str(mean_absolute_error(y, predictions)))
print('RMSE: ' + str(np.sqrt(mean_squared_error(y, predictions))))

# ----- ARMA-GARCH ----- #
returns = pd.DataFrame(np.log(df.diff()).dropna())
returns.rename(columns={"close": "log_return_rate"}, inplace=True)
returns.replace([np.inf, -np.inf], np.nan, inplace=True)
returns.dropna(inplace=True)

returns_array_like = [x for l in returns.values for x in l]
print(returns_array_like)
_, pvalue, *_ = adfuller(returns_array_like)
print("p-value:", pvalue)

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(returns.values, lags=40, alpha=0.05, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(returns.values, lags=40, alpha=0.05, ax=ax2)
plt.show()

import statsmodels.tsa.stattools as sts

resid = sts.arma_order_select_ic(returns, max_ar=4, max_ma=4, ic=["aic", "bic", "hqic"], trend="nc",
                                 fit_kw=dict(method="css"))
print(f"AIC-order: {resid.aic_min_order}")
print(f"BIC-order: {resid.bic_min_order}")
print(f"HQIC-order: {resid.hqic_min_order}")

arma_mod01 = sm.tsa.ARMA(returns, (1, 0)).fit(),
             print(arma_mod01.summary())
print("-------------------------")
print(arma_mod01.params)

# autocorrelation test on the residuals
from statsmodels.stats import diagnostic

resid = arma_mod01.resid
_, pvalue, _, bppvalue = diagnostic.acorr_ljungbox(resid, lags=None, boxpierce=True)
print(pvalue, "\n", bppvalue)

# test ARCH effect on the residuals
*_, fpvalue = diagnostic.het_arch(resid)
print(fpvalue)

fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(111)
fig = sm.graphics.tsa.plot_acf(resid.values ** 2, lags=40, ax=ax1)
plt.show()

from statsmodels.stats.stattools import jarque_bera

_, jbpv, *_ = jarque_bera(returns.values)
print("pvalue-->", jbpv)

from arch import arch_model
from arch.univariate import ZeroMean, GARCH, StudentsT, ConstantMean

arch_mod = ConstantMean(returns)
arch_mod.volatility = GARCH(1, 0, 1)
arch_mod.distribution = StudentsT()
res = arch_mod.fit()
print(res.summary())
print(" ")
print("The estimated parameters: ")
print("---------------------------------")
print(res.params)

mu = arma_mod01.params[0]
theta = arma_mod01.params[1]
omega = res.params[1]
alpha = res.params[2]
beta = res.params[3]
print(f"mu: {mu}, theta: {theta}, omega: {omega}, alpha: {alpha}, beta: {beta}")

sigma_t = res.conditional_volatility.iloc[-1]
sigma_forecast = np.sqrt(omega + alpha * res.resid.iloc[-1] ** 2 + beta * res.conditional_volatility.iloc[-1] ** 2)
epsilon_t = sigma_t * np.random.standard_normal()
epsilon_forecast = sigma_forecast * np.random.standard_normal()
returns_forecast = mu + epsilon_forecast + theta * epsilon_t
print("returns forecast:", returns_forecast)


def returns_predict(period):
    returns_pool = []
    for i in range(period):
        sigma_t = res.conditional_volatility.iloc[-1]
        epsilon_t = sigma_t * np.random.standard_normal()
        sigma_forecast = np.sqrt(omega + alpha * epsilon_t ** 2 + beta * sigma_t ** 2)
        epsilon_forecast = sigma_forecast * np.random.standard_normal()
        returns_forecast = mu + epsilon_forecast + theta * epsilon_t
        returns_pool.append(returns_forecast)
        sigma_t = sigma_forecast
    return returns_pool


train_returns = pd.DataFrame(np.log(test).diff().dropna())
train_returns.rename(columns={'close': 'log_return_rate'}, inplace=True)

plt.figure(figsize=(14, 8))
fig = plt.figure()
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
ax = fig.add_subplot(111)
ax.plot(df.index[-73:-63], train_returns['log_return_rate'][:10], color='green', label='Real Stock Price')
ax2 = ax.twinx()
ax2.plot(df.index[-73:-63], returns_predict(10), color='red', label='Predicted Retrun')
ax.legend(loc=(0.02, 0.95))
ax.grid()
ax.set_xlabel("Time")
ax.set_ylabel("Return")
ax2.set_yticks([])
ax2.legend(loc=(0.02, 0.91))
plt.show()
