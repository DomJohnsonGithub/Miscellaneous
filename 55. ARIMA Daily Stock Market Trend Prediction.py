import numpy as np
from pathlib import Path
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

import warnings

warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)

# ----- Import Data ----- #
DATA_STORE = Path("C:\\Users\\domin\\PycharmProjects\\Miscellaneous\\dow_stocks.h5")

with pd.HDFStore(DATA_STORE, "r") as store:
    df = store.get("DOW/stocks")

idx = pd.IndexSlice
df = df.loc[idx["HD", :], ["close", "volume"]].droplevel(0)

print(df.isnull().any())
print(df.isnull().sum())

print(df)


# ----- Plot the Data ----- #
def plot_price_vol(price, vol):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), dpi=115, sharex=True)
    price.plot(ax=axes[0], label="Home Depot, Inc. Daily Adj Close Price", lw=0.678, c="burlywood")
    axes[0].grid(True, axis="both")
    axes[0].set_title("HD Daily Price Series")
    axes[0].legend(loc="best", fontsize=6)
    vol.plot(ax=axes[1], label="volume", color="orange", lw=0.5)
    axes[1].legend(loc="best")
    plt.grid(True, axis="y")
    plt.show()


plot_price_vol(df.close, df.volume)

# ----- ACF and PACF ----- #
"""
Generate ACF plots for different lags p = {1, 2, 3...} and list down
the top 5 optimal lags in sorted order of decreasing ACF.
"""
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 9), dpi=95, sharex=True)
sm.graphics.tsa.plot_acf(df.close.values.squeeze(), lags=50, ax=axes[0])
sm.graphics.tsa.plot_pacf(df.close.values.squeeze(), lags=50, ax=axes[1])
plt.show()

# ----- Case 1: p=1, q=0 ----- #
arima_mod01 = sm.tsa.ARIMA(df.close, order=(1, 1, 0)).fit()
print(arima_mod01.params)
print(arima_mod01.summary())
print("Durbin Watson:",
      sm.stats.durbin_watson(
          arima_mod01.resid.values))  # measure of autocorrelation in the residuals (< 2 indicates autocorrelation)
print("^ measures autocorrelation in residuals (below 2 indicates autocorrelation)")

fig = plt.figure(figsize=(10, 6), dpi=100)
ax = fig.add_subplot(111)
ax = arima_mod01.resid.plot(ax=ax, label="Residuals of ARIMA(1, 1, 0) model")
plt.legend(loc="best")
plt.show()

# The Ljung-Box test is a statistical test of whetherany of a group of autocorrelations of a time seires
# are different from zero
resid = arima_mod01.resid
r, q, p = sm.tsa.acf(resid, qstat=True)

data = np.c_[range(1, 37), r[1:], q, p]
table = pd.DataFrame(data, columns=["lag", "AC", "Q", "Prob(>Q)"])
print(table.set_index("lag"))

# ----- Seasonal Decompose ----- #
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df.close, period=12)
fig = plt.figure()
fig = decomposition.plot()
plt.show()

# ----- Test Stationarity ----- #
from statsmodels.tsa.stattools import adfuller


def test_stationarity(ts):
    # Determining Rolling Statistics
    rolmean = ts.rolling(window=12).mean()
    rolstd = ts.rolling(window=12).std()

    # Plot the rolling stats
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(ts, color="b", label="Original Price Series of Home Depot")
    mean = plt.plot(rolmean, color="red", label="Rolling Mean")
    std = plt.plot(rolstd, color="black", label="Rolling Standard Deviation")
    plt.legend(loc="best")
    plt.title("Rolling Mean & StdDev")
    plt.show()

    # Perform ADF Test:
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(ts, autolag="AIC")
    dfoutput = pd.Series(dftest[0:4], index=["Test Statistic", "p-value", "# Lags Used", "No. of Obs. Used"])
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)


test_stationarity(df.close)

data_log = df.close.apply(lambda x: np.log(x))
test_stationarity(data_log)

returns = df.close.diff()
returns.dropna(inplace=True)
test_stationarity(returns)

# ----- SARIMAX ----- #
sarimax = sm.tsa.statespace.SARIMAX(df.close, trend="n", order=(1, 0, 1),
                                    seasonal_order=(1, 0, 1, 12), enforce_stationarity=False)
results = sarimax.fit()
print(results.summary())

sarimax = sm.tsa.statespace.SARIMAX(df.close, trend="ct", order=(1, 0, 1),
                                    seasonal_order=(1, 0, 1, 12), enforce_stationarity=False)
results = sarimax.fit()
print(results.summary())

print("-------------------------")
print("-------------------------")
# ----- CHOICE OF ARMA MODEL ----- #
modARMA101 = sm.tsa.statespace.SARIMAX(df.close, trend="n", order=(1, 0, 1), enforce_stationarity=False)
resultARMA101 = modARMA101.fit()
print(resultARMA101.summary())
print("Durbin Watson:", sm.stats.durbin_watson(resultARMA101.resid.values),
      ". Below 2 indicates autocorrelation in the residuals.")

modARMA201 = sm.tsa.statespace.SARIMAX(df.close, trend="n", order=(2, 0, 1), enforce_stationarity=False)
resultARMA201 = modARMA201.fit()
print(resultARMA201.summary())
print("Durbin Watson:", sm.stats.durbin_watson(resultARMA201.resid.values),
      ". Below 2 indicates autocorrelation in the residuals.")

modARMA301 = sm.tsa.statespace.SARIMAX(df.close, trend="n", order=(3, 0, 1), enforce_stationarity=False)
resultARMA301 = modARMA301.fit()
print(resultARMA301.summary())
print("Durbin Watson:", sm.stats.durbin_watson(resultARMA301.resid.values),
      ". Below 2 indicates autocorrelation in the residuals.")

modARMA401 = sm.tsa.statespace.SARIMAX(df.close, trend="n", order=(4, 0, 1), enforce_stationarity=False)
resultARMA401 = modARMA401.fit()
print(resultARMA401.summary())
print("Durbin Watson:", sm.stats.durbin_watson(resultARMA401.resid.values),
      ". Below 2 indicates autocorrelation in the residuals.")

modARMA501 = sm.tsa.statespace.SARIMAX(df.close, trend="n", order=(5, 0, 1), enforce_stationarity=False)
resultARMA501 = modARMA501.fit()
print(resultARMA501.summary())
print("Durbin Watson:", sm.stats.durbin_watson(resultARMA501.resid.values),
      ". Below 2 indicates autocorrelation in the residuals.")

modARMA102 = sm.tsa.statespace.SARIMAX(df.close, trend="n", order=(1, 0, 2), enforce_stationarity=False)
resultARMA102 = modARMA102.fit()
print(resultARMA102.summary())
print("Durbin Watson:", sm.stats.durbin_watson(resultARMA102.resid.values),
      ". Below 2 indicates autocorrelation in the residuals.")

modARMA202 = sm.tsa.statespace.SARIMAX(df.close, trend="n", order=(2, 0, 2), enforce_stationarity=False)
resultARMA202 = modARMA202.fit()
print(resultARMA202.summary())
print("Durbin Watson:", sm.stats.durbin_watson(resultARMA202.resid.values),
      ". Below 2 indicates autocorrelation in the residuals.")

modARMA302 = sm.tsa.statespace.SARIMAX(df.close, trend="n", order=(3, 0, 2), enforce_stationarity=False)
resultARMA302 = modARMA302.fit()
print(resultARMA302.summary())
print("Durbin Watson:", sm.stats.durbin_watson(resultARMA302.resid.values),
      ". Below 2 indicates autocorrelation in the residuals.")

modARMA402 = sm.tsa.statespace.SARIMAX(df.close, trend="n", order=(4, 0, 2), enforce_stationarity=False)
resultARMA402 = modARMA402.fit()
print(resultARMA402.summary())
print("Durbin Watson:", sm.stats.durbin_watson(resultARMA402.resid.values),
      ". Below 2 indicates autocorrelation in the residuals.")

modARMA502 = sm.tsa.statespace.SARIMAX(df.close, trend="n", order=(5, 0, 2), enforce_stationarity=False)
resultARMA502 = modARMA502.fit()
print(resultARMA502.summary())
print("Durbin Watson:", sm.stats.durbin_watson(resultARMA502.resid.values),
      ". Below 2 indicates autocorrelation in the residuals.")

aic = {
    "SARIMAX (1, 0, 1)": resultARMA101.aic,
    "SARIMAX (2, 0, 1)": resultARMA201.aic,
    "SARIMAX (3, 0, 1)": resultARMA301.aic,
    "SARIMAX (4, 0, 1)": resultARMA401.aic,
    "SARIMAX (5, 0, 1)": resultARMA501.aic,
    "SARIMAX (1, 0, 2)": resultARMA102.aic,
    "SARIMAX (2, 0, 2)": resultARMA202.aic,
    "SARIMAX (3, 0, 2)": resultARMA302.aic,
    "SARIMAX (4, 0, 2)": resultARMA402.aic,
    "SARIMAX (5, 0, 2)": resultARMA502.aic
}

min_aic = min(zip(aic.values(), aic.keys()))
print("\nMinimum AIC (Best Model Fit):", min_aic)

# ----- CHOICE OF ARIMA MODEL ----- #
"""
Various transformations to stationarize data:
1. Deflation by CPI
2. Logarithmic
3. First Difference
4. Seasonal Difference
5. Seasonal Adjustment
"""
# ----- ARIMA ----- #
df["returns"] = df.close.diff()
df.dropna(inplace=True)
test_stationarity(df.returns)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), dpi=105)
sm.graphics.tsa.plot_acf(df.returns.values.squeeze(), ax=axes[0], lags=50)
sm.graphics.tsa.plot_pacf(df.returns.values.squeeze(), ax=axes[1], lags=50)
plt.show()

import pmdarima as pm
from pmdarima.arima.utils import ndiffs

adf_test = pm.arima.stationarity.ADFTest(alpha=.05)  # test whether we should difference at the 5% significance level
p_val, should_diff = adf_test.should_diff(df.close)
print("Original Close Price Series:")
print("p-val: ", p_val, "should_diff: ", should_diff)

# Estimate the number of differences using an ADF test:
n_adf = ndiffs(df.close, test='adf')  # -> 0
# Or a KPSS test (auto_arima default):
n_kpss = ndiffs(df.close, test='kpss')  # -> 0
# Or a PP test:
n_pp = ndiffs(df.close, test='pp')  # -> 0
print("Estimated no. of difference using test on Original Price Series:")
print(f"ADF Test: {n_adf}, KPSS Test: {n_kpss}, PP Test: {n_pp}")

adf_test = pm.arima.stationarity.ADFTest(alpha=.05)  # test whether we should difference at the 5% significance level
p_val, should_diff = adf_test.should_diff(df.returns)
print("Differenced Series (Returns):")
print("p-val: ", p_val, "should_diff: ", should_diff)

# Estimate the number of differences using an ADF test:
n_adf = ndiffs(df.returns, test='adf')  # -> 0
# Or a KPSS test (auto_arima default):
n_kpss = ndiffs(df.returns, test='kpss')  # -> 0
# Or a PP test:
n_pp = ndiffs(df.returns, test='pp')  # -> 0
print("Estimated no. of difference using test on Differenced Series (Returns):")
print(f"ADF Test: {n_adf}, KPSS Test: {n_kpss}, PP Test: {n_pp}")

decomposition = seasonal_decompose(x=df.close, model="multiplicative", period=252, extrapolate_trend="freq")
fig = plt.figure()
decomposition.plot()
plt.show()

data = df.close.copy()
data = data.resample("M").mean()
print(data)

decomposition = seasonal_decompose(x=data, model="multiplicative", period=12, extrapolate_trend="freq")
fig = plt.figure()
decomposition.plot()
plt.show()

stepwise_model = pm.auto_arima(data,
                               start_p=1,
                               start_q=1,
                               max_p=5,
                               max_q=5,
                               max_d=2,
                               m=12,
                               start_P=0,
                               start_Q=0,
                               max_Q=3,
                               max_P=3,
                               stationary=False,
                               seasonal=True,
                               d=1, D=1,
                               trace=True,
                               information_criterion="aic",
                               error_action="ignore",
                               suppress_warnings=True,
                               stepwise=True)

print(stepwise_model.aic())

print(stepwise_model.summary())

stepwise_model.plot_diagnostics(figsize=(7, 5))
plt.show()

# ----- Train Test Split ----- #
size = 0.8
train = data[:int(len(data) * size)]
test = data[int(len(data) * size):]

stepwise_model.fit(train)
future_forecast = stepwise_model.predict(,,,,
print(future_forecast)

future_forecast = pd.DataFrame(future_forecast, index=test.index, columns=["Prediction"])

pd.concat([test, future_forecast], axis=1).plot()
plt.show()

pd.concat([data, future_forecast], axis=1).plot()
plt.show()


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)  # ME
    mae = np.mean(np.abs(forecast - actual))  # MAE
    mpe = np.mean((forecast - actual) / actual)  # MPE
    rmse = np.mean((forecast - actual) ** 2) ** .5  # RMSE
    return ({'mape': mape, 'me': me, 'mae': mae,
             'mpe': mpe, 'rmse': rmse})


for i, val in forecast_accuracy(future_forecast.values, test.values).items():
    print(i, val)

# Now we've evaluated on our test data and our satisfied with our model,
# we refit our model to our entire data set and then forecast into the future
stepwise_model.fit(data)
future_forecast = stepwise_model.predict(,,,,
print(data.index[-1:], data[-1:])
print(future_forecast)  # home depot price @ 5 months into 2019 is around 190

# ----- Instead of using the monthly prices ----- #
# model with the daily prices using auto_arima
size = 0.8
train = df.close[:int(len(df.index) * size)]
test = df.close[int(len(df.index) * size):]
print("Length of test set:", len(test))

daily_model = pm.arima.auto_arima(train, start_p=1, start_q=1,
                                  test="adf",
                                  max_p=4, max_q=4,
                                  d=None, seasonal=False,
                                  start_P=0, D=0, trace=True,
                                  error_action="ignore",
                                  suppress_warnings=True,
                                  stepwise=True)

print(daily_model.aic())
print(daily_model.summary())
daily_model.plot_diagnostics(figsize=(7, 5))
plt.show()

daily_model.fit(train)

forecast = daily_model.predict(,,,,
forecast = pd.DataFrame(forecast.values, index=test.index, columns=["Prediction"])

# plot prediction on the test set
plt.plot(train, label="train")
plt.plot(test, label="test")
plt.plot(forecast, label="Prediction")
plt.show()

for i, val in forecast_accuracy(forecast.values, test.values).items():
    print(i, val)

daily_model.fit(df.close)
forecast_2day = daily_model.predict(,,,,
forecast_5day = daily_model.predict(,,,,  # 1 week
forecast_10day = daily_model.predict(,,,,  # 2 week
forecast_21day = daily_model.predict(,,,,  # 1 month
forecast_63day = daily_model.predict(,,,,  # quarterly prediction
forecast_126day = daily_model.predict(,,,,  # semi_annual prediction
forecast_252day = daily_model.predict(,,,,  # annual pred.

preds = {
    "2day": forecast_2day,
    "5day": forecast_5day,
    "10day": forecast_10day,
    "21day": forecast_21day,
    "63day": forecast_63day,
    "126day": forecast_126day,
    "252day": forecast_252day,
}

for i, val in preds.items():
    print(i, val.values[-1])
