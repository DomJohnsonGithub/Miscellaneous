import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima.utils import ndiffs
import statistics as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf
import warnings
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)

# Import Data
DATA_STORE = Path("C:\\Users\\domin\\PycharmProjects\\Miscellaneous\\dow_stocks.h5")

with pd.HDFStore(DATA_STORE, "r") as store:
    df = store.get("DOW/stocks")

# Select an Individual Stock to Forecast
idx = pd.IndexSlice
df = df.loc[idx["NKE", :], "close"]

# Drop MultiIndex
df = df.droplevel(0)

# Resample to Monthly Data
df = df.resample("M").last()

# ADF Test for Stationarity
result = adfuller(df)
print("ADF Statistic: %f" % result[0])
print("p-value: %f" % result[1])

if result[1] >= 0.05:
    print(f"\np-value is {result[1]} >= 0.05. Therefore, differencing is required")
else:
    print(f"\np-value is {result[1]} <= 0.05. Therefore, the time series is stationary")

# Create DF with the original and differenced series
df = pd.concat([df, df.diff(), df.diff(2)], axis=1)
df.columns = ["close", "1st_diff", "2nd_diff"]
df.dropna(inplace=True)
df.reset_index(inplace=True)

# as p-value >= 0.05 (significance level), difference the series and see the ACF Plot
plt.rcParams.update(({"figure.figsize": (10, 8), "figure.dpi": 120}))

# Original Series
fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True)
axes[0, 0].plot(df.close)
axes[0, 0].set_title("Original Monthly Nike Close Price Series")
plot_acf(df.close, lags=200, use_vlines=True, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df["1st_diff"])
axes[1, 0].set_title("1st Differencing")
plot_acf(df["1st_diff"], lags=200, use_vlines=True, ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df["2nd_diff"])
axes[2, 0].set_title("2nd Differencing")
plot_acf(df["2nd_diff"], lags=200, use_vlines=True, ax=axes[2, 1])
plt.show()

y = df.close
# Adf Test, KPSS Test, PP Test
print("ADF Test:", ndiffs(y, test="adf"), "KPSS Test:", ndiffs(y, test="kpss"), "PP Test:", ndiffs(y, test="pp"))
x = [ndiffs(y, test="adf"), ndiffs(y, test="kpss"), ndiffs(y, test="pp")]
print("\nFrom all 3 stationarity tests, the most common no. of differencing terms is:", st.mode(x))

# Order of AR term
# PACF Plot of 1st Differenced Series
plt.rcParams.update({"figure.figsize": (9, 3), "figure.dpi": 120})

fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(df["1st_diff"])
axes[0].set_title("1st Differencing")
axes[1].set(ylim=(0, 5))
plot_pacf(df["1st_diff"], lags=(0.5 * len(df)) - 1, use_vlines=True, ax=axes[1])
plt.show()

# Order of MA term
plt.rcParams.update({"figure.figsize": (9, 3), "figure.dpi": 120})
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df["1st_diff"])
axes[0].set_title("1st Differencing")
plot_acf(df["1st_diff"], lags=200, use_vlines=True, ax=axes[1])
plt.show()

# Build ARIMA model
model = ARIMA(df.close, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

# Note: ar p-val is >= 0.05 and therefore, not significant. Remove the ar component
model = ARIMA(df.close, order=(0, 1, 1))
model_fit = model.fit()
print(model_fit.summary())
# Note: the AIC decreases which is good! Better model.

# Now, plot the residuals to ensure there are no patterns
# (that is, look for constant mean and variance)
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1, 2, figsize=(10, 7))
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind="kde", title="Density", ax=ax[1])
plt.show()

# Out-sample-prediction #

# Create Train and Test Set
train = df.close[:int(0.8 * len(df))]
test = df.close[int(0.8 * len(df)):]
print(f"\ntrain set length: {len(train)}, test set length: {len(test)}")

# Build Model
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()

# Forecast
fc = model_fit.forecast(len(test), alpha=0.05)  # 95% conf

# Make a pd.Series()
fc_series = pd.Series(fc, index=test.index)
# lower_series = pd.Series(conf[:, 0], index=test.index)
# upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.rcParams.update({"figure.figsize": (12, 5), "figure.dpi": 100})
plt.figure()
plt.plot(train, label="Training")
plt.plot(test, label="Actual")
plt.plot(fc_series, label="Forecast")
# plt.fill_between(lower_series.index, lower_series, upper_series, color="k", alpha=.15)
plt.title("Forecast vs Actuals")
plt.legend(loc="best", fontsize=8)
plt.show()

model = ARIMA(train, order=(1, 2, 1))
model_fit = model.fit()
print(model_fit.summary())

fc = model_fit.forecast(len(test), alpha=0.05)  # 95% conf

fc_series = pd.Series(fc, index=test.index)
# lower_series = pd.Series(conf[:, 0], index=test.index)
# upper_series = pd.Series(conf[:, 1], index=test.index)

plt.figure(figsize=(12, 5), dpi=100)
plt.plot(train, label="Training")
plt.plot(test, label="Actuals")
plt.plot(fc_series, label="Forecast")
# plt.fill_between(lower_series.index, lower_series, upper_series.index, upper_series, color="k", alpha=.15)
plt.title("Forecast vs Actuals")
plt.legend(loc="best", fontsize=8)
plt.show()


# Accuracy Metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)  # ME
    mae = np.mean(np.abs(forecast - actual))  # MAE
    mpe = np.mean((forecast - actual) / actual)  # MPE
    rmse = np.mean((forecast - actual) ** 2) ** .5  # RMSE
    corr = np.corrcoef(forecast, actual)[0, 1]  # corr
    mins = np.amin(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    minmax = 1 - np.mean(mins / maxs)  # minmax
    acf1 = acf(fc - test)[1]  # ACF1
    return ({'mape': mape, 'me': me, 'mae': mae,
             'mpe': mpe, 'rmse': rmse, 'acf1': acf1,
             'corr': corr, 'minmax': minmax})


for i, val in forecast_accuracy(fc, test.values).items():
    print(i, val)

# AutoARIMA
model = pm.auto_arima(df.close, start_p=1, start_q=1,
                      test="adf",
                      max_p=5, max_q=5,
                      m=1,
                      d=None,
                      seasonal=False,
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action="ignore",
                      suppress_warnings=True,
                      stepwise=True)
print(model.summary())


# Interpreting Residual Plots
# use the stepwise_fit
model.plot_diagnostics(figsize=(7, 5))
plt.show()

# Seems to be a Good Fit
# Forecast
n_periods = 10
fc, confint = model.predict(,,,,
index_of_fc = np.arange(len(df.close), len(df.close) + n_periods)

fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

plt.plot(df.close)
plt.plot(fc_series, color="darkgreen")
plt.fill_between(lower_series.index, lower_series,
                 upper_series, color="k", alpha=.15)
plt.title("Final Forecast of Nike Monthly Close Price")
plt.show()

# SARIMA Model
fig, axes = plt.subplots(2, 1, figsize=(10, 5), dpi=100, sharex=True)
axes[0].plot(df.close, label="NIKE Original Monthly Series")
axes[0].plot(df.close.diff(1), label="Usual Differencing")
axes[0].set_title("Usual Differencing")
axes[0].legend(loc="best", fontsize=10)

axes[1].plot(df.close, label="Original Series")
axes[1].plot(df.close.diff(12), label="Seasonal Differencing", color="green")
axes[1].set_title("Sesaonal Differencing")
plt.legend(loc="best", fontsize=10)
plt.suptitle("Nike Monthly Adj Close Price")
plt.show()

smodel = pm.auto_arima(df.close, start_p=1, start_q=1, test="adf", max_p=5, max_q=5, m=12,
                       start_P=0, seasonal=True, d=None, D=1, trace=True, error_action="ignore",
                       suppress_warnings=True, stepwise=True)
print(smodel.summary())

n_periods = 24
model_fit, confint = smodel.predict(,,,,
index_of_fc = np.arange(len(df.close), len(df.close) + n_periods)

fc_series = pd.Series(model_fit, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

plt.plot(df.close)
plt.plot(fc_series, color="darkgreen")
plt.fill_between(lower_series.index, lower_series, upper_series, color="k", alpha=.15)
plt.title("SARIMA - Final Forecast of Nike Monthly Adj Close Price (24 months ahead)")
plt.show()

# SARIMAX Model
# compute the seasonal index so that it can be forced as a (exogenous) predictor to the SARIMAX model

# multiplicative seasonal component
df.set_index("date", inplace=True)
df.drop(columns=["1st_diff", "2nd_diff"], inplace=True)
data = df[-36:]

result_mul = seasonal_decompose(data,  # 3 years
                                model="multiplicative",
                                extrapolate_trend="freq")

seasonal_index = result_mul.seasonal[-12:].to_frame()
seasonal_index["month"] = pd.to_datetime(seasonal_index.index).month

# merge with the base data
data["month"] = data.index.month
df = pd.merge(data, seasonal_index, how="left", on="month")
df.columns = ["close", "month", "seasonal_index"]
df.index = data.index  # reassign the index

print(df)

# SARIMAX Model
sxmodel = pm.auto_arima(df[["close"]], exogenous=df[["seasonal_index"]],
                        start_p=1, start_q=1, test="adf", max_p=3, max_q=3, m=12,
                        start_P=0, seasonal=True, d=None, D=1, trace=True,
                        error_action="ignore", suppress_warnings=True, stepwise=True)
print(sxmodel.summary())

# Forecast
n_periods = 24
model_fitted, confint = sxmodel.predict(,,,,

index_of_fc = pd.date_range(data.index[-1], periods=n_periods, freq="MS")

fitted_series = pd.Series(model_fitted.values, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

plt.plot(data.close)
plt.plot(fitted_series, color="darkgreen")
plt.fill_between(lower_series.index, lower_series,
                 upper_series, color="k", alpha=.15)
plt.title("SARIMAX Forecast of Nike Monthly Adj Close Price Series")
plt.show()
