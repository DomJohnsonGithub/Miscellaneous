import numpy as np
from pathlib import Path
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import pmdarima as pm
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
data = df.close.copy()


# -----------------------------------
fod = df.copy().resample("M").mean()
fod["date"] = fod.index
fod["month"] = fod.index.month_name()
fod["year"] = fod.index.year
fod.drop(columns=["volume", "date"], inplace=True)
fod = fod.reset_index()
print(fod)

fod_pivot = fod.pivot("month", "year", "close")

import seaborn as sns

sns.heatmap(fod_pivot, annot=True, cmap="winter", linewidths=0.2, linecolor="black", cbar=True)
plt.show()
# ------------------------------------

# ----- Create Train and Test Sets ----- #
size = 0.8
train = data[:int(len(data) * size)]
test = data[int(len(data) * size):]

fig = plt.figure()
train.plot(label="Train")
test.plot(label="Test")
plt.legend(loc="best")
plt.show()

# ----- Build Model ----- #
model = pm.auto_arima(train, start_p=1, start_q=1, max_p=4, max_q=4, seasonal=False, test="adf", d=1, D=1,
                      trace=True, error_action="ignore", suppress_warnings=True, stepwise=True)

print(model.aic())
print(model.summary())
model.plot_diagnostics(figsize=(10, 6))

# ----- Fit the Model & Forecast ----- #
model.fit(train)
forecast = model.predict(,,,,

forecast = pd.DataFrame(forecast.values, index=test.index, columns=["Prediction"])

fig = plt.figure()
plt.plot(train, label="Train", color="green")
plt.plot(test, label="Test", color="blue")
plt.plot(forecast, label="Predicted", color="red")
plt.legend(loc="best")
plt.grid(True)
plt.show()


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)  # ME
    mae = np.mean(np.abs(forecast - actual))  # MAE
    mpe = np.mean((forecast - actual) / actual)  # MPE
    rmse = np.mean((forecast - actual) ** 2) ** .5  # RMSE
    return ({'mape': mape, 'me': me, 'mae': mae,
             'mpe': mpe, 'rmse': rmse})


for i, val in forecast_accuracy(forecast.values, test.values).items():
    print(i, val)

# ----- Seasonal ARIMA ----- #
data = data.resample("M").mean()

size = 0.8
train = data[:int(len(data) * size)]
test = data[int(len(data) * size):]

model = pm.auto_arima(train, start_p=1, start_q=1, max_p=4, max_q=4, max_d=2, seasonal=True, test="adf", d=None, D=1,
                      m=12,
                      max_P=4, max_Q=4, trace=True, error_action="ignore", suppress_warnings=True, stepwise=True)

print(model.aic())
print(model.summary())
model.plot_diagnostics(figsize=(10, 6))

model.fit(train)
forecast = model.predict(,,,,
forecast = pd.DataFrame(forecast.values, index=test.index, columns=["Prediction"])

fig = plt.figure()
plt.plot(train, label="Train", color="green")
plt.plot(test, label="Test", color="blue")
plt.plot(forecast, label="Predicted", color="red")
plt.legend(loc="best")
plt.grid(True)
plt.show()

for i, val in forecast_accuracy(forecast.values, test.values).items():
    print(i, val)

import scipy.stats as ss

error = test.values - forecast.Prediction

error = pd.Series(error, index=test.index)
plt.figure(figsize=(20, 10), dpi=100)
plt.subplot(121)
plt.plot(error, color="#ff33CC")
plt.title("Error Distribution Over Time")
plt.subplot(122)
ss.probplot(error, plot=plt)
plt.show()

plt.figure(figsize=(20, 10))
pm.autocorr_plot(error)
plt.show()

# ---------------------------------

import yfinance as yf
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

hd = yf.download("HD", start=datetime(2019, 1, 1), end=datetime(2019, 4, 1))
hd = hd["Adj Close"].resample("M").mean()

# Now fit model to entire dataset and make predictions of the future
model.fit(data)

future_forecast_1 = model.predict(,,,,
future_forecast_2 = model.predict(,,,,
future_forecast_3 = model.predict(,,,,

month_forecasts = {
    "1 month": future_forecast_1,
    "2 month": future_forecast_2,
    "3 month": future_forecast_3
}

for i, val in month_forecasts.items():
    print(i, val[-1])

print(hd[:-1])
# ----------------------------------
