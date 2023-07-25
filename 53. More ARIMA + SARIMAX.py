import numpy as np
from pathlib import Path
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ----- Import Data ----- #
DATA_STORE = Path("C:\\Users\\domin\\PycharmProjects\\Miscellaneous\\dow_stocks.h5")

with pd.HDFStore(DATA_STORE, "r") as store:
    df = store.get("DOW/stocks")

idx = pd.IndexSlice
df = df.loc[idx[["AXP", "KO", "MCD", "WBA"], :], "close"]

amx = df.loc[idx["AXP", :]]
cola = df.loc[idx["KO", :]]
mcd = df.loc[idx["MCD", :]]
wlg = df.loc[idx["WBA", :]]

df = pd.concat([amx, cola, mcd, wlg], axis=1, ignore_index=False)
df.columns = ["American Express", "Coca Cola", "Mc.Donalds", "Walgreens"]
print(df)

# ----- Viz the Daily Adj Close Prices ----- #
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 7), dpi=110, sharex=True)
axes[0].plot(df["American Express"], label="AXP", color="k")
axes[0].legend(loc="best")
axes[0].set_title("American Express")

axes[1].plot(df["Coca Cola"], label="KO", color="g")
axes[1].legend(loc="best")
axes[1].set_title("Coca Cola")

axes[2].plot(df["Mc.Donalds"], label="MCD", color="r")
axes[2].legend(loc="best")
axes[2].set_title("Mc.Donalds")

axes[3].plot(df["Walgreens"], label="WBA", color="y")
axes[3].legend(loc="best")
axes[3].set_title("Walgreen Boots Alliance")
plt.show()

# ----- Histogram/ Density Viz of the Price Series' ----- #
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8, 5), dpi=110)

df["American Express"].hist(ax=axes[0, 0], bins=20, label="AXP", color="k")
axes[0, 0].legend(loc="best")
axes[0, 0].set_title("American Express")
df["American Express"].plot(ax=axes[1, 0], kind="kde", label="AXP", color="k")

df["Coca Cola"].hist(ax=axes[0, 1], bins=20, label="KO", color="g")
axes[0, 1].legend(loc="best")
axes[0, 1].set_title("Coca Cola")
df["Coca Cola"].plot(ax=axes[1, 1], kind="kde", label="KO", color="g")

df["Mc.Donalds"].hist(ax=axes[2, 0], bins=20, label="MCD", color="y")
axes[2, 0].legend(loc="best")
axes[2, 0].set_title("Mc.Donalds")
df["Mc.Donalds"].plot(ax=axes[3, 0], kind="kde", label="MCD", color="y")

df["Walgreens"].hist(ax=axes[2, 1], label="WBA", color="b")
axes[2, 1].legend(loc="best")
axes[2, 1].set_title("Walgreen Boots Alliance")
df["Walgreens"].plot(ax=axes[3, 1], kind="kde", label="WBA", color="b")

plt.show()


# ----- ACF and PACF Plot of Time Series ----- #
def plot_acf_pacf(df):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 6), dpi=100)
    plot_acf(df, lags=50, zero=False, ax=axes[0], title=f"Autocorrelation")
    plot_pacf(df, lags=50, zero=False, ax=axes[1], title=f" Partial Autocorrelation")
    return plt.show()


plot_acf_pacf(df["American Express"])
# plot_acf_pacf(df["Coca Cola"])
# plot_acf_pacf(df["Mc.Donalds"])
# plot_acf_pacf(df["Walgreens"])

# ----- Transform data from non-stationary to stationary ----- #
# will perform on the Walgreen Price Series
# Method 1: difference data
X = df.copy()
X = X["Walgreens"]
stationary = X.diff(1)

log_stationary = np.log(X).diff()

stationary.dropna(inplace=True);
log_stationary.dropna(inplace=True)

# Method 2: take the log
# stationary = np.log(X)

# Method 3: take the square root
# stationary = np.sqrt(X)

# Method 4: proportional change
# stationary = X.pct_change(1)

# ----- Augmented Dickey-Fuller Test for Stationarity Check ----- #
result_01 = adfuller(stationary)
result_02 = adfuller(log_stationary)

# test statistic - more neg. means more likely to be stationary
print("Differenced Series, ADF Statistic: %f" % result_01[0])
print("Log Differenced Series, ADF Statistic: %f" % result_02[0])

# p-value - reject null-hypothesis: non-stationary
print("Differenced Series, p-value: %f" % result_01[1])
print("Log Differenced Series, p-value: %f" % result_02[1])

# critical test statistics - p-values: test statistic for null hypothesis
print("Critical Values:")
for key, value, in result_01[4].items():
    print("\t%s: %.3f" % (key, value))

# plot the stationary datasets
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 9), dpi=120)

stationary.plot(ax=axes[0, 0])
axes[0, 0].set_title("Walgreens Differenced Stationary Series")
plot_acf(stationary, lags=50, zero=False, ax=axes[1, 0], title=f"Autocorrelation")
plot_pacf(stationary, lags=50, zero=False, ax=axes[2, 0], title=f" Partial Autocorrelation")

log_stationary.plot(ax=axes[0, 1])
axes[0, 1].set_title("Walgreens Log Differenced Stationary Series")
plot_acf(stationary, lags=50, zero=False, ax=axes[1, 1], title=f"Autocorrelation")
plot_pacf(stationary, lags=50, zero=False, ax=axes[2, 1], title=f" Partial Autocorrelation")
plt.show()


# ----- Searching over SARIMA model orders ----- #
class auto_arima():
    def __init__(self, df, start_p=1, start_q=1, max_p=10, max_q=10, \
                 seasonal=False, information_criterion="aic"):
        self.df = df
        self.start_p = start_p
        self.start_q = start_q
        self.max_p = max_p
        self.max_q = max_q
        self.seasonal = seasonal
        self.information_criterion = information_criterion

    def arima_results(self):
        results = pm.auto_arima(
            self.df,
            start_p=self.start_p,
            start_q=self.start_q,
            max_p=self.max_p,
            max_q=self.max_q,
            seasonal=self.seasonal,
            # m = 14,
            # D = 1,
            # start_P = 1,
            # start_Q = 1,
            # max_P = 10,
            # max_Q = 10,
            information_criterion=self.information_criterion,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
            scoring="mse"

        )
        return results


# ----- Train and Test Split ----- #
def train_test_split(X, size=0.9):
    train = X[:int(X.shape[0] * size)]
    test = X[int(X.shape[0] * size):]
    return train, test


train, test = train_test_split(X=df["Walgreens"])
print(f"Train: {len(train)}, Test: {len(test)}")

# plot the train and test datasets
fig, ax = plt.subplots(figsize=(10, 4))
train.plot(ax=ax, label="train")
test.plot(ax=ax, label="test")
ax.legend(loc="best")
ax.set_title("Train and Test Sets")
ax.grid(True)
plt.show()

# ----- Fit the Model with Auto-Arima ----- #
arima_model = auto_arima(train)
results = arima_model.arima_results()

# Check residuals
# Prob(Q) - p-value for null hypothesis that residuals are uncorrelated
# Prob(JB) - p-value for null hypothesis that residuals are normal
print(results.summary())

# Residuals are not correlated and are not normally distributed

# Plot diagnostics - check residuals:
# 1.Standardized residual - should be white noise
# 2.Histogram plus estimated density - expected normal distribution
# and kde overlap each other
# 3.Normal Q-Q - all points should lay on red line, except perhaps
# for some values at both ends
# 4.Correlogram - acf plot, 95% should not be significant
plot_diag = results.plot_diagnostics(figsize=(10, 7))
plt.show()

# ----- Out-of-sample Multi-Step Forecast Based on the auto_arima Results ----- #
predicted = results.predict(,,,,
predicted = pd.DataFrame(predicted.values, index=test.index, columns=["predicted"])


def plot_train_test(train, test, df):
    # plot the predictions for validation set
    plt.figure(figsize=(10, 4))
    plt.plot(train, label='train')
    plt.plot(test, label='test')
    plt.plot(df, label='predicted')
    plt.legend()
    plt.show()

    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(test, df))
    print(f'RMSE: {rmse:.2f}')


plot_train_test(train, test, predicted)


# ----- ARIMA Multi-Step Forecast is pretty bad, so let's compare it with one-step-forecast ------ #
def one_step_forecast():
    predicted, conf_int = results.predict(,,,,
    return (
        predicted.tolist()[0],
        np.asarray(conf_int).tolist()[0]
    )


predictions = []
confidence_intervals = []

for x in test:
    predicted, conf = one_step_forecast()
    predictions.append(predicted)
    confidence_intervals.append(conf)

    # updates the existing model
    results.update(x)

# ----- Out-of-sample One-step-forecast Based on auto_arima Results ----- #
predicted = pd.DataFrame(predictions, index=test.index, columns=["predictions"])

# plot the real price vs one-step-forecast
plot_train_test(train, test, predicted)

# calculate RMSE
rmse = np.sqrt(mean_squared_error(test, predictions))
print("RMSE: %.4f" % rmse)

# Forecast with Confidence Intervals
lower_limits = [row[0] for row in confidence_intervals]
upper_limits = [row[1] for row in confidence_intervals]

# plot predictions with conf_int
plt.figure(figsize=(10, 4))
plt.plot(train, label="train")
plt.plot(test, label="test")
plt.plot(predicted, label="predicted")
plt.fill_between(test.index, lower_limits, upper_limits,
                 color="pink", label="confidence intervals")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()


# ----- Making Predictions Another Way ----- #
# SARIMAX
# one-step-ahead in-sample predictions with uncertainty
def sarimax_model(df, trend="ct", steps=100, dynamic=False):
    model = SARIMAX(df, order=(7, 1, 7), trend=trend)
    results = model.fit()
    one_step_forecast = results.get_prediction(start=-steps, dynamic=dynamic)
    # Get in-sample predicted mean values
    predictions = one_step_forecast.predicted_mean
    # Get confidence intervals of in-sample forecasts
    confidence_intervals = one_step_forecast.conf_int()
    lower_limits = confidence_intervals["lower Walgreens"]
    upper_limits = confidence_intervals["upper Walgreens"]
    return predictions, lower_limits, upper_limits


# Run Model
predictions, lower_limits, upper_limits = \
    sarimax_model(df=test, trend="ct", steps=100, dynamic=False)


# Plot real data
def plot_sarimax_pred(df, steps=100):
    plt.figure(figsize=(10, 4))
    plt.plot(df.index[-steps:], df[-steps:],
             color="b", label="real data")
    # plot predictions
    plt.plot(predictions.index, predictions,
             color="g", label="predicted")
    # plot CI's
    plt.fill_between(lower_limits.index, lower_limits, upper_limits,
                     color="pink", label="confidence intervals")
    plt.legend(loc="best")
    plt.show()


plot_sarimax_pred(df=test, steps=100)

# Dynamic Forecast for the next 50 days
# Run model
predictions, lower_limits, upper_limits = \
    sarimax_model(df=df["Walgreens"], trend="ct", steps=50, dynamic=True)
# plot dynamic forecast
plot_sarimax_pred(df=df["Walgreens"], steps=400)

# One-step-forecast for next 50 days
predictions, lower_limits, upper_limits = \
    sarimax_model(df=df["Walgreens"], trend="ct", steps=50, dynamic=False)
# plot one-step-forecast for comparison with multi-step-forecast
plot_sarimax_pred(df=df["Walgreens"], steps=400)

# ----- Compare Predictions with Real Data ----- #
forecast_vs_real = pd.concat([round(df["Walgreens"][-50:], 2), round(predictions, 2)], axis=1)
forecast_vs_real.columns = ["Real Price", "Forecast"]
forecast_vs_real["Error_%"] = round(abs(forecast_vs_real["Forecast"] - forecast_vs_real["Real Price"]) / \
                                    forecast_vs_real["Real Price"] * 100, 1)
print(forecast_vs_real)
print(f"Mean error: {round(forecast_vs_real['Error_%'].mean(), 1)} %")
