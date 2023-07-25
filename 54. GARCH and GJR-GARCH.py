import numpy as np
from pathlib import Path
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from arch import arch_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ----- Import Data ----- #
DATA_STORE = Path("C:\\Users\\domin\\PycharmProjects\\Miscellaneous\\dow_stocks.h5")

with pd.HDFStore(DATA_STORE, "r") as store:
    df = store.get("DOW/stocks")

idx = pd.IndexSlice
df = df.loc[idx[["WBA"], :], ["close", "volume"]].droplevel(0)
print(df)


# ----- Train Test Split ----- #
def train_test_split(X, size=0.9):
    train = X[:int(X.shape[0] * size)]
    test = X[int(X.shape[0] * size):]
    return train, test


train, test = train_test_split(X=df["close"], size=0.9)
print(f"Train Set: {len(train)}, Test Set: {len(test)}")

# viz the train and test datasets
fig, ax = plt.subplots(figsize=(10, 4))
train.plot(ax=ax, label="train")
test.plot(ax=ax, label="test")
plt.title("Walgreens Boots Alliance")
ax.legend(loc="best")
plt.show()


# ----- Calculate Volatility ----- #
def returns_vol(df, column):
    # calc. returns as a percentage of price changes
    df["Returns"] = df[column].pct_change() * 100
    # calc. standard deviation of retruns
    volatility = df.Returns.std()
    df.dropna(inplace=True)
    return df, volatility


df, volatility = returns_vol(df, column="close")

# ----- Plot the Price Returns and Print Volatility ----- #
plt.figure(figsize=(10, 4))
plt.plot(df["Returns"], label="real data")
plt.title("Walgreens Daily Returns")
plt.legend(loc="best")
plt.show()

print(f"""Daily Volatility: {volatility:.2f}%
Month Volatility: {np.sqrt(21) * volatility:.2f}%
Annual Volatility: {np.sqrt(252) * volatility:.2f}%""")


# ----- Specify GARCH Model Function ----- #
def garch_model(df, p=1, o=0, q=1, mean="constant", vol="GARCH",
                dist="normal"):
    model = arch_model(
        df,
        p=p, o=o, q=q,
        mean=mean,
        vol=vol,
        dist=dist
    )
    return model


df.dropna(inplace=True)
print(df)

# ----- Implement a Basic GARCH Model ----- #
basic_gm = garch_model(df["Returns"])
# Fit the model
gm_result = basic_gm.fit(disp="off", show_warning=False)

# Display model fitting summary
print(gm_result.summary())

# ----- Plot Fitted Results ----- #
plt.rc("figure", figsize=(10, 7))
gm_result.plot()
plt.show()

# ----- Obtain Model Estimated Residuals and Volatiliy ----- #
# Residual = Predicted price - Actual price
gm_resid = gm_result.resid
# Predicted price volatility (std)
gm_std = gm_result.conditional_volatility

# Calc. the standardized residuals
gm_std_resid = gm_resid / gm_std

# ----- Diagnose Plot of Residuals ----- #
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), dpi=100)
ax1.hist(gm_std_resid, bins=50)
ax1.set_title("Standardized Residuals")
qqplot(gm_std_resid, line="s", ax=ax2)
ax2.set_title("Normal Q_Q Plot")
plot_acf(gm_std_resid, ax=ax3)
plot_pacf(gm_std_resid, ax=ax4)
plt.show()

# ----- Specify GJR-GARCH Model Assumption ----- #
gjr_gm = garch_model(df["Returns"], o=1, dist="skewt")

# fit the model
gjr_result = gjr_gm.fit(disp="off", show_warning=False)

# get the model volatility
gjr_vol = gjr_result.conditional_volatility

# ----- Plot the Model Fitting Results ----- #
plt.rc("figure", figsize=(10, 4))
gm_vol = gm_result.conditional_volatility
plt.plot(df['Returns'], color='grey',
         label='Daily Returns', alpha=0.4)
plt.plot(gm_vol, label='GARCH Volatility')
plt.plot(gjr_vol, label='Skewed-t GJR-GARCH Volatility')
plt.legend(loc='upper left')
plt.show()

print(f'Correlation coef between GARCH & skewed-t GJR=GARCH: {round(np.corrcoef(gm_vol, gjr_vol)[0, 1], 2)}')

# ----- In-sample Rolling Window One-step-forecast ----- #
index = df.index
end_loc = 577
forecasts = {}
for i in range(50):
    # specify fixed rolling window size for model fitting
    gm_result = basic_gm.fit(last_obs=i + end_loc, disp="off",
                             show_warning=False)
    # conduct 1-period variance forecast and save the result
    temp_result = gm_result.forecast(horizon=1).variance
    fcast = temp_result.iloc[i + end_loc]
    forecasts[fcast.name] = fcast

# Save all forecasts to a dataframe
forecast_var = pd.DataFrame(forecasts).T
forecast_var.index = df.index[-50:]

# Plot in-sample forcast and real volatility
plt.plot(df["Returns"][-100:], color="grey",
         label="Daily Returns", alpha=0.4)
plt.plot(forecast_var, label="GARCH Volatility")
plt.legend(loc="best")
plt.show()

# ----- Now, Forecast on Test Data ----- #
test = pd.DataFrame(test)
train = pd.DataFrame(train)
train, _ = returns_vol(train, column="close")
test, _ = returns_vol(test, column="close")

# Plot train and test returns
fig, ax = plt.subplots(figsize=(10, 4))
train["Returns"].plot(ax=ax, label="train")
test["Returns"].plot(ax=ax, label="test")
plt.title("Walgreens Returns")
ax.legend(loc="best")
plt.show()

# ----- Out-of-sample Rolling Window One-Step Forecast ----- #
rolling_predictions = []
for i in range(test.shape[0]):
    model = garch_model(train["Returns"], p=1, o=1, q=1)
    model_fit = model.fit(update_freq=1, disp="off", show_warning=False)
    pred = model_fit.forecast(horizon=1)
    train = train.append(test.iloc[i])
    rolling_predictions.append(np.sqrt(pred.variance.values[-1, :][0]))

# plot out-of-sample forecast and real volatility
rolling_predictions = pd.DataFrame(rolling_predictions, index=test.index)
plt.plot(train["Returns"], label="Daily Returns", alpha=.4)
plt.plot(test["Returns"], label="Daily Returns", alpha=.4)
plt.plot(rolling_predictions, label="GARCH Volatility")
plt.legend(loc="best")
plt.show()

# plot only test data with forecasted returns
plt.plot(test["Returns"], color="C1", label="Daily Returns", alpha=.4)
plt.plot(rolling_predictions, color="C2", label="GARCH Volatility")
plt.legend(loc="best")
plt.show()
