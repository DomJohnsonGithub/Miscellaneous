import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import Data
raw = pd.read_csv('aiif_eikon_eod_data.csv', index_col=0, parse_dates=True, delimiter=",").dropna()

symbol = ".SPX"
data = pd.DataFrame(raw[symbol]).rename(columns={symbol: "price"})

# Price Prediction
lags = 5
for lag in range(1, lags + 1):
    data[f"lag_{lag}"] = data["price"].shift(lag)

data.dropna(inplace=True)
print(data)

# Regression Coefficients
reg = np.linalg.lstsq(data.iloc[:, 1:], data.loc[:, "price"], rcond=None)[0]
print(reg)

data["prediction"] = np.dot(data.iloc[:, 1:], reg)

# Visualize price prediction, pred for tomorrow's rate is roughly
# today's rate. More or less a shift to right by one trading day
data[["price", "prediction"]].loc["2019-10-1":].plot(figsize=(10, 6))
plt.show()


# Predicting future returns
data["return"] = np.log(data.price / data.price.shift(1))
lags = 5
for i in range(1, lags + 1):
    data[f"return_lag{i}"] = data["return"].shift(1)
data.dropna(inplace=True)
print(data.iloc[:, 7:])

regression = np.linalg.lstsq(data.iloc[:, 8:], data.iloc[:, 7], rcond=None)[0]
print(regression)

data["PRED"] = np.dot(data.iloc[:, 8:], regression)

print(data)
data[["return", "PRED"]].iloc[lags:].plot(figsize=(10, 6))
plt.show()

# Instead of using magnitude, but direction instead
hits = np.sign(data["return"] * data["PRED"]).value_counts()
print(hits)

print(hits.values[0] / sum(hits))  # hit ratio

# Predicting future market direction
# improve hit ratio by using sign instead of actual log returns.
df = data[["price", "return"]].copy()

lags = 5
for i in range(1, lags + 1):
    df[f"return_lag{i}"] = df["return"].shift(i)
df.dropna(inplace=True)
print(df)

regression = np.linalg.lstsq(df.iloc[:, 2:], np.sign(df["return"]), rcond=None)[0]

df["prediction"] = np.sign(np.dot(df.iloc[:, 2:], regression))
print(df["prediction"].value_counts())  # for the prediction step only sign is relevant

print(df)

hits = np.sign(df["return"] * df["prediction"]).value_counts()
print(hits)
print(hits.values[0] / sum(hits))  # increased the hit ratio

df['strategy'] = df['prediction'] * df['return']
df[['return', 'strategy']].sum().apply(np.exp)

df[['return', 'strategy']].dropna().cumsum(
).apply(np.exp).plot(figsize=(10, 6))
plt.show()
