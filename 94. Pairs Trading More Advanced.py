import numpy as np
import pandas as pd
from datetime import datetime
import pandas_datareader.data as pdr
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from tqdm import tqdm
from itertools import accumulate
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.isotonic import IsotonicRegression
from polyfit import PolynomRegressor, Constraints
from scipy.stats import mode
import yfinance as yf

sns.set_style("darkgrid")

stocks = ["DHR", "TMO"]
start, end = datetime(2000, 1, 1), datetime.now()
df = yf.download(stocks, start, end)["Close"]
df.dropna(inplace=True)

window = 40  # 12
exog = sm.add_constant(df.DHR)
rolling_ols = RollingOLS(endog=df.TMO, exog=exog, window=window, min_nobs=12).fit()
beta = - np.array(rolling_ols.params.iloc[:, 1].dropna())  # rolling beta
df = df.iloc[window - 1:, :]

df["Spread"] = np.log(df.TMO) + beta * np.log(df.DHR)


# -------------------------------


class TradingHorizon:
    """
    Indicative of trading period. A pair may be deemed tradable
    if we are satisfied with the range of trade periods or zero-
    crossing frequencies generated by the bootstrap. It's the
    degree of mean reversion.
    """

    @staticmethod
    def time_between_zero_crossings(spread):
        threshold = np.mean(spread)  # equilibrium (log-run mean)
        groups = accumulate([0] + [(a >= threshold) != (b >= threshold) \
                                   for a, b in zip(spread.values, spread.values[1:])])
        counts = sorted(Counter(groups).items())
        above = [c for n, c in counts if (n % 2 == 0) == (spread.values[0] >= threshold)]
        below = [c for n, c in counts if (n % 2 == 0) != (spread.values[0] >= threshold)]
        return np.array(above + below)

    @staticmethod
    def bootstrap_zero_crossing_rate(spread, simulations):
        median = []
        for i in range(simulations):
            time_between_crossings = TradingHorizon.time_between_zero_crossings(spread)
            n = int(np.random.randint(0, len(spread) - np.max(time_between_crossings), size=1))
            bootstrap_sample = spread[n:n + np.max(time_between_crossings) + 300]
            median.append(np.median(TradingHorizon.time_between_zero_crossings(bootstrap_sample)))

        return int(np.mean(median, 0))


trading_horizon = TradingHorizon()
print(trading_horizon.bootstrap_zero_crossing_rate(df.Spread, simulations=1000))

fig, axes = plt.subplots(nrows=2, ncols=1, dpi=50)
axes[0].plot(df.Spread)
axes[1].plot((df.Spread - df.Spread.rolling(100).mean()) / df.Spread.rolling(100).std())
plt.show()

spread = np.array(df.Spread[-75:])
print(len(spread))

equilibrium, standard_deviation = np.mean(spread), np.std(spread)
range_of_sigma_around_mean = np.round(np.arange(0, 3.1, 0.1), 1)
count_exceeding_upper_threshold, count_exceeding_lower_threshold = [], []
up_sigma, down_sigma = [], []
for i in range_of_sigma_around_mean:
    upper_theoretical_threshold = equilibrium + i * standard_deviation
    lower_theoretical_threshold = equilibrium - i * standard_deviation
    for j in range(75):

        if spread[j - 1] < upper_theoretical_threshold <= spread[j]:
            count_exceeding_upper_threshold.append(j)
            up_sigma.append(i)

        if spread[j - 1] > lower_theoretical_threshold >= spread[j]:
            count_exceeding_lower_threshold.append(j)
            down_sigma.append(i)

up_data = pd.DataFrame(count_exceeding_upper_threshold[1:], index=up_sigma[1:])
down_data = pd.DataFrame(count_exceeding_lower_threshold[1:], index=down_sigma[1:])
probability = [(len(up_data[up_data.index == i]) + len(down_data[down_data.index == i])) / 75 \
               for i in range_of_sigma_around_mean]

up_prob = [len(up_data[up_data.index == i]) / 75 for i in range_of_sigma_around_mean]
down_prob = [len(down_data[down_data.index == i]) / 75 for i in range_of_sigma_around_mean]
monotonic_adj_up_prob = np.minimum.accumulate(up_prob)
monotonic_adj_down_prob = np.minimum.accumulate(down_prob)

deg = 5
polyestimator = PolynomRegressor(deg=deg, regularization="l2")
polyestimator.fit(x=np.reshape(range_of_sigma_around_mean, (-1, 1)),
                  y=monotonic_adj_up_prob, loss="l2",
                  constraints={0: Constraints(monotonicity="dec")})
upper_threshold_from_profit = range_of_sigma_around_mean[np.argmax(range_of_sigma_around_mean * \
                                                                   polyestimator.predict(np.reshape(
                                                                       range_of_sigma_around_mean, (-1, 1))))]

polyestimator.fit(x=np.reshape(range_of_sigma_around_mean, (-1, 1)),
                  y=monotonic_adj_down_prob, loss="l2",
                  constraints={0: Constraints(monotonicity="dec")})
lower_threshold_from_profit = range_of_sigma_around_mean[np.argmax(range_of_sigma_around_mean * \
                                                                   polyestimator.predict(np.reshape(
                                                                       range_of_sigma_around_mean, (-1, 1))))]

# monotonic adjustment of the probability curve
monotonic_adjusted_probability = np.minimum.accumulate(probability)

# fit a polynomial with regularization (Tikhonov-Miller)
deg = 5
polyestimator = PolynomRegressor(deg=deg, regularization="l2")
polyestimator.fit(x=np.reshape(range_of_sigma_around_mean, (-1, 1)),
                  y=monotonic_adjusted_probability, loss="l2",
                  constraints={0: Constraints(monotonicity="dec")})
regularized_curve = polyestimator.predict(np.reshape(range_of_sigma_around_mean, (-1, 1)))

plt.plot(range_of_sigma_around_mean, probability, c="blue")
plt.plot(range_of_sigma_around_mean, monotonic_adjusted_probability, c="orange")
plt.plot(range_of_sigma_around_mean, regularized_curve, c="red")
plt.show()

plt.plot(range_of_sigma_around_mean, range_of_sigma_around_mean * probability, c="blue")
plt.plot(range_of_sigma_around_mean, range_of_sigma_around_mean * monotonic_adjusted_probability, c="orange")
plt.plot(range_of_sigma_around_mean, range_of_sigma_around_mean * regularized_curve, c="red")
plt.show()

best_threshold_from_profit = range_of_sigma_around_mean[np.argmax(range_of_sigma_around_mean * regularized_curve)]

print(best_threshold_from_profit[0][0])
print(upper_threshold_from_profit[0][0])
print(lower_threshold_from_profit[0][0])

# ------------------------------------------------

# portfolio_ending_value = []
# sharpe = []
# sigmas = np.arange(0, 3.1, 0.1)
# cumulative_rets = []
# for sigma in sigmas:
#     equilibrium = df["Spread"].mean()
#     # std = np.sqrt(np.exp(model.smoothed_state[0] / 2))
#     # std = garch
#     # std = df["Spread"].std()
#     # std = vol
#     upper_threshold = equilibrium + sigma * std
#     lower_threshold = equilibrium - sigma * std
#
#     # plt.plot(df.Spread)
#     # plt.plot(df.index, upper_threshold)
#     # plt.plot(df.index, lower_threshold)
#     # plt.show()
#
#     long_entry = (df["Spread"] <= lower_threshold) & (df["Spread"].shift(1) > lower_threshold)
#     long_exit = (df["Spread"] >= equilibrium) & (df["Spread"].shift(1) < equilibrium)
#     units_long = np.full(shape=len(df), fill_value=np.nan)
#     units_long[np.where(long_entry == True)] = 1
#     units_long[np.where(long_exit == True)] = 0
#     units_long[:window - 1] = 0
#     units_long = pd.Series(units_long).fillna(method="pad")
#
#     short_entry = (df["Spread"] >= upper_threshold) & (df["Spread"].shift(1) < lower_threshold)
#     short_exit = (df["Spread"] <= equilibrium) & (df["Spread"].shift(1) > equilibrium)
#     units_short = np.full(shape=len(df), fill_value=np.nan)
#     units_short[np.where(short_entry == True)] = -1
#     units_short[np.where(short_exit == True)] = 0
#     units_short[:window - 1] = 0
#     units_short = pd.Series(units_short).fillna(method="pad")
#
#     num_units = units_long + units_short
#     num_units.index = df.index
#     spread_pct_change = (df["Spread"] - df["Spread"].shift(1)) / (
#             (df["DHR"] * np.abs(df["hr"])) + df["TMO"])
#     port_rets = spread_pct_change * num_units.shift(1)
#
#     cum_rets = port_rets.cumsum()
#     cum_rets += 1
#
#     portfolio_ending_value.append(cum_rets[-1])
#     cumulative_rets.append(cum_rets)
#     sharpe.append((port_rets.mean() / port_rets.std()) * np.sqrt(252))
#
# # -------------------------------------------------------
# port_rets = pd.DataFrame(index=sigmas)
# port_rets["Port_Rets"] = portfolio_ending_value
# port_rets["Sharpe"] = sharpe
# print(port_rets)
#
# plt.plot(port_rets["Port_Rets"], c="blue")
# plt.plot(port_rets["Sharpe"], c="red")
# plt.show()
#
# jet = plt.get_cmap('jet')
# colors = iter(jet(np.linspace(0, 1, len(cumulative_rets))))
# for i, j in enumerate(cumulative_rets):
#     plt.plot(j * 10_000, color=next(colors), label=f"{port_rets.index[i]}σ")
#     plt.legend(loc="upper left")
# plt.show()
