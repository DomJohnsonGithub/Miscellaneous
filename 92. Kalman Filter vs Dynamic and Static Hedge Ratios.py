import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.tsa import stattools as ts
from pykalman import KalmanFilter, UnscentedKalmanFilter
import yfinance as yf

sns.set_style("darkgrid")

start = datetime(2000, 1, 1)
end = datetime.now()
stocks = ["TMO", "DHR"]

df = yf.download(stocks, start, end)
df = df.stack(1).swaplevel(0, 1).sort_index()
idx = pd.IndexSlice
tmo = df.loc[idx[stocks[0], :], "Close"].droplevel(0)
dhr = df.loc[idx[stocks[1], :], "Close"].droplevel(0)

df = pd.concat([tmo, dhr], axis=1)
df.columns = ["TMO", "DHR"]

# Correlation
corr = df["TMO"].corr(df["DHR"])
print("Correlation:", corr)

# Mutual Information
from sklearn.metrics import mutual_info_score


def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins=bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


rice = int(2 * (len(df) ** (1 / 3)))
mi = calc_MI(df.TMO.values, df.DHR.values, bins=rice)
print("Mutual Information: ", mi)

# Static Hedge Ratio
reg = LinearRegression().fit(np.reshape(df.TMO.values, (-1, 1)), np.reshape(df.DHR.values, (-1, 1)))
static_hedge_ratio = reg.coef_[0][0]
print(f'The static hedge ratio is {round(static_hedge_ratio, 2)}')

# Spread
spread = df.DHR - static_hedge_ratio * df.TMO

# stationary test
adf_results = sm.tsa.adfuller(spread.values)
print('ADF Statistic: %f' % adf_results[0])
print('p-value: %f' % adf_results[1])


#  Kalman filter regression
class KalmanFilterPairs:
    def __init__(self, delta, Ve, stock_1, stock_2):
        self.delta = delta  # parameter that adjusts the sensitivity of the state update
        self.Ve = Ve  # state noise variance
        self.stock_1 = stock_1  # observed variable
        self.stock_2 = stock_2  # variable that is part of the observation matrix

    def KalmanFilterRegression(self):
        trans_cov = self.delta / (1 - self.delta) * np.eye(2)  # How much random walk wiggles
        obs_mat = np.expand_dims(np.vstack([[self.stock_1], [np.ones(len(self.stock_1))]]).T, axis=1)

        kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,  # y is 1-dimensional, (alpha, beta) is 2-dimensional
                          initial_state_mean=[0, 0],
                          initial_state_covariance=np.ones((2, 2)),
                          transition_matrices=np.eye(2),
                          observation_matrices=obs_mat,
                          observation_covariance=self.Ve,
                          transition_covariance=trans_cov)

        # Use the observations y to get running estimates and errors for the state parameters
        state_means, state_covs = kf.filter(self.stock_2.values)
        return state_means, state_covs

    @staticmethod
    def draw_slope_intercept_changes(prices, state_means):
        """
        Plot the slope and intercept changes from the
        Kalman Filter calculated values.
        """
        pd.DataFrame(
            dict(
                slope=state_means[:, 0],
                intercept=state_means[:, 1]
            ), index=prices.index
        ).plot(subplots=True)
        plt.show()

    @staticmethod
    def calc_kalman_spread(state_means, prices):
        return prices.iloc[:, 1] - state_means[:, 0] * prices.iloc[:, 0]

    @staticmethod
    def plot_static_vs_dynamic_spread(static_spread, state_means, df):
        static_spread.plot(c="green", label="Static hedge ratio spread")
        kalman_spread = KalmanFilterPairs.calc_kalman_spread(state_means, prices=df)
        kalman_spread.plot(c="red", label="Kalman Filter Spread")
        plt.plot(df.index, state_means[:, 1], c="orange", label="Kalman Filter Mean")
        plt.show()


kfp = KalmanFilterPairs(delta=5e-6, Ve=0.5, stock_1=df.TMO, stock_2=df.DHR)
state_means, state_covariances = kfp.KalmanFilterRegression()  # state_means[:, 0] is the hedge ratio
kfp.draw_slope_intercept_changes(df, state_means)
kfp.plot_static_vs_dynamic_spread(static_spread=spread, state_means=state_means, df=df)

kalman_spread = kfp.calc_kalman_spread(state_means, df)

from arch.unitroot.unitroot import VarianceRatio

vratio_test = VarianceRatio(kalman_spread, lags=20).summary()
print(vratio_test)

# stationary test
adf_results = sm.tsa.adfuller(kalman_spread.values)
print('ADF Statistic: %f' % adf_results[0])
print('p-value: %f' % adf_results[1])

# ----------------------------
from sklearn.preprocessing import MinMaxScaler
from matplotlib.widgets import MultiCursor

sc = MinMaxScaler()

tmo_sc = sc.fit_transform(np.reshape(df.TMO.values, (-1, 1)))
dhr_sc = sc.fit_transform(np.reshape(df.DHR.values, (-1, 1)))

fig, axes = plt.subplots(nrows=2, ncols=1, dpi=50, sharex=True)
axes[0].plot(df.index, tmo_sc, c="red", label="TMO")
axes[0].plot(df.index, dhr_sc, c="blue", label="DHR")
axes[1].plot(df.index, kalman_spread, c="green")
axes[1].plot(df.index, state_means[:, 1], c="orange")

multi = MultiCursor(fig.canvas, axes, horizOn=True, c="black", lw=0.7, alpha=0.9)
axes[0].legend()
plt.show()
# ----------------------------


fig, axes = plt.subplots(nrows=3, ncols=1, dpi=50, sharex=True)
axes[0].plot(df.index, tmo_sc, c="red", label="TMO")
axes[0].plot(df.index, dhr_sc, c="blue", label="DHR")

axes[1].plot(df.index, kalman_spread, c="green")

axes[2].plot(df.index, np.log(df.TMO) / np.log(df.DHR), c="purple")

multi = MultiCursor(fig.canvas, axes, horizOn=True, c="black", lw=0.7, alpha=0.9)
axes[0].legend()
plt.show()
