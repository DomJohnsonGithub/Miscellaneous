import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
import seaborn as sns
from datetime import datetime
import seaborn as sns
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.statespace.tools import (
    constrain_stationary_univariate, unconstrain_stationary_univariate)
import arch
from statsmodels.regression.rolling import RollingOLS

sns.set_style("darkgrid")


def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0


def half_life(spread):
    """
    Half life of mean-reversion.
    """
    spread_lag = spread.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]

    spread_ret = spread - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]

    spread_lag2 = sm.add_constant(spread_lag)

    model = sm.OLS(spread_ret, spread_lag2)
    res = model.fit()
    halflife = int(round(-np.log(2) / np.array(res.params)[1], 0))

    if halflife <= 0:
        halflife = 1
    return halflife


class QLSV(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        # Convert to log squares
        endog = np.log(endog ** 2)

        # Initialize the base model
        super(QLSV, self).__init__(endog, k_states=1, k_posdef=1,
                                   initialization='stationary')

        # Setup the observation covariance
        self['obs_intercept', 0, 0] = -1.27
        self['design', 0, 0] = 1
        self['obs_cov', 0, 0] = np.pi ** 2 / 2
        self['selection', 0, 0] = 1.

    @property
    def param_names(self):
        return ['phi', 'sigma2_eta', 'mu']

    @property
    def start_params(self):
        return np.r_[0.9, 1., 0.]

    def transform_params(self, params):
        return np.r_[constrain_stationary_univariate(params[:1]), params[1] ** 2, params[2:]]

    def untransform_params(self, params):
        return np.r_[unconstrain_stationary_univariate(params[:1]), params[1] ** 0.5, params[2:]]

    def update(self, params, **kwargs):
        super(QLSV, self).update(params, **kwargs)

        gamma = params[2] * (1 - params[0])
        self['state_intercept', 0, 0] = gamma
        self['transition', 0, 0] = params[0]
        self['state_cov', 0, 0] = params[1]


def tls(X, y):
    """Total Least Squares"""
    if len(X.shape) is 1:
        n = 1
        X = X.reshape(len(X), 1)
    else:
        n = np.array(X).shape[1]  # the number of variable of X

    Z = np.vstack((X.T, y)).T
    U, s, Vt = np.linalg.svd(Z, full_matrices=True)

    V = Vt.T
    Vxy = V[:n, n:]
    Vyy = V[n:, n:]
    a_tls = - Vxy / Vyy  # total least squares soln

    Xtyt = - Z.dot(V[:, n:]).dot(V[:, n:].T)
    Xt = Xtyt[:, :n]  # X error
    y_tls = (X + Xt).dot(a_tls)

    fro_norm = np.linalg.norm(Xtyt, 'fro')  # Frobenius norm

    return float(a_tls)


if __name__ == "__main__":

    # Data
    stocks = ["DHR", "TMO"]
    start, end = datetime(2000, 1, 1), datetime.now()
    df = yf.download(stocks, start, end)["Close"]
    df.dropna(inplace=True)

    # Log retruns
    log_rets = np.log1p(df.TMO.pct_change()[1:])

    # ----------- Volatility Models can be used for Pairs Trading ------------- #
    # # Quasi-likelihood stochastic volatility
    # endog = log_rets - log_rets.mean()
    # model = QLSV(endog)
    # res = model.fit(cov_type="robust")
    #
    # # Garch Model
    # garch = arch.arch_model(y=log_rets * 100, vol="GARCH", mean="constant", p=1, q=1).fit().conditional_volatility
    #
    # # Plot of Volatilities
    # fig, ax = plt.subplots(figsize=(13, 5))
    # ax.plot(log_rets.index, np.abs(log_rets), c="blue")
    # ax.plot(log_rets.index, np.exp(res.smoothed_state[0] / 2), c="orange")
    # ax.plot(log_rets.index, np.sqrt(garch), c="green")
    # ax.set(title='Figure 1 of Harvey et al. (1994)')
    # plt.show()
    # print(garch)
    # ----------------------------------------------------------------------------

    # # Total Least Squares
    # c = tls(np.array(df.DHR), np.array(df.TMO))
    #
    # # RollingTLS
    # hedge_ratios = []
    # lookback = 200
    # for i in range(len(df)):
    #     hedge_ratios.append(float(
    #         - np.linalg.svd(np.array(df.iloc[i - lookback + 1:i + 1, :]), full_matrices=True)[2].T[1, 0] /
    #         np.linalg.svd(np.array(df.iloc[i - lookback + 1:i + 1, :]), full_matrices=True)[2].T[1, 1]))
    # df = df.iloc[lookback - 1:, :]
    # df["beta"] = np.array(hedge_ratios)[lookback - 1:]
    # print(df)

    # Rolling OLS
    window = 40  # 12
    exog = sm.add_constant(df.DHR)
    rolling_ols = RollingOLS(endog=df.TMO, exog=exog, window=window, min_nobs=12).fit()
    beta = - np.array(rolling_ols.params.iloc[:, 1].dropna())
    df = df.iloc[window - 1:, :]

    # spread = df.TMO + beta * df.DHR
    log_spread = np.log(df.TMO) + beta * np.log(df.DHR)
    print(half_life(log_spread[-100:]))
    print(half_life(log_spread))

    # -----------------------------
    # Equilibrium - mean value
    # print(np.mean(log_spread))

    # # Time spent above or below zero consecutively
    # from itertools import accumulate
    # from collections import Counter
    # ts = log_spread
    # treshold = 0
    # groups = accumulate([0] + [(a >= treshold) != (b >= treshold) for a, b in zip(ts, ts[1:])])
    # counts = sorted(Counter(groups).items())
    # above = [c for n, c in counts if (n % 2 == 0) == (ts[0] >= treshold)]
    # below = [c for n, c in counts if (n % 2 == 0) != (ts[0] >= treshold)]
    # print(np.median(above))
    # print(np.median(below))

    # can use this above for time based stops
    # --------------------------------

    # Equilibrium - mean value for the static threshold
    print(np.mean(log_spread))

    # Bollinger Band Threshold
    print(log_spread.std())
    print(log_spread.rolling(200).std())

    window = 200
    sigma = 1.5

    # BackTest
    df["Spread"] = log_spread
    df["hr"] = beta
    df["Equilibrium"] = df["Spread"].rolling(window=window).mean()
    df["Upper_Threshold"] = df["Spread"].rolling(window=window).mean() + sigma * df["Spread"].rolling(
        window=window).std()
    df["Lower_Threshold"] = df["Spread"].rolling(window=window).mean() + sigma * -df["Spread"].rolling(
        window=window).std()
    df.dropna(inplace=True)

    df["Long_Entry"] = (df["Spread"] <= df["Lower_Threshold"]) & (df["Spread"].shift(1) > df["Lower_Threshold"])
    df["Long_Exit"] = (df["Spread"] >= df["Equilibrium"]) & (df["Spread"].shift(1) < df["Equilibrium"])
    df["Units_Long"] = np.nan
    df.loc[df["Long_Entry"], "Units_Long"] = 1
    df.loc[df["Long_Exit"], "Units_Long"] = 0
    df["Units_Long"][:window - 1] = 0
    df["Units_Long"] = df["Units_Long"].fillna(method="pad")

    df["Short_Entry"] = (df["Spread"] >= df["Upper_Threshold"]) & (df["Spread"].shift(1) < df["Upper_Threshold"])
    df["Short_Exit"] = (df["Spread"] <= df["Equilibrium"]) & (df["Spread"].shift(1) > df["Equilibrium"])
    df["Units_Short"] = np.nan
    df.loc[df["Short_Entry"], "Units_Short"] = -1
    df.loc[df["Short_Exit"], "Units_Short"] = 0
    df["Units_Short"][:window - 1] = 0
    df["Units_Short"] = df["Units_Short"].fillna(method="pad")

    df["Num_Units"] = df["Units_Long"] + df["Units_Short"]
    df["Spread_pct_change"] = (df["Spread"] - df["Spread"].shift(1)) / (
            (df["DHR"] * np.abs(df["hr"])) + df["TMO"])
    df["Port_rets"] = df["Spread_pct_change"] * df["Num_Units"].shift(1)

    df["Cum_rets"] = df["Port_rets"].cumsum()
    df["Cum_rets"] = (df["Cum_rets"] + 1) * 10_000

    print("Portfolio cumulative returns: ", np.round(df["Cum_rets"][-1], 2))

    try:
        sharpe = ((df["Port_rets"].mean() / df["Port_rets"].std()) * np.sqrt(252))
        print("\nSharpe Ratio (A): ", sharpe)

        geometric_mean_port_return = np.prod(df.Port_rets + 1) ** (1 / len(df["Port_rets"])) - 1
        geometric_sharpe_ratio = geometric_mean_port_return / df["Port_rets"].std() * np.sqrt(252)
        print("Geometric Sharpe Ratio (A): ", geometric_sharpe_ratio)

        mu = df["Port_rets"].mean()
        sigma = df["Port_rets"].std()
        K = df["Port_rets"].kurt()
        S = df["Port_rets"].std()
        Zc = 1.644854
        Z = -Zc + ((-Zc ** 2 - 1) / 6) * S + ((-Zc ** 3 - 3 * -Zc) / 24) * K - ((2 * -Zc ** 3 - 5 * -Zc) / 36) * S ** 2
        MVAR = mu - Z * sigma
        modified_sharpe_ratio = mu / MVAR * np.sqrt(252)
        print("Modified Sharpe Ratio (A): ", modified_sharpe_ratio)

        geometric_modified_sharpe_ratio = geometric_mean_port_return / MVAR * np.sqrt(252)
        print("Modified Geometric Sharpe Ratio (A): ", geometric_modified_sharpe_ratio)

    except ZeroDivisionError:
        sharpe = 0.0
        print("\nSharpe Ratio (A): ", sharpe)

    peaks = df["Cum_rets"].cummax()
    drawdown = (df["Cum_rets"] - peaks) / peaks * 100

    fig, axes = plt.subplots(nrows=3, ncols=1, dpi=50, sharex=True)
    axes[0].plot(df["Spread"])
    axes[0].plot(df["Equilibrium"], c="red")
    axes[0].plot(df["Upper_Threshold"], c="orange")
    axes[0].plot(df["Lower_Threshold"], c="orange")
    axes[1].plot(df["Cum_rets"])
    axes[2].plot(drawdown)
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------

    # # Hedge Ratio
    # df["hr"] = beta
    #
    # # Spread
    # df["spread"] = df[stocks[1]] + (df[stocks[0]] * df["hr"])
    #
    # # ------------------------------------------------------------------------------------------------------------------
    # entry_threshold = 2
    # exit_threshold = 0
    #
    # # Stationarity test on spread
    # adf_results = adfuller(df["spread"])
    # print("\nADF Statistic: %f" % adf_results[0])
    # print("p-value: %f" % adf_results[1])
    #
    # # Hurst Exponent of last year
    # print("\nHurst Exponent: ", hurst(df.spread[-252:].values))
    #
    # # Variance Ratio to see if a random walk or stationary/mean-reverting
    # # A statistical test to significance of hurst exponent
    # vratio_test = VarianceRatio(df.spread[-252:].values, lags=10)
    # print("\n", vratio_test.summary())
    #
    # # Half-life for z-score window
    # mean_reversion_life = half_life(df.spread[-252:])
    # print("\nHalf-Life of Mean Reversion: ", mean_reversion_life, "days")
    #
    # # Z-score for entry and exits of trading strategy
    # mean_spread = df.spread.rolling(mean_reversion_life).mean()
    # std_spread = df.spread.rolling(mean_reversion_life).std()
    # df["z_score"] = (df.spread - mean_spread) / std_spread
    #
    # # Create Positions #
    #
    # # Number of units long
    # df["Long_Entry"] = ((df["z_score"] < - entry_threshold)) & (df["z_score"].shift(1) > - entry_threshold)
    # df["Long_Exit"] = ((df["z_score"] > - exit_threshold)) & (df["z_score"].shift(1) < - exit_threshold)
    # df["Units_Long"] = np.nan
    # df.loc[df["Long_Entry"], "Units_Long"] = 1
    # df.loc[df["Long_Exit"], "Units_Long"] = 0
    # # df["Units_Long"][0] = 0
    # df.iloc[0, 7] = 0
    # df["Units_Long"] = df["Units_Long"].fillna(method="pad")
    #
    # # Number of units short
    # df["Short_Entry"] = ((df["z_score"] > entry_threshold)) & (df["z_score"].shift(1) < entry_threshold)
    # df["Short_Exit"] = ((df["z_score"] < exit_threshold)) & (df["z_score"].shift(1) > exit_threshold)
    # df["Units_Short"] = np.nan
    # df.loc[df["Short_Entry"], "Units_Short"] = -1
    # df.loc[df["Short_Exit"], "Units_Short"] = 0
    # # df["Units_Short"][0] = 0
    # df.iloc[0, 10] = 0
    # df["Units_Short"] = df["Units_Short"].fillna(method="pad")
    #
    # # Plot prices with spread and z-score
    # fig, axes = plt.subplots(nrows=3, ncols=1, dpi=50, sharex=True)
    # from sklearn.preprocessing import MinMaxScaler
    # sc = MinMaxScaler()
    # axes[0].plot(df.index, sc.fit_transform(np.reshape(df[stocks[0]].values, (-1, 1))), c="blue", label=stocks[0])
    # axes[0].plot(df.index, sc.fit_transform(np.reshape(df[stocks[1]].values, (-1, 1))), c="red", label=stocks[1])
    # axes[0].set_ylim(-0.001, 1.001)
    # axes[0].legend()
    # axes[1].plot(df["spread"], c="seagreen")
    # axes[2].plot(df["z_score"], c="purple")
    # long_entry = df[df["Long_Entry"] == True]["z_score"]
    # long_exit = df[df["Long_Exit"] == True]["z_score"]
    # short_entry = df[df["Short_Entry"] == True]["z_score"]
    # short_exit = df[df["Short_Exit"] == True]["z_score"]
    # axes[2].scatter(long_entry.index, long_entry, c="orange", marker="^", s=80)
    # axes[2].scatter(short_entry.index, short_entry, c="orange", marker="v", s=80)
    # axes[2].scatter(long_exit.index, long_exit, c="yellow", marker="^", s=80)
    # axes[2].scatter(short_exit.index, short_exit, c="yellow", marker="v", s=80)
    # axes[2].axhline(y=entry_threshold, c="grey", lw=0.9, ls="--")
    # axes[2].axhline(y=exit_threshold, c="grey", lw=0.9, ls="--")
    # axes[2].axhline(y=-entry_threshold, c="grey", lw=0.9, ls="--")
    # axes[2].axhline(y=-exit_threshold, c="grey", lw=0.9, ls="--")
    # multi = MultiCursor(fig.canvas, axes, horizOn=True, c="black", alpha=0.9)
    # plt.subplots_adjust(left=0.017, bottom=0.017, right=0.99, top=0.99, hspace=0.02)
    # plt.show()
    #
    # df["Num_Units"] = df["Units_Long"] + df["Units_Short"]
    # df["Spread_pct_change"] = (df["spread"] - df["spread"].shift(1)) / (
    #         (df["DHR"] * np.abs(df["hr"])) + df["TMO"])
    # df["Port_rets"] = df["Spread_pct_change"] * df["Num_Units"].shift(1)
    #
    # df["Cum_rets"] = df["Port_rets"].cumsum()
    # df["Cum_rets"] = df["Cum_rets"] + 1
    #
    #
    # try:
    #     sharpe = ((df["Port_rets"].mean() / df["Port_rets"].std()) * np.sqrt(252))
    #     print("\nSharpe Ratio (A): ", sharpe)
    #
    #     geometric_mean_port_return = np.prod(df.Port_rets + 1) ** (1 / len(df["Port_rets"])) - 1
    #     geometric_sharpe_ratio = geometric_mean_port_return / df["Port_rets"].std() * np.sqrt(252)
    #     print("Geometric Sharpe Ratio (A): ", geometric_sharpe_ratio)
    #
    #     mu = df["Port_rets"].mean()
    #     sigma = df["Port_rets"].std()
    #     K = df["Port_rets"].kurt()
    #     S = df["Port_rets"].std()
    #     Zc = 1.644854
    #     Z = -Zc + ((-Zc ** 2 - 1) / 6) * S + ((-Zc ** 3 - 3 * -Zc) / 24) * K - ((2 * -Zc ** 3 - 5 * -Zc) / 36) * S ** 2
    #     MVAR = mu - Z * sigma
    #     modified_sharpe_ratio = mu / MVAR * np.sqrt(252)
    #     print("Modified Sharpe Ratio (A): ", modified_sharpe_ratio)
    #
    #     geometric_modified_sharpe_ratio = geometric_mean_port_return / MVAR * np.sqrt(252)
    #     print("Modified Geometric Sharpe Ratio (A): ", geometric_modified_sharpe_ratio)
    #
    # except ZeroDivisionError:
    #     sharpe = 0.0
    #     print("\nSharpe Ratio (A): ", sharpe)
    #
    #
    # peaks = df["Cum_rets"].cummax()
    # drawdown = (df["Cum_rets"] - peaks)/peaks * 100
    #
    # fig, axes = plt.subplots(nrows=2, ncols=1, dpi=50, sharex=True)
    # axes[0].plot(df.Cum_rets, c="blue")
    # axes[1].plot(drawdown, c="red")
    # plt.show()
