import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
import seaborn as sns
from datetime import datetime
import pandas_datareader.data as pdr
from statsmodels.tsa.stattools import coint, adfuller
from arch.unitroot.cointegration import phillips_ouliaris
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import statsmodels.api as sm
from arch.unitroot.unitroot import VarianceRatio
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from itertools import accumulate
from collections import Counter
import yfinance as yf

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


if __name__ == "__main__":

    # Data
    stocks = ["DHR", "TMO"]
    start, end = datetime(2000, 1, 1), datetime.now()
    df = yf.download(stocks, start, end)["Close"]
    df.dropna(inplace=True)
    # ------------------------------------------------

    from statsmodels.regression.rolling import RollingOLS

    window = 40  # 12
    exog = sm.add_constant(df.DHR)
    rolling_ols = RollingOLS(endog=df.TMO, exog=exog, window=window, min_nobs=12).fit()
    beta = - np.array(rolling_ols.params.iloc[:, 1].dropna())
    df = df.iloc[window - 1:, :]

    df["Spread"] = np.log(df.TMO) + beta * np.log(df.DHR)
    df["hr"] = beta

    # ------------------------------------------------
    import arch

    garch = arch.arch_model(y=df.Spread, vol="GARCH", mean="constant", dist="normal", p=1, q=1).fit(update_freq=10,
                                                                                                    disp="off")
    garch = np.sqrt(garch.conditional_volatility)

    # --------------------------------------------------

    from statsmodels.tsa.statespace.tools import (
        constrain_stationary_univariate, unconstrain_stationary_univariate)


    # Quasi-likelihood stochastic volatility
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


    model = QLSV(df.Spread).fit(cov_type="robust")

    # ----------------------------------------------------

    vol = df.Spread.rolling(21).std() * np.sqrt(21 / 252)

    print(np.exp(-3))

    plt.plot(df.Spread / 100)
    plt.show()

    # -----------------------------------------------------

    plt.plot(garch.index, garch)
    plt.plot(garch.index, np.sqrt(np.exp(model.smoothed_state[0] / 2)))
    plt.plot(vol)
    plt.show()

    # ------------------------------------------------

    portfolio_ending_value = []
    sharpe = []
    sigmas = np.arange(0, 3.1, 0.1)
    cumulative_rets = []
    for sigma in sigmas:
        equilibrium = df["Spread"].mean()
        # std = np.sqrt(np.exp(model.smoothed_state[0] / 2))
        # std = garch
        # std = df["Spread"].std()
        std = vol
        upper_threshold = equilibrium + sigma * std
        lower_threshold = equilibrium - sigma * std

        # plt.plot(df.Spread)
        # plt.plot(df.index, upper_threshold)
        # plt.plot(df.index, lower_threshold)
        # plt.show()

        long_entry = (df["Spread"] <= lower_threshold) & (df["Spread"].shift(1) > lower_threshold)
        long_exit = (df["Spread"] >= equilibrium) & (df["Spread"].shift(1) < equilibrium)
        units_long = np.full(shape=len(df), fill_value=np.nan)
        units_long[np.where(long_entry == True)] = 1
        units_long[np.where(long_exit == True)] = 0
        units_long[:window - 1] = 0
        units_long = pd.Series(units_long).fillna(method="pad")

        short_entry = (df["Spread"] >= upper_threshold) & (df["Spread"].shift(1) < lower_threshold)
        short_exit = (df["Spread"] <= equilibrium) & (df["Spread"].shift(1) > equilibrium)
        units_short = np.full(shape=len(df), fill_value=np.nan)
        units_short[np.where(short_entry == True)] = -1
        units_short[np.where(short_exit == True)] = 0
        units_short[:window - 1] = 0
        units_short = pd.Series(units_short).fillna(method="pad")

        num_units = units_long + units_short
        num_units.index = df.index
        spread_pct_change = (df["Spread"] - df["Spread"].shift(1)) / (
                (df["DHR"] * np.abs(df["hr"])) + df["TMO"])
        port_rets = spread_pct_change * num_units.shift(1)

        cum_rets = port_rets.cumsum()
        cum_rets += 1

        portfolio_ending_value.append(cum_rets[-1])
        cumulative_rets.append(cum_rets)
        sharpe.append((port_rets.mean() / port_rets.std()) * np.sqrt(252))

    # -------------------------------------------------------
    port_rets = pd.DataFrame(index=sigmas)
    port_rets["Port_Rets"] = portfolio_ending_value
    port_rets["Sharpe"] = sharpe
    print(port_rets)

    plt.plot(port_rets["Port_Rets"], c="blue")
    plt.plot(port_rets["Sharpe"], c="red")
    plt.show()

    jet = plt.get_cmap('jet')
    colors = iter(jet(np.linspace(0, 1, len(cumulative_rets))))
    for i, j in enumerate(cumulative_rets):
        plt.plot(j * 10_000, color=next(colors), label=f"{port_rets.index[i]}σ")
        plt.legend(loc="upper left")
    plt.show()
