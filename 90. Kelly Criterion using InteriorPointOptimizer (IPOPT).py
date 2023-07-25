from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import talib as ta
import warnings
import yfinance as yf
from pyomo.environ import *
import sys

sys.path.append("C:\\msys64\\home\\domin\\Ipopt\\src\\Apps\\AmplSolver\\ipopt.exe")
warnings.filterwarnings("ignore")
sns.set_style("darkgrid")


def fetch_data(symbol, from_date, to_date, drop_cols=False, cols_to_drop=None):
    """ Fetch OHLC data."""
    df = yf.download(symbol, from_date, to_date)
    if drop_cols == True:
        df = df.drop(columns=cols_to_drop)

    return df


def outlier_treatment(data, lookback, n, method):
    """Use moving average to get a residual series from the
        original dataframe. We use the IQR and quantiles to
        make anomalous data-points nan values. Then we replace
        these nan values using interpolation with a linear method.
    """
    ma = pd.DataFrame(index=data.index)  # moving averages of each column
    for i, j in data.items():
        ma[f"{i}"] = ta.SMA(j.values, timeperiod=lookback)

    res = data - ma  # residual series

    Q1 = res.quantile(0.25)  # Quantile 1
    Q3 = res.quantile(0.75)  # Quantile 3
    IQR = Q3 - Q1  # IQR

    lw_bound = Q1 - (n * IQR)  # lower bound
    up_bound = Q3 + (n * IQR)  # upper bound

    res[res <= lw_bound] = np.nan  # set values outside range to NaN
    res[res >= up_bound] = np.nan

    res = res.interpolate(method=method)  # interpolation replaces NaN values

    prices = pd.DataFrame((res + ma))  # recompose original dataframe
    prices.dropna(inplace=True)  # drop NaN values

    return prices


def buildKCOptModel(returns: np.array, varcov: np.matrix,
                    rfr: float = 0):
    assert returns.shape[0] == varcov.shape[0]
    assert returns.shape[0] == varcov.shape[1]

    m = ConcreteModel()

    # Indices
    m.i = RangeSet(0, returns.shape[0] - 1)

    # Decision variables
    m.f = Var(m.i, domain=UnitInterval)

    # Parameters
    m.mu = Param(m.i,
                 initialize={i: m for i, m in zip(m.i, returns)})
    m.sigma = Param(m.i, m.i,
                    initialize={(i, j): varcov[i, j]
                                for i in m.i
                                for j in m.i})

    # Constraints
    @m.Constraint()
    def fullyInvestedConstraint(m):
        return sum(m.f[i] for i in m.i) == 1

    # Objective
    @m.Objective(sense=maximize)
    def objective(m):
        return (rfr + sum(m.f[i] * (m.mu[i] - rfr) for i in m.i) - sum(
            sum(m.f[i] * m.sigma[i, j] * m.f[j] for j in m.i)
            for i in m.i) / 2)

    return m


if __name__ == "__main__":

    # Fetch Data
    # DATA_STORE = "C:\\Users\\Dominic\\PycharmProjects\\End-to-end\\FEATURES.h5"
    # with pd.HDFStore(DATA_STORE) as store:
    #     df = store.get("FEATURES")
    #
    # df = df.dropna(axis=1, how="all")

    # Fetch OHLC Data
    symbol = ["TMO", "DHR", "RHI", "CSL"]  # ticker
    from_date = datetime(2000, 1, 1)
    to_date = datetime(2022, 2, 25)

    df = fetch_data(symbol=symbol, from_date=from_date,
                    to_date=to_date, drop_cols=True, cols_to_drop=["Adj Close"])

    # Kelly Criterion - determine the position size that optimizes your risk and returns
    df.drop(columns=["Low", "Open", "High", "Volume"], inplace=True)
    df.columns = df.columns.swaplevel()

    # Create returns
    returns = df.apply(lambda x: x / x.shift(1))
    returns = returns.rename(columns={'Close': 'returns'})

    # Mean and Variance of returns
    lookback = 252  # 1 year lookback in trading days
    means = returns.rolling(lookback).mean().rename(columns={'returns': 'mean'})
    var = returns.rolling(lookback).var().rename(columns={'returns': 'var'})

    df = pd.concat([returns, means, var], axis=1)

    # Get covariance matrices and transform to 3D array
    n = returns.shape[1]
    cov = returns.droplevel(1, axis=1).rolling(lookback).cov().values.reshape(-1, n, n)

    rfr = 0  # risk-free-rate

    fracs = np.zeros((df.shape[0], n))
    fracs[:] = np.nan
    g = np.zeros(df.shape[0])
    g[:] = np.nan
    for i, (ts, row) in enumerate(df.iterrows()):
        if i < lookback:
            continue
        means = row.loc[(slice(None), 'mean')].values
        var = row.loc[(slice(None), 'var')].values
        varcov = cov[i]
        np.fill_diagonal(varcov, var)
        model = buildKCOptModel(means, varcov, rfr)
        results = SolverFactory('ipopt',
                                executable="C:\\msys64\\home\\domin\\Ipopt\\src\\Apps\\AmplSolver\\ipopt.exe").solve(
            model)
        fracs[i] = np.array([model.f[j].value for j in model.f])
        g[i] = model.objective.expr()

    df_fracs = pd.DataFrame(fracs, columns=returns.columns,
                            index=returns.index).rename(
        columns={'returns': 'fraction'})
    df_fracs.columns = df_fracs.columns.droplevel(1)

    df_g = pd.DataFrame(g, index=returns.index)
    df_g.columns = pd.MultiIndex.from_arrays(
        [['g'], ['g']])

    print(df_fracs)

    for i in df_fracs.columns:
        plt.plot(df_fracs.loc[:, f"{i}"], label=f"{i}")
    plt.suptitle("KELLY CRITERION - PORTFOLIO ALLOCATION")
    plt.legend(loc="best")
    plt.show()
