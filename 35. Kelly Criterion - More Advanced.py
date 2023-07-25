from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as pdr
import yfinance as yf
from numba import jit, float64
import time

sns.set_style("darkgrid")


# --------------------------------------------------------------------------------------------

# Long only buy-and-hold Kelly Criterion daily re-balance
# if f* > 1 we use leverage and will have negative
def calcKelly(mean, std, r):
    return (mean - r) / std ** 2


def getKellyFactor(returns: pd.Series, r=0.01,
                   max_leverage=None, periods=252, rolling=True):
    '''
    Calculates the Kelly Factor for each time step based
    on the parameters provided.
    '''
    if rolling:
        std = returns.rolling(periods).std()
        mean = returns.rolling(periods).mean()
    else:
        std = returns.expanding(periods).std()
        mean = returns.expanding(periods).mean()

    r_daily = np.log((1 + r) ** (1 / 252))
    print("r_daily", r_daily)
    kelly_factor = calcKelly(mean, std, r_daily)

    # No shorts
    kelly_factor = np.where(kelly_factor < 0, 0, kelly_factor)
    if max_leverage is not None:
        kelly_factor = np.where(kelly_factor > max_leverage,
                                max_leverage, kelly_factor)

    return kelly_factor


ticker = 'SPY'
yfObj = yf.Ticker(ticker)
data = yfObj.history(start='1993-01-01', end='2020-12-31')
# Drop unused columns
data.drop(['Open', 'High', 'Low', 'Volume', 'Dividends',
           'Stock Splits'], axis=1, inplace=True)

print(data)


def LongOnlyKellyStrategy(data, r=0.02, max_leverage=None, periods=252, rolling=True):
    data['returns'] = data['Close'] / data['Close'].shift(1)
    data['log_returns'] = np.log(data['returns'])
    data['kelly_factor'] = getKellyFactor(data['log_returns'], r, max_leverage, periods, rolling)

    cash = np.zeros(data.shape[0])
    equity = cash.copy()
    portfolio = cash.copy()
    portfolio[0] = 1
    cash[0] = 1
    for i, _row in enumerate(data.iterrows()):
        row = _row[1]
        if np.isnan(row['kelly_factor']):
            portfolio[i] += portfolio[i - 1]
            cash[i] += cash[i - 1]
            continue

        portfolio[i] += cash[i - 1] * (1 + r) ** (1 / 252) + equity[i - 1] * row['returns']
        equity[i] += portfolio[i] * row['kelly_factor']
        cash[i] += portfolio[i] * (1 - row['kelly_factor'])

    data['cash'] = cash
    data['equity'] = equity
    data['portfolio'] = portfolio
    data['strat_returns'] = data['portfolio'] / data['portfolio'].shift(1)
    data['strat_log_returns'] = np.log(data['strat_returns'])
    data['strat_cum_returns'] = data['strat_log_returns'].cumsum()
    data['cum_returns'] = data['log_returns'].cumsum()

    return data


kelly = LongOnlyKellyStrategy(data.copy())

fig, ax = plt.subplots(2, figsize=(12, 8), sharex=True)

ax[0].plot(np.exp(kelly['cum_returns']) * 100, label='Buy and Hold')
ax[0].plot(np.exp(kelly['strat_cum_returns']) * 100, label='Kelly Model')
ax[0].set_ylabel('Returns (%)')
ax[0].set_title('Buy-and-hold and Long-Only Strategy with Kelly Sizing')
ax[0].legend()

ax[1].plot(kelly['kelly_factor'])
ax[1].set_ylabel('Leverage')
ax[1].set_xlabel('Date')
ax[1].set_title('Kelly Factor')

plt.tight_layout()
plt.show()


def getStratStats(log_returns: pd.Series, risk_free_rate: float = 0.02):
    stats = {}  # Total Returns
    stats['tot_returns'] = np.exp(log_returns.sum()) - 1

    # Mean Annual Returns
    stats['annual_returns'] = np.exp(log_returns.mean() * 252) - 1

    # Annual Volatility
    stats['annual_volatility'] = log_returns.std() * np.sqrt(252)

    # Sortino Ratio
    annualized_downside = log_returns.loc[log_returns < 0].std() * \
                          np.sqrt(252)
    stats['sortino_ratio'] = (stats['annual_returns'] - risk_free_rate) / annualized_downside

    # Sharpe Ratio
    stats['sharpe_ratio'] = (stats['annual_returns'] - risk_free_rate) / stats['annual_volatility']

    # Max Drawdown
    cum_returns = log_returns.cumsum() - 1
    peak = cum_returns.cummax()
    drawdown = peak - cum_returns
    max_idx = drawdown.argmax()
    stats['max_drawdown'] = 1 - np.exp(cum_returns[max_idx]) / np.exp(peak[max_idx])

    # Max Drawdown Duration
    strat_dd = drawdown[drawdown == 0]
    strat_dd_diff = strat_dd.index[1:] - strat_dd.index[:-1]
    strat_dd_days = strat_dd_diff.map(lambda x: x.days).values
    strat_dd_days = np.hstack([strat_dd_days,
                               (drawdown.index[-1] - strat_dd.index[-1]).days])
    stats['max_drawdown_duration'] = strat_dd_days.max()

    return stats


max_leverage = np.arange(1, 6)

fig, ax = plt.subplots(2, figsize=(15, 10), sharex=True)
data_dict = {}
df_stats = pd.DataFrame()

for l in max_leverage:
    kelly = LongOnlyKellyStrategy(data.copy(), max_leverage=l)
    data_dict[l] = kelly.copy()

    ax[0].plot(np.exp(kelly['strat_cum_returns']) * 100,
               label=f'Max Leverage = {l}')
    ax[1].plot(kelly['kelly_factor'], label=f'Max Leverage = {l}')
    stats = getStratStats(kelly['strat_log_returns'])
    df_stats = pd.concat([df_stats,
                          pd.DataFrame(stats, index=[f'Leverage={l}'])])

ax[0].plot(np.exp(kelly['cum_returns']) * 100, label='Buy and Hold',
           linestyle=':')
ax[0].set_ylabel('Returns (%)')
ax[0].set_title('Buy-and-hold and Long-Only Strategy with Kelly Sizing')
ax[0].legend()

ax[1].set_ylabel('Leverage')
ax[1].set_xlabel('Date')
ax[1].set_title('Kelly Factor')

plt.tight_layout()
plt.show()

# View statistics
stats = pd.DataFrame(getStratStats(kelly['log_returns']), index=['Buy and Hold'])
df_stats = pd.concat([stats, df_stats])
print(df_stats.iloc[:, 0:4])
print(df_stats.iloc[:, 4:])

# different time frames
max_leverage = 3

periods = 252 * np.array([0.25, 0.5, 0.75, 1, 2, 3, 4])

fig, ax = plt.subplots(2, figsize=(15, 10), sharex=True)
data_dict = {}
df_stats = pd.DataFrame()
for p in periods:
    p = int(p)
    kelly = LongOnlyKellyStrategy(data.copy(), periods=p,
                                  max_leverage=max_leverage)
    data_dict[p] = kelly.copy()
    ax[0].plot(np.exp(kelly['strat_cum_returns']) * 100,
               label=f'Days = {p}')
    ax[1].plot(kelly['kelly_factor'], label=f'Days = {p}', linewidth=0.5)
    stats = getStratStats(kelly['strat_log_returns'])
    df_stats = pd.concat([df_stats,
                          pd.DataFrame(stats, index=[f'Days={p}'])])

ax[0].plot(np.exp(kelly['cum_returns']) * 100, label='Buy and Hold',
           linestyle=':')
ax[0].set_ylabel('Returns (%)')
ax[0].set_title(
    'Buy-and-hold and Long-Only Strategy with Kelly Sizing ' +
    'and Variable Lookback Periods')
ax[0].legend()

ax[1].set_ylabel('Leverage')
ax[1].set_xlabel('Date')
ax[1].set_title('Kelly Factor')

plt.tight_layout()
plt.show()

stats = pd.DataFrame(getStratStats(kelly['log_returns']), index=['Buy and Hold'])
df_stats = pd.concat([stats, df_stats])
print(df_stats.iloc[:, 0:4])
print(df_stats.iloc[:, 4:])


# Kelly money management for trading strategy
def KellySMACrossOver(data, SMA1=50, SMA2=200, r=0.01,
                      periods=252, max_leverage=None, rolling=True):
    '''
    Sizes a simple moving average cross-over strategy according
    to the Kelly Criterion.
    '''
    data['returns'] = data['Close'] / data['Close'].shift(1)
    data['log_returns'] = np.log(data['returns'])
    # Calculate positions
    data['SMA1'] = data['Close'].rolling(SMA1).mean()
    data['SMA2'] = data['Close'].rolling(SMA2).mean()
    data['position'] = np.nan
    data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, 0)
    data['position'] = data['position'].ffill().fillna(0)
    data['_strat_returns'] = data['position'].shift(1) * \
                             data['returns']
    data['_strat_log_returns'] = data['position'].shift(1) * \
                                 data['log_returns']
    # Calculate Kelly Factor using the strategy's returns
    kf = getKellyFactor(data['_strat_log_returns'], r,
                        max_leverage, periods, rolling)
    data['kelly_factor'] = kf

    cash = np.zeros(data.shape[0])
    equity = np.zeros(data.shape[0])
    portfolio = cash.copy()
    portfolio[0] = 1
    cash[0] = 1
    for i, _row in enumerate(data.iterrows()):
        row = _row[1]
        if np.isnan(kf[i]):
            portfolio[i] += portfolio[i - 1]
            cash[i] += cash[i - 1]
            continue

        portfolio[i] += cash[i - 1] * (1 + r) ** (1 / 252) + equity[i - 1] * row['returns']
        equity[i] += portfolio[i] * row['kelly_factor']
        cash[i] += portfolio[i] * (1 - row['kelly_factor'])

    data['cash'] = cash
    data['equity'] = equity
    data['portfolio'] = portfolio
    data['strat_returns'] = data['portfolio'] / data['portfolio'].shift(1)
    data['strat_log_returns'] = np.log(data['strat_returns'])
    data['strat_cum_returns'] = data['strat_log_returns'].cumsum()
    data['cum_returns'] = data['log_returns'].cumsum()
    return data


kelly_sma = KellySMACrossOver(data.copy(), max_leverage=3)

fig, ax = plt.subplots(2, figsize=(15, 8), sharex=True)

ax[0].plot(np.exp(kelly_sma['cum_returns']) * 100, label='Buy-and-Hold')
ax[0].plot(np.exp(kelly_sma['strat_cum_returns']) * 100, label='SMA-Kelly')
ax[0].plot(np.exp(kelly_sma['_strat_log_returns'].cumsum()) * 100, label='SMA')
ax[0].set_ylabel('Returns (%)')
ax[0].set_title('Moving Average Cross-Over Strategy with Kelly Sizing')
ax[0].legend()

ax[1].plot(kelly_sma['kelly_factor'])
ax[1].set_ylabel('Leverage')
ax[1].set_xlabel('Date')
ax[1].set_title('Kelly Factor')

plt.tight_layout()
plt.show()

sma_stats = pd.DataFrame(getStratStats(kelly_sma['log_returns']),
                         index=['Buy and Hold'])
sma_stats = pd.concat([sma_stats,
                       pd.DataFrame(getStratStats(kelly_sma['strat_log_returns']),
                                    index=['Kelly SMA Model'])])
sma_stats = pd.concat([sma_stats,
                       pd.DataFrame(getStratStats(kelly_sma['_strat_log_returns']),
                                    index=['SMA Model'])])

print(sma_stats)

# --------------------------------------------------------------
# Multi-asset
from pyomo.environ import (ConcreteModel, RangeSet, Var, maximize, Param, UnitInterval,
                           SolverFactory, PercentFraction, PositiveReals, Reals, Any, NegativeReals)

start = '2020-01-01'
end = '2021-12-31'
tickers = ['AAPL', 'F', 'GE', 'CVX']
yfObj = yf.Tickers(tickers)
data = yfObj.history(start=start, end=end)
data.drop(['High', 'Low', 'Open', 'Volume', 'Stock Splits', 'Dividends'],
          axis=1, inplace=True)
data.columns = data.columns.swaplevel()

print(data)


def buildKCOptModel(returns: np.array, varcov: np.matrix,
                    rfr: float = 0):
    assert returns.shape[0] == varcov.shape[0]
    assert returns.shape[0] == varcov.shape[1]

    m = ConcreteModel()

    # Indices
    m.i = RangeSet(0, returns.shape[0] - 1)

    # Decision variables - what weights can be given for each random variable
    m.f = Var(m.i, domain=UnitInterval)

    # Parameters
    m.mu = Param(m.i, initialize={i: m for i, m in zip(m.i, returns)})
    m.sigma = Param(m.i, m.i, initialize={(i, j): varcov[i, j] for i in m.i for j in m.i})

    # Constraints - all money invested if 1
    @m.Constraint()
    def fullyInvestedConstraint(m):
        return sum(m.f[i] for i in m.i) == 1

    # Objective
    @m.Objective(sense=maximize)
    def objective(m):
        return (rfr + sum(m.f[i] * (m.mu[i] - rfr) for i in m.i) - \
                sum(
                    sum(m.f[i] * m.sigma[i, j] * m.f[j] for j in m.i)
                    for i in m.i) / 2)

    return m


def getKCOpt(data: pd.DataFrame, lookback=252, rfr=0, print_model_summary=False):
    global model
    returns = data.loc[:, (slice(None), 'Close')] / \
              data.loc[:, (slice(None), 'Close')].shift(1)
    returns = returns.rename(columns={'Close': 'returns'})
    means = returns.rolling(lookback).mean().rename(
        columns={'returns': 'mean'})
    var = returns.rolling(lookback).var().rename(
        columns={'returns': 'var'})
    df = pd.concat([returns, means, var], axis=1)
    # Get covariance matrices and transform to 3D array
    n = returns.shape[1]
    cov = returns.droplevel(1, axis=1).rolling(lookback).cov().values.reshape(
        -1, n, n)

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
        results = SolverFactory(
            'ipopt',
            executable="C:\\msys64\\home\\domin\\Ipopt\\src\\Apps\\AmplSolver\\ipopt.exe").solve(model)
        fracs[i] = np.array([model.f[j].value for j in model.f])
        g[i] = model.objective.expr()

    if print_model_summary:
        model.pprint()
    else:
        pass

    df_fracs = pd.DataFrame(fracs, columns=returns.columns,
                            index=returns.index).rename(
        columns={'returns': 'fraction'})
    df_g = pd.DataFrame(g, index=returns.index)
    df_g.columns = pd.MultiIndex.from_arrays(
        [['g'], ['g']])

    return pd.concat([data, df, df_fracs, df_g], axis=1)


df = getKCOpt(data, lookback=252, rfr=0.02, print_model_summary=True)

print(df)

fig, ax = plt.subplots(2, figsize=(12, 8), sharex=True)

ax[0].plot(df.loc[:, (slice(None), 'fraction')] * 100)
ax[0].set_ylabel('Portfolio Allocation (%)')
ax[0].set_title('Kelly Optimal Portfolio Allocation')
labels = [i for i in list(df.columns.levels[0]) if i in tickers]
ax[0].legend(labels=labels)

ax[1].plot((df.loc[:, 'g'] - 1) * 100)

ax[1].set_xlabel('Date')
ax[1].set_ylabel('Daily Growth Rate (%)')
ax[1].set_title('Kelly Optimal Growth Rate')

plt.show()


# -----------------------------------------------------------------
# Comparing Heuristic vs Quadratic Programming Kelly criterion
def buildKCOptModel(returns: np.array, varcov: np.matrix,
                    rfr: float = 0, lam_max: float = 3):
    m = ConcreteModel()

    # Indices
    m.i = RangeSet(0, returns.shape[0] - 1)

    # Decision variables
    m.f = Var(m.i, domain=Reals)
    m.x = Var(m.i, domain=Reals)

    # Parameters
    m.mu = Param(m.i,
                 initialize={i: m for i, m in zip(m.i, returns)})
    m.sigma = Param(m.i, m.i,
                    initialize={(i, j): varcov[i, j]
                                for i in m.i
                                for j in m.i})
    m.lam_max = lam_max

    # Constraints
    @m.Constraint()
    def maxLeverageConstraint(m):
        return sum(m.x[i] for i in m.i) <= m.lam_max

    @m.Constraint(m.i)
    def posFraction(m, i):
        return m.x[i] - m.f[i] >= 0

    @m.Constraint(m.i)
    def negFraction(m, i):
        return m.x[i] + m.f[i] >= 0

    # Objective
    @m.Objective(sense=maximize)
    def objective(m):
        return (rfr + sum(m.f[i] * (m.mu[i] - rfr) for i in m.i) - sum(
            sum(m.f[i] * m.sigma[i, j] * m.f[j] for j in m.i)
            for i in m.i) / 2)

    return m


class OptimalAllocation:
    def __init__(self, tickers: list, max_leverage: float = 3,
                 lookback: int = 252, rfr: float = 0,
                 start: str = "2000-01-01", end: str = "2021-12-31",
                 rebalance_freq: int = 1):

        self.tickers = tickers
        self.max_leverage = max_leverage
        self.lookback = lookback
        self.start = start
        self.end = end
        self.rfr = rfr
        self.rebalance_freq = rebalance_freq

        self.data = self._getData()
        self._calcStats()

    def _getData(self):
        yfObj = yf.Tickers(self.tickers)
        data = yfObj.history(start=self.start, end=self.end)
        data.drop(["High", "Low", "Open", "Volume", "Stock Splits",
                   "Dividends"], axis=1, inplace=True)
        data.columns = data.columns.swaplevel()
        data.dropna(inplace=True)
        return data

    def _calcStats(self):
        # Calc returns
        returns = self.data.loc[:, (slice(None), "Close")] / \
                  self.data.loc[:, (slice(None), "Close")].shift(1)
        returns = returns.rename(columns={"Close": "returns"})

        means = returns.rolling(self.lookback).mean().rename(
            columns={"returns": "mean"})
        # Calculate covariance matrices and transform to 3D array
        n = returns.shape[1]
        self.cov = returns.droplevel(1, axis=1).rolling(
            self.lookback).cov().values.reshape(-1, n, n)
        self.data = pd.concat([self.data, returns, means], axis=1)

    def calcKCUnconstrainedAllocation(self):
        '''
        Calculates the allocation fractions for the unconstrained
        Kelly Criterion case.
        '''
        fracs = np.zeros((len(self.data), len(self.tickers)))
        fracs[:] = np.nan
        for i, (ts, row) in enumerate(self.data.iterrows()):
            if i < self.lookback:
                continue
            means = row.loc[(slice(None)), "mean"].values

            F = np.dot(means, np.linalg.inv(self.cov[i]))
            fracs[i] = F

        df_fracs = pd.DataFrame(fracs, index=self.data.index)
        midx = pd.MultiIndex.from_arrays(
            [self.tickers, len(self.tickers) * ['unconstrained_fracs']])
        df_fracs.columns = midx
        return df_fracs

    def calcKCHeuristicAllocation(self, kelly_level: float = 1):
        '''
        Calculates the allocation fractions using a simple max leverage
        heuristic for the Kelly Criterion.
        kelly_level: allows setting to full kelly (1) half-kelly (0.5)
                     or any other multiple. This takes the solution and
                     scales it down accordingly to reduce actual
                     leverage.
        '''
        df_fracs = self.calcKCUnconstrainedAllocation()
        heur_fracs = df_fracs.apply(
            lambda x: kelly_level * self.max_leverage * np.abs(x) / \
                      np.abs(x).sum() * np.sign(x),
            axis=1)
        heur_fracs = heur_fracs.rename(
            columns={'unconstrained_fracs': 'heuristic_fracs'})
        return heur_fracs

    def calcKCQuadProdAllocation(self, kelly_level: float = 1):
        '''
        Calculates optimal allocation fractions by solving a quadratic
        program according to the Kelly Criterion.

        kelly_level: allows setting to full kelly (1) half-kelly (0.5)
                     or any other multiple. This takes the solution from
                     the QP and scales it down accordingly to reduce
                     actual leverage.
        '''
        fracs = np.zeros((len(self.data), len(self.tickers)))
        fracs[:] = np.nan
        g = fracs[:, 0].copy()
        for i, (ts, row) in enumerate(self.data.iterrows()):
            if i < self.lookback:
                continue
            means = row.loc[(slice(None)), "mean"].values
            cov = self.cov[i]
            model = buildKCOptModel(means, cov, self.rfr,
                                    self.max_leverage)
            results = SolverFactory(
                'ipopt', executable="C:\\msys64\\home\\domin\\Ipopt\\src\\Apps\\AmplSolver\\ipopt.exe").solve(model)
            fracs[i] = np.array([model.f[j].value * kelly_level
                                 for j in model.f])
            g[i] = model.objective.expr()
        df_fracs = pd.DataFrame(fracs, index=self.data.index)
        midx = pd.MultiIndex.from_arrays(
            [self.tickers, len(self.tickers) * ['qp_fracs']])
        df_fracs.columns = midx
        return df_fracs

    def calcEqualAllocation(self):
        '''
        Rebalance so that the portfolio maintains a constant, equal
        allocation among each of the assets.
        '''
        fracs = np.ones((len(self.data), len(self.tickers))) / \
                len(self.tickers)
        fracs[:self.lookback] = np.nan
        df_fracs = pd.DataFrame(fracs, index=self.data.index)
        midx = pd.MultiIndex.from_arrays(
            [self.tickers, len(self.tickers) * ['eq_fracs']])
        df_fracs.columns = midx
        return df_fracs


# Initialize
opt = OptimalAllocation(['SPY', 'XLE', 'GLD', 'IEF'], max_leverage=1, lookback=252, rfr=.02)

# Calculate optimal allocations
uc_fracs = opt.calcKCUnconstrainedAllocation()
heur_fracs = opt.calcKCHeuristicAllocation(kelly_level=1)
qp_fracs = opt.calcKCQuadProdAllocation(kelly_level=1)
eq_fracs = opt.calcEqualAllocation()

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
labels = opt.tickers

fig, ax = plt.subplots(3, figsize=(12, 8), sharex=True)

ax[0].plot(uc_fracs * 100)
ax[0].set_title('Unconstrained Allocation')
ax[0].semilogy()

ax[1].plot(heur_fracs * 100)
ax[1].set_ylabel('Portfolio Allocation (%)')
ax[1].set_title('Heuristic Allocation')

ax[2].plot(qp_fracs * 100)
ax[2].set_xlabel('Date')
ax[2].set_title('Optimal Allocation')
ax[2].legend(labels=labels, ncol=len(labels),
             bbox_to_anchor=(0.68, -0.3))

plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------------------------------
# Apply kelly criterion to a Trading Strategy - simple SMA strategy

def getStratStats(log_returns: pd.Series, risk_free_rate: float = 0.02):
    stats = {}  # Total Returns
    stats['tot_returns'] = np.exp(log_returns.sum()) - 1

    # Mean Annual Returns
    stats['annual_returns'] = np.exp(log_returns.mean() * 252) - 1

    # Annual Volatility
    stats['annual_volatility'] = log_returns.std() * np.sqrt(252)

    # Sortino Ratio
    annualized_downside = log_returns.loc[log_returns < 0].std() * \
                          np.sqrt(252)
    stats['sortino_ratio'] = (stats['annual_returns'] -
                              risk_free_rate) / annualized_downside

    # Sharpe Ratio
    stats['sharpe_ratio'] = (stats['annual_returns'] - risk_free_rate) / stats['annual_volatility']

    # Max Drawdown
    cum_returns = log_returns.cumsum() - 1
    peak = cum_returns.cummax()
    drawdown = peak - cum_returns
    max_idx = drawdown.argmax()
    stats['max_drawdown'] = 1 - np.exp(cum_returns[max_idx]) / np.exp(peak[max_idx])

    # Max Drawdown Duration
    strat_dd = drawdown[drawdown == 0]
    strat_dd_diff = strat_dd.index[1:] - strat_dd.index[:-1]
    strat_dd_days = strat_dd_diff.map(lambda x: x.days).values
    strat_dd_days = np.hstack([strat_dd_days,
                               (drawdown.index[-1] - strat_dd.index[-1]).days])
    stats['max_drawdown_duration'] = strat_dd_days.max()

    return stats


# Kelly money management for trading strategy
def KellySMACrossOver(data, SMA1=50, SMA2=200, r=0.01,
                      periods=252, max_leverage=None, rolling=True):
    '''
    Sizes a simple moving average cross-over strategy according
    to the Kelly Criterion.
    '''
    data['returns'] = data['Close'] / data['Close'].shift(1)
    data['log_returns'] = np.log(data['returns'])
    # Calculate positions
    data['SMA1'] = data['Close'].rolling(SMA1).mean()
    data['SMA2'] = data['Close'].rolling(SMA2).mean()
    data['position'] = np.nan
    data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, 0)
    data['position'] = data['position'].ffill().fillna(0)
    data['_strat_returns'] = data['position'].shift(1) * \
                             data['returns']
    data['_strat_log_returns'] = data['position'].shift(1) * \
                                 data['log_returns']
    # Calculate Kelly Factor using the strategy's returns
    kf = getKellyFactor(data['_strat_log_returns'], r,
                        max_leverage, periods, rolling)
    data['kelly_factor'] = kf

    cash = np.zeros(data.shape[0])
    equity = np.zeros(data.shape[0])
    portfolio = cash.copy()
    portfolio[0] = 1
    cash[0] = 1
    for i, _row in enumerate(data.iterrows()):
        row = _row[1]
        if np.isnan(kf[i]):
            portfolio[i] += portfolio[i - 1]
            cash[i] += cash[i - 1]
            continue

        portfolio[i] += cash[i - 1] * (1 + r) ** (1 / 252) + equity[i - 1] * row['returns']
        equity[i] += portfolio[i] * row['kelly_factor']
        cash[i] += portfolio[i] * (1 - row['kelly_factor'])

    data['cash'] = cash
    data['equity'] = equity
    data['portfolio'] = portfolio
    data['strat_returns'] = data['portfolio'] / data['portfolio'].shift(1)
    data['strat_log_returns'] = np.log(data['strat_returns'])
    data['strat_cum_returns'] = data['strat_log_returns'].cumsum()
    data['cum_returns'] = data['log_returns'].cumsum()
    return data


kelly_sma = KellySMACrossOver(data.copy(), max_leverage=3)

fig, ax = plt.subplots(2, figsize=(15, 8), sharex=True)

ax[0].plot(np.exp(kelly_sma['cum_returns']) * 100, label='Buy-and-Hold')
ax[0].plot(np.exp(kelly_sma['strat_cum_returns']) * 100, label='SMA-Kelly')
ax[0].plot(np.exp(kelly_sma['_strat_log_returns'].cumsum()) * 100, label='SMA')
ax[0].set_ylabel('Returns (%)')
ax[0].set_title('Moving Average Cross-Over Strategy with Kelly Sizing')
ax[0].legend()

ax[1].plot(kelly_sma['kelly_factor'])
ax[1].set_ylabel('Leverage')
ax[1].set_xlabel('Date')
ax[1].set_title('Kelly Factor')

plt.tight_layout()
plt.show()

sma_stats = pd.DataFrame(getStratStats(kelly_sma['log_returns']),
                         index=['Buy and Hold'])
sma_stats = pd.concat([sma_stats,
                       pd.DataFrame(getStratStats(kelly_sma['strat_log_returns']),
                                    index=['Kelly SMA Model'])])
sma_stats = pd.concat([sma_stats,
                       pd.DataFrame(getStratStats(kelly_sma['_strat_log_returns']),
                                    index=['SMA Model'])])
print(sma_stats)
