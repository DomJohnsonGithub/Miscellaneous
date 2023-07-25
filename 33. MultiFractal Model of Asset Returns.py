import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mps
from itertools import chain
import pandas as pd
import seaborn as sns
import pandas_datareader.data as pdr
from datetime import datetime
import yfinance as yf
from fractalmarkets.mmar.brownian_motion import BrownianMotion
from fractalmarkets.mmar.brownian_motion_multifractal_time import BrownianMotionMultifractalTime
from scipy import interpolate
from fbm import MBM  # multi-fractional brownian motion
import math

sns.set_style("darkgrid")


# Define Random Walk generator
def random_walk(n, ipo):
    x, y = 0, ipo
    xpos = [0]
    ypos = [ipo]
    for i in range(1, n + 1):
        step = np.random.uniform(0, 1)
        if step < 0.5:
            x += 1
            y += 0.01
        if step > 0.5:
            x += 1
            y += -0.01
        xpos.append(x)
        ypos.append(y)
    return [xpos, ypos]


# Generate and plot a single RW, set ipo=$50
rw = random_walk(1000, ipo=50)
plt.plot(rw[0], rw[1], label="Random Walk")
plt.legend()
plt.show()

# Get multiple random walks and plot
rws = pd.DataFrame(np.array([i for i in chain(*[random_walk(1000, ipo=80) for i in range(10)])]).T).drop(columns=np.arange(0, 20, 2))
rws.columns = [f"RW_{i}" for i in range(0, len(rws.columns))]

fig = plt.figure()
for i in rws:
    plt.plot(rws[i], label=f"{i}")
plt.legend(loc="best")
plt.show()

# Testing the law of Large Numbers Theory
tail = rws.tail(1).squeeze().mean()
print("Avg of all the final values of each RW: ", tail)

# Model only gives -1% or +1%
# Another way to see how unrealistic this model can be is to look at daily changes
daily_change = rws.copy().pct_change().dropna()

plt.plot(daily_change["RW_0"])
plt.grid(True)
plt.show()

sns.distplot(daily_change["RW_0"], kde=True, rug=True, norm_hist=True, color="orange")
plt.title("Distribution of Daily Changes")
plt.show()

# Slight change with uniformity to generate random walk
def random_walk(n, ipo):
    x, y = 0, ipo
    xpos = [0]
    ypos = [ipo]
    for i in range(1, n + 1):
        step = np.random.uniform(0, 1)
        if step < 0.5:
            x += 1
            y += np.random.uniform(0, 0.02)
        if step > 0.5:
            x += 1
            y += np.random.uniform(-0.02, 0)
        xpos.append(x)
        ypos.append(y)
    return [xpos, ypos]


rws = pd.DataFrame(np.array([i for i in chain(*[random_walk(1000, ipo=80) for i in range(10)])]).T).drop(columns=np.arange(0, 20, 2))
rws.columns = [f"RW_{i}" for i in range(0, len(rws.columns))]

fig = plt.figure()
for i in rws:
    plt.plot(rws[i], label=f"{i}")
plt.legend(loc="best")
plt.show()

# The Daily variances now look more realistic
daily_change = rws.copy().pct_change().dropna()

plt.plot(daily_change["RW_0"])
plt.show()

sns.distplot(daily_change["RW_0"], kde=True, rug=True, norm_hist=True, color="orange")
plt.title("Distribution of Daily Changes")
plt.show()

# Daily changes are variable
def random_walk(n, ipo):
    x, y = 0, ipo
    xpos = [0]
    ypos = [ipo]
    for i in range(1, n + 1):
        step = np.random.normal(0, 1)
        if step < 0.5:
            x += 1
            y += np.random.normal(0, 0.01)
        if step > 0.5:
            x += 1
            y -= np.random.normal(0, 0.01)
        xpos.append(x)
        ypos.append(y)
    return [xpos, ypos]


rws = pd.DataFrame(np.array([i for i in chain(*[random_walk(1000, ipo=80) for i in range(10)])]).T).drop(columns=np.arange(0, 20, 2))
rws.columns = [f"RW_{i}" for i in range(0, len(rws.columns))]

for i in rws:
    plt.plot(rws[i], label=f"{i}")
plt.legend(loc="best")
plt.show()

# The Daily variances now look more realistic
daily_change = rws.copy().pct_change().dropna()

plt.plot(daily_change["RW_0"])
plt.show()

sns.distplot(daily_change["RW_0"], kde=True, rug=True, norm_hist=True, color="orange")
plt.title("Distribution of Daily Changes")
plt.show()

# GEOMETRIC BROWNIAN MOTION
start_date = datetime(2019, 12, 1)
end_date = datetime(2019, 12, 31)
pred_end_date = datetime(2020, 1, 31)  # date for which we want to predict up until

alibaba = yf.download("BABA", start=start_date, end=end_date)["Close"].dropna()

# number of trading days between 1 Jan 2020 and 31 Jan 2020 for T variable
n_of_weekdays = pd.date_range(start=pd.to_datetime(end_date,
                                                 format="%Y-%m-%d") + pd.Timedelta('1 days'),
                            end=pd.to_datetime(pred_end_date,
                                               format="%Y-%m-%d")).to_series(
).map(lambda x:
      1 if x.isoweekday() in range(1, 6) else 0).sum()

So = alibaba[-1]  # this is the last date of our alibaba close price series
dt = 1  # time increment, daily, e.g. dt=0.3 would equal data published every 7.5 hours, as 0.3 = days
T = n_of_weekdays  # trading days, a month (depending on month), prediction horizon # The time unit for these two parameters has to be the same: dt & T

N = T / dt  # number of time points in the prediction time horizon. Here time increment is 1 day and we will get predictions for 22 trading days
# This means, we have 22 different time points(days) and we will have 22 predictions at the end.
t = np.arange(1, int(N) + 1)  # time progression in our model

returns = ((alibaba - alibaba.shift(1)) / alibaba.shift(1)).dropna()
mu = np.mean(returns)  # mean
sigma = np.std(returns)  # standard deviation

# b - prediction time point
scen_size = 10
b = {str(scen): np.random.normal(0, 1, int(N)) for scen in range(1, scen_size + 1)}

# W - brownian motion path
W = {str(scen): b[str(scen)].cumsum() for scen in range(1, scen_size + 1)}

# Components of GBM: Drift and Diffusion
drift = (mu - 0.5 * sigma ** 2) * t
diffusion = {str(scen): sigma * W[str(scen)] for scen in range(1, scen_size + 1)}

S = np.array([So * np.exp(drift + diffusion[str(scen)]) for scen in range(1, scen_size + 1)])
S = np.hstack((np.array([[So] for scen in range(scen_size)]), S))

# Plot the predictions of daily price for Jan 2020
baba_jan = yf.download("BABA", start=end_date, end=pred_end_date)["Close"]
print(baba_jan)

for i in range(scen_size):
    plt.title("Daily Volatility: " + str(sigma))
    plt.plot(pd.date_range(start=alibaba.index.max(),
                           end=pred_end_date, freq='D').map(lambda x:
                                                            x if x.isoweekday() in range(1, 6) else np.nan).dropna(),
             S[i, :], label="GBM Predicted Price Path")
    plt.ylabel('Stock Prices, €')
    plt.xlabel('Prediction Days')
plt.plot(baba_jan, linestyle="--", lw=3., label="Actual Close Price")
plt.legend(loc="upper left")
plt.show()

# Dataframe format for predictions - first 10 scenarios only
preds_df = pd.DataFrame(S.swapaxes(0, 1)[:, :10]).set_index(
    pd.date_range(start=alibaba.index.max(),
                  end=pred_end_date, freq='D').map(lambda x:
                                                   x if x.isoweekday() in range(1, 6) else np.nan).dropna()
).reset_index(drop=False)

# Viz the daily changes of a GBM Path and the distribution to look for normality (Gaussian Series)
daily_change = (preds_df.iloc[1:, 7].copy() / preds_df.iloc[1:, 7].copy().shift(1) - 1).dropna()

# sanity checks
plt.plot(daily_change)
plt.show()

sns.distplot(daily_change, kde=True, rug=True, norm_hist=True, color="orange")
plt.show()

# THE MULTI-FRACTAL MODEL OF ASSET RETURNS
# Brownian Motion
bm = BrownianMotion(9, .457, .603, randomize_segments=True)
data = bm.simulate()  # [ [x, y], ... , [x_n, y_n] ]
f = interpolate.interp1d(data[:, 0], data[:, 1])

y = f(np.arange(0, 1, 0.001))
x = np.linspace(0, 1, len(y), endpoint=True)

y_diff = [b - a for a, b in zip(y[:-1], y[1:])]

fig, axs = plt.subplots(2)
fig.suptitle('Brownian Motion')
axs[0].plot(x, y, 'b-')
axs[1].bar(x[:-1], y_diff, align='edge', width=0.001, alpha=0.5)
bar_list = filter(lambda x: isinstance(x, mps.Rectangle), axs[1].get_children())
for bar, ret in zip(bar_list, y_diff):
    if ret >= 0:
        bar.set_facecolor('green')
    else:
        bar.set_facecolor('red')

z1 = np.array(y)
z2 = np.array([0] * len(y))

axs[0].fill_between(x, y, 0,
                    where=(z1 >= z2),
                    alpha=0.30, color='green', interpolate=True)
axs[0].fill_between(x, y, 0,
                    where=(z1 < z2),
                    alpha=0.30, color='red', interpolate=True)
plt.show()

# Brownian Motion in MultiFractal Time
bmmt = BrownianMotionMultifractalTime(9, x=0.457, y=0.603, randomize_segments=True, randomize_time=True, M=[0.6, 0.4])
data = bmmt.simulate()  # [ [x, y], ..., [x_n, y_n]]

f = interpolate.interp1d(data[:, 0], data[:, 1])

y = f(np.arange(0, 1, .001))
x = np.linspace(0, 1, len(y), endpoint=True)

y_diff = [b - a for a, b in zip(y[:-1], y[1:])]

fig, axs = plt.subplots(2)
fig.suptitle('Brownian Motion in Multi-fractal Time')
axs[0].plot(x, y, 'b-')
axs[1].bar(x[:-1], y_diff, align='edge', width=0.001, alpha=0.5)
bar_list = filter(lambda x: isinstance(x, mps.Rectangle), axs[1].get_children())
for bar, ret in zip(bar_list, y_diff):
    if ret >= 0:
        bar.set_facecolor('green')
    else:
        bar.set_facecolor('red')

z1 = np.array(y)
z2 = np.array([0] * len(y))

axs[0].fill_between(x, y, 0,
                    where=(z1 >= z2),
                    alpha=0.30, color='green', interpolate=True)
axs[0].fill_between(x, y, 0,
                    where=(z1 < z2),
                    alpha=0.30, color='red', interpolate=True)
plt.show()


# Another package for generation of multi-fractional Brownian motion.
# Riemann–Liouville fractional integral representation of mBm.

# ex. Hurst fn wrt time
def h(t):
    return 0.25 * math.sin(20 * t) + 0.5


m = MBM(n=1024, hurst=h, length=1, method="riemannliouville")

# generate a mBm realization
mbm_sample = m.mbm()
# generate a mGn realization
mgn_sample = m.mgn()
# get the time associated with the mBm
t_values = m.times()

mbm_df = pd.DataFrame(mbm_sample, index=pd.date_range(start=datetime(2010, 1, 1), periods=len(mbm_sample)), columns=["MBM"])

ipo = 50.00
mbm_df = mbm_df + ipo

plt.plot(mbm_df, label="MultiFractional Brownian Motion", c="r")
plt.legend(loc="best")
plt.show()

# # For one-off samples of mBm or mGn there are separate functions available:
# from fbm import mbm, mgn, times
# # define a hurst fn
# def hurst(t):
#     return 0.75 - 0.5*t
#
# # Generate a mbm realization
# mbm_sample = mbm(n=1024, hurst=h, length=1, method='riemannliouville')
# # Generate a fGn realization
# mgn_sample = mgn(n=1024, hurst=h, length=1, method='riemannliouville')
# # Get the times associated with the mBm
# t_values = times(n=1024, length=1)
