import statsmodels.api as sm
import numpy as np
import pandas as pd
from datetime import datetime
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from pathlib import Path
from statsmodels.tsa.regime_switching.tests.test_markov_regression import fedfunds
from statsmodels.tsa.regime_switching.tests.test_markov_regression import ogap, inf
from statsmodels.tsa.regime_switching.tests.test_markov_regression import areturns

# NBER recession data
usrec = pdr.DataReader('USREC', 'fred', start=datetime(1947, 1, 1), end=datetime(2021, 6, 17))

# Federal Funds Rate with Switching Intercept
dta_fedfunds = pd.Series(fedfunds, index=pd.date_range('1954-07-01', '2010-10-01', freq='QS'))

# Plot the data
dta_fedfunds.plot(title='Federal funds rate', figsize=(12, 3))
plt.show()

# Fit the model
# (a switching mean is the default of the MarkovRegession model)
mod_fedfunds = sm.tsa.MarkovRegression(dta_fedfunds, k_regimes=2)
res_fedfunds = mod_fedfunds.fit()
print(res_fedfunds.summary())

res_fedfunds.smoothed_marginal_probabilities[1].plot(
    title='Probability of being in the high regime', figsize=(12, 3));
plt.show()

print(res_fedfunds.expected_durations)
print("Low regime expected to persist for nearly: ", round(res_fedfunds.expected_durations[0] / 4, 1), "yrs")
print("High regime expected to persist for roughly: ", round(res_fedfunds.expected_durations[1] / 4, 1), "yrs")

# Federal funds rate with switching intercept and lagged dependent variable
mod_fedfunds2 = sm.tsa.MarkovRegression(dta_fedfunds.iloc[1:], k_regimes=2, exog=dta_fedfunds.iloc[:-1])
res_fedfunds2 = mod_fedfunds2.fit()
print(res_fedfunds2.summary())

res_fedfunds2.smoothed_marginal_probabilities[0].plot(
    title='Probability of being in the high regime', figsize=(12, 3))
plt.show()

print(res_fedfunds2.expected_durations)
print("Low regime expected to persist for nearly: ", round(res_fedfunds2.expected_durations[0] / 4, 1), "yrs")
print("High regime expected to persist for roughly: ", round(res_fedfunds2.expected_durations[1] / 4, 1), "yrs")

# Taylor Rule with 2 or 3 regimes
dta_ogap = pd.Series(ogap, index=pd.date_range('1954-07-01', '2010-10-01', freq='QS'))
dta_inf = pd.Series(inf, index=pd.date_range('1954-07-01', '2010-10-01', freq='QS'))

exog = pd.concat((dta_fedfunds.shift(), dta_ogap, dta_inf), axis=1).iloc[4:]

# Fit the 2-regime model
mod_fedfunds3 = sm.tsa.MarkovRegression(dta_fedfunds.iloc[4:], k_regimes=2, exog=exog)
res_fedfunds3 = mod_fedfunds3.fit()

# Fit the 3-regime model
np.random.seed(12345)
mod_fedfunds4 = sm.tsa.MarkovRegression(dta_fedfunds.iloc[4:], k_regimes=3, exog=exog)
res_fedfunds4 = mod_fedfunds4.fit(search_reps=20)
print(res_fedfunds3.summary())
print(res_fedfunds4.summary())

fig, axes = plt.subplots(3, figsize=(10, 7))
ax = axes[0]
ax.plot(res_fedfunds4.smoothed_marginal_probabilities[0])
ax.set(title='Smoothed probability of a low-interest rate regime')
ax = axes[1]
ax.plot(res_fedfunds4.smoothed_marginal_probabilities[1])
ax.set(title='Smoothed probability of a medium-interest rate regime')
ax = axes[2]
ax.plot(res_fedfunds4.smoothed_marginal_probabilities[2])
ax.set(title='Smoothed probability of a high-interest rate regime')
plt.show()

# SWITCHING VARIANCES
# Get the federal funds rate data
dta_areturns = pd.Series(areturns, index=pd.date_range('2004-05-04', '2014-5-03', freq='W'))

# Plot the data
dta_areturns.plot(title='Absolute returns, S&P500', figsize=(12, 3))
plt.show()

# Fit the model
mod_areturns = sm.tsa.MarkovRegression(
    dta_areturns.iloc[1:], k_regimes=2, exog=dta_areturns.iloc[:-1], switching_variance=True)
res_areturns = mod_areturns.fit()
print(res_areturns.summary())

res_areturns.smoothed_marginal_probabilities[0].plot(
    title='Probability of being in a low-variance regime', figsize=(12, 3));
plt.show()


# Our own data with Markov Switching Dynamic Regression
def perform_markov_regression(df, column):
    model = sm.tsa.MarkovRegression(df.iloc[1:, column], k_regimes=2, exog=df.iloc[:-1, column],
                                    switching_variance=True)
    res = model.fit()
    df = df.merge(res.smoothed_marginal_probabilities[0], on="Date", how="left")
    return df


DATA_SOURCE = Path("C:\\Users\\domin\\PycharmProjects\\Miscellaneous\\ASSETS.h5")
with pd.HDFStore(DATA_SOURCE, "r") as store:
    df = store.get("ASSET/data")

idx = pd.IndexSlice
df = df.loc[idx[:, "KO"], ["High", "Low", "Open", "Close", "Volume"]].droplevel(level=1)
print(df.isna().any())

df["Returns"] = df.Close.pct_change()
df.dropna(inplace=True)

df = perform_markov_regression(df, 5)

print(df)

plt.plot(df.iloc[:, -1])
plt.show()

# Markov Autoregressive Models
usrec = pdr.DataReader('USREC', 'fred', start=datetime(1947, 1, 1), end=datetime(2013, 4, 1))

# HAMILTONIAN (1989) SWITCHING MODEL OF GNP
# Get the RGNP data to replicate Hamilton
dta = pd.read_stata('https://www.stata-press.com/data/r14/rgnp.dta').iloc[1:]
dta.index = pd.DatetimeIndex(dta.date, freq='QS')
dta_hamilton = dta.rgnp

# Plot the data
dta_hamilton.plot(title='Growth rate of Real GNP', figsize=(12,3))
plt.show()

# Fit the model
mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
res_hamilton = mod_hamilton.fit()
print(res_hamilton.summary())

fig, axes = plt.subplots(2, figsize=(7,7))
ax = axes[0]
ax.plot(res_hamilton.filtered_marginal_probabilities[0])
ax.fill_between(usrec.index, 0, 1, where=usrec['USREC'].values, color='k', alpha=0.1)
ax.set_xlim(dta_hamilton.index[4], dta_hamilton.index[-1])
ax.set(title='Filtered probability of recession')
ax = axes[1]
ax.plot(res_hamilton.smoothed_marginal_probabilities[0])
ax.fill_between(usrec.index, 0, 1, where=usrec['USREC'].values, color='k', alpha=0.1)
ax.set_xlim(dta_hamilton.index[4], dta_hamilton.index[-1])
ax.set(title='Smoothed probability of recession')
plt.show()

print(res_hamilton.expected_durations)

# KIM, NELSON, AND STARTZ (1998) THREE-STATE VARIANCE SWITCHING
ew_excs = requests.get('http://econ.korea.ac.kr/~cjkim/MARKOV/data/ew_excs.prn').content
raw = pd.read_table(BytesIO(ew_excs), header=None, skipfooter=1, engine='python')
raw.index = pd.date_range('1926-01-01', '1995-12-01', freq='MS')
dta_kns = raw.loc[:'1986'] - raw.loc[:'1986'].mean()

# Plot the dataset
dta_kns[0].plot(title='Excess returns', figsize=(12, 3))
plt.show()

# Fit the model
mod_kns = sm.tsa.MarkovRegression(dta_kns, k_regimes=3, trend='nc', switching_variance=True)
res_kns = mod_kns.fit()
print(res_kns.summary())

fig, axes = plt.subplots(3, figsize=(10,7))
ax = axes[0]
ax.plot(res_kns.smoothed_marginal_probabilities[0])
ax.set(title='Smoothed probability of a low-variance regime for stock returns')
ax = axes[1]
ax.plot(res_kns.smoothed_marginal_probabilities[1])
ax.set(title='Smoothed probability of a medium-variance regime for stock returns')
ax = axes[2]
ax.plot(res_kns.smoothed_marginal_probabilities[2])
ax.set(title='Smoothed probability of a high-variance regime for stock returns')
plt.show()

# FILARDO (1994) TIME-VARYING TRANSITION PROBABILITIES
# Get the dataset
filardo = requests.get('http://econ.korea.ac.kr/~cjkim/MARKOV/data/filardo.prn').content
dta_filardo = pd.read_table(BytesIO(filardo), sep=' +', header=None, skipfooter=1, engine='python')
dta_filardo.columns = ['month', 'ip', 'leading']
dta_filardo.index = pd.date_range('1948-01-01', '1991-04-01', freq='MS')

dta_filardo['dlip'] = np.log(dta_filardo['ip']).diff()*100
# Deflated pre-1960 observations by ratio of std. deviations
std_ratio = dta_filardo['dlip']['1960-01-01':].std() / dta_filardo['dlip'][:'1959-12-01'].std()
dta_filardo['dlip'][:'1959-12-01'] = dta_filardo['dlip'][:'1959-12-01'] * std_ratio

dta_filardo['dlleading'] = np.log(dta_filardo['leading']).diff()*100
dta_filardo['dmdlleading'] = dta_filardo['dlleading'] - dta_filardo['dlleading'].mean()

# Plot the data
dta_filardo['dlip'].plot(title='Standardized growth rate of industrial production', figsize=(13,3))
dta_filardo['dmdlleading'].plot(title='Leading indicator', figsize=(13,3))
plt.show()

# Markov Autoregressive Model
mod_filardo = sm.tsa.MarkovAutoregression(
    dta_filardo.iloc[2:]['dlip'], k_regimes=2, order=4, switching_ar=False,
    exog_tvtp=sm.add_constant(dta_filardo.iloc[1:-1]['dmdlleading']))

np.random.seed(23)
res_filardo = mod_filardo.fit(search_reps=20)
print(res_filardo.summary())

fig, ax = plt.subplots(figsize=(12,3))
ax.plot(res_filardo.smoothed_marginal_probabilities[0])
ax.fill_between(usrec.index, 0, 1, where=usrec['USREC'].values, color='gray', alpha=0.2)
ax.set_xlim(dta_filardo.index[6], dta_filardo.index[-1])
ax.set(title='Smoothed probability of a low-production state')
plt.show()

res_filardo.expected_durations[0].plot(
    title='Expected duration of a low-production state', figsize=(12,3))
plt.show()
