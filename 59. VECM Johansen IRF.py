import os
import sys
import warnings
from datetime import date
import pandas as pd
import pandas_datareader.data as web
import numpy as np
from datetime import datetime
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.gridspec as gridspec
import seaborn as sns
import yfinance as yf
from scipy.stats import norm, normaltest, skew, kurtosis
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.api import VAR, VARMAX
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import probplot, moment
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')
import statsmodels.tsa.vector_ar.vecm as vecm

# ----- VECM Estimation and Analysis ----- #
"""
VECM imposes additional restriction due to the existence of non-stationary but co-integrated data forms. 
It utilizes the co-integration restriction information into its specifications. After the cointegration 
is known then the next test process is done by using error correction method. Through VECM we can interpret 
long term and short term equations. We need to determine the number of co-integrating relationships. The 
advantage of VECM over VAR is that the resulting VAR from VECM representation has more efficient coefficient 
estimates.

Good examples:
-Stocks that belong to the same sector.
-WTI crude oil and Brent crude oil.
-AUD/USD and NZD/USD.
-Yield curves and futures calendar spreads (Alexander, 2002).

I got data from Yahoo Finance for the closing prices of the FANG stocks going back 5 years. This means the 
data set lacks data for how these companies' stocks acted in a recession. Regardless, I proceeded with my 
analysis. I tested for degree of integration, Granger causality, and Johansen integration. Based on the 
results, I proceeded with vector error correction modeling (VECM), which is a technique for simultaneously 
estimating multiple time series with at least 1 cointegrated relationship.
"""

# ----- Get Data (TS) ----- #
start, end = datetime(2014, 1, 1), datetime(2019, 9, 9)
tickers = ["NVDA", "AMZN", "NFLX", "GOOG"]
df = yf.download(tickers, start, end).squeeze().drop(
    columns=["High", "Open", "Low", "Close", "Volume"]).rename(columns={
    "Adj Close": "close"})
df = df.stack(level=1).swaplevel(i=0, j=1)
idx = pd.IndexSlice

df = [df.loc[idx[i, :], :].droplevel(0) for i in tickers]
df = pd.concat([df[0], df[1], df[2], df[3]], axis=1)
df.columns = tickers
print(df)
# df is raw data

diff_df = df.copy().diff().dropna()  # diff data
dbldiff_df = df.copy().diff().diff().dropna()  # twice differenced data
logrets_df = np.log(df.copy()).diff().dropna()  # log returns (1 diff)

# ----- Data Properties ----- #
"""
Stationarity is where the mean an variance of a time series is not dependent on time. 
When graphed, it looks like white noise. Due to several problems caused by modeling 
non-stationary time series, chiefly autocorrelation, you need to check if a time-series 
is stationary prior to modeling. If it is not stationary, you need to get it there through differencing.
"""
# ACF Plots
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 8), dpi=120, sharex=True)
for i, j in zip(range(4), tickers):
    plot_acf(df.iloc[1:, i], ax=axes[i, 0], lags=50)
    axes[i, 0].set_title("ACF for %s" % j)
for i, j in zip(range(4), tickers):
    plot_acf(diff_df.iloc[1:, i], ax=axes[i, 1], lags=50)
    axes[i, 1].set_title("ACF for %s" % j)

for i, j in zip(range(4), tickers):
    plot_acf(dbldiff_df.iloc[:, i], ax=axes[i, 2], lags=50)
    axes[i, 2].set_title("ACF for %s" % j)

for i, j in zip(range(4), tickers):
    plot_acf(logrets_df.iloc[1:, i], ax=axes[i, 3], lags=50)
    axes[i, 3].set_title("ACF for %s" % j)
plt.suptitle("ACF for Original Series, Diff Series, Double Diff Series, Log Returns")
plt.tight_layout()
plt.show()

# ADF Tests
df = df.iloc[2:, :]
diff_df = diff_df.iloc[1:, :]
dbldiff_df = dbldiff_df.iloc[:, :]
logrets_df = logrets_df.iloc[1:, :]

print("----------------------------------------------------")
print("ADF TEST FOR ORIGINAL SERIES:")
for i in tickers:
    for j in ["nc", "c", "ct"]:
        result = adfuller(df.loc[:, i], regression=j)
        print('ADF Statistic with %s for %s: %f' % (j, i, result[0]),
              'p-value: %f' % result[1])
print("----------------------------------------------------")
print("ADF TEST FOR DIFFERENCED SERIES:")
for i in tickers:
    for j in ["nc", "c", "ct"]:
        result = adfuller(diff_df.loc[:, i], regression=j)
        print('ADF Statistic with %s for %s: %f' % (j, i, result[0]),
              'p-value: %f' % result[1])
print("----------------------------------------------------")
print("ADF TEST FOR SECOND DIFFERENCED SERIES:")
for i in tickers:
    for j in ["nc", "c", "ct"]:
        result = adfuller(dbldiff_df.loc[:, i], regression=j)
        print('ADF Statistic with %s for %s: %f' % (j, i, result[0]),
              'p-value: %f' % result[1])
print("----------------------------------------------------")
print("ADF TEST FOR LOG RETURNS:")
for i in tickers:
    for j in ["nc", "c", "ct"]:
        result = adfuller(logrets_df.loc[:, i], regression=j)
        print('ADF Statistic with %s for %s: %f' % (j, i, result[0]),
              'p-value: %f' % result[1])
print("----------------------------------------------------")

# Histogram Plots and Density
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 8), dpi=120, sharex=True)
for i, j in zip(range(4), tickers):
    sns.distplot(a = df.iloc[:, i], ax=axes[i, 0], bins=50, kde=True, hist=True, rug=True, fit=norm, color="orange",
                 kde_kws={"color": "red", "lw": 1, "label": "(KDE)"})
    axes[i, 0].set_title(j)
    axes[i, 0].legend(loc="best")
    axes[i, 0].grid(True)
for i, j in zip(range(4), tickers):
    sns.distplot(a=diff_df.iloc[:, i], ax=axes[i, 1], bins=50, kde=True, hist=True, rug=True, fit=norm,
                 color="cyan", kde_kws={"color": "green", "lw": 1, "label": "(KDE)"})
    axes[i, 1].set_title(j)
    axes[i, 1].legend(loc="best")
    axes[i, 1].grid(True)
for i, j in zip(range(4), tickers):
    sns.distplot(a=dbldiff_df.iloc[:, i], ax=axes[i, 2], bins=50, kde=True, hist=True, rug=True, fit=norm,
                 color="blue", kde_kws={"color": "purple", "lw": 1, "label": "(KDE)"})
    axes[i, 2].set_title(j)
    axes[i, 2].legend(loc="best")
    axes[i, 2].grid(True)
for i, j in zip(range(4), tickers):
    sns.distplot(a=logrets_df.iloc[:, i], ax=axes[i, 3], bins=50, kde=True, hist=True, rug=True, fit=norm,
                 color="b", kde_kws={"color": "crimson", "lw": 1, "label": "(KDE)"})
    axes[i, 3].set_title(j)
    axes[i, 3].legend(loc="best")
    axes[i, 3].grid(True)
plt.suptitle("Histogram plus Kernel Density Estimation(Raw Series, Diff Series, Dbl Diff Series, Log Returns")
plt.tight_layout()
plt.show()

# ----- GRANGER CAUSALITY ----- #
"""
Establishing causality in observational data is notoriously difficult. Granger causality is a lower bar. 
It simply says that if previous values of X can predict future values of y, then X Granger causes y. It 
is performed by estimating the regression of the lagged values of X on y and performing an F-test. If the 
p-value is small enough, you reject the null hypothesis that all the coefficients of the lagged values of 
X are 0. In plain English, small p-values say that the lagged Xs have predictive power on future y, with a 
corresponding level of confidence.

Better to perform on stationary data otherwise might result in spurious regression.

Where P<=0.1 & P<=0.05 is 10% and 5% significance level, can say that stocks lags has predictive power, in this
case a lag of AMZN seems to have predictive power on all other stocks. Given the results of multi-directional 
Grange causality, VEC modeling seems a plausible choice.
"""
from itertools import permutations
stocks_perms = list(permutations(tickers, 2))
for i in range(len(stocks_perms)):
    temp_list = list(stocks_perms[i])
    temp_df = diff_df[temp_list]
    print("DOES A LAG OF " + temp_list[1] + " PREDICT " + temp_list[0])
    print(grangercausalitytests(temp_df, maxlag=1, addconst=True, verbose=True))
    print("")
    print("")

# ----- JOHANSEN COINTEGRATION ----- #
"""
For VEC modeling to be appropriate to model these stocks, Π (the vector of loading coefficients times the 
vector of error-correction coefficients that constitutes the error correction term in a VECM) times the 
column vector of dependent variables must be ~I(0) because all other terms on the right and left of the 
equation are ~I(0). For this to be true, Πx ⃑ = 0 must be true, where x ⃑ is a column vector of the dependent 
variables because change is 0 in equilibrium and there is equilibrium in the long-run. If the rank of Π is 0, 
then Π is the null matrix. If the rank is equal to the number of dependent variables, then x ⃑ = 0 must equal 
zero for the relationship to hold. In either case, there are no cointegrating vectors and there is no long-run 
relationship between the variables. If the rank of Π is greater than 0 and less than the number of dependent 
variables, there is a cointegrated relationship between the variables.

I used Johansen's Trace Test to establish cointegration. The null hypothesis is the rank of matrix Π is i and 
the alternative hypothesis is the rank of Π is equal to k, the number of endogenous variables, where i starts 
at 0 and proceeds sequentially to k.

The results of this test are that there is 1 cointegrating relationship between the FANG stocks at 1, 3, and 4 
lags during the sample period at the 95% significance level. VEC modeling is appropriate to modeling these stocks.
"""
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_trace(y, p):
    N, l = y.shape
    joh_trace = coint_johansen(y, 0, p)
    r = 0
    for i in range(l):
        if joh_trace.lr1[i] > joh_trace.cvt[i, 1]:  # 0: 90%, 1: 95%, 2: 99%
            r = i + 1
    joh_trace.r = r
    return joh_trace

# loop through all 1 to 10 lags of trading days
for i in range(1, 11):
    # tests for cointegration at i lags
    joh_trace = johansen_trace(df[tickers], i)
    print("Using the Trace Test, there are", joh_trace.r,
          "cointegrating vectors at %s lags between the FANG Stocks" % i)
    print("")

# - The results of this test are that there is 1 cointegrating relationship between the FANG
#   stocks at 1, 2, 3, 4, 5, 6 and 8 lags during the sample period at the 95% significance level. VEC
#   modeling is appropriate to modeling these stocks.

# ----- VECM ESTIMATION and ANALYSIS ----- #
"""
The loading coefficients (alphas) are the speed of adjustment to the long-run relationship. They are the percentage of 
disequilibrium from the long-run equilibrium that disapates in one period. The alphas for the closing price of FB and 
NFLX are not statistically significant. If 0.05 <= p <= 0.1 then significance of the alpha would be ambiguous as sig 
at 10% level but not the 5%. AMZN p value is statistically significant. The alpha for GOOGL is statistically significant 
at the 0.05 significance level and estimated at 0.0363.

This information means FB and NFLX (and possibly any ones with 0.05 <= p_value <= 0.1) are weakly exogenous to AMZN & GOOGL. 
Weak exogeneity is the concept that deviations from the long-run do not directly affect the weakly exogenous variable. The 
effect comes from the subsequent lags from the non-weakly exogenous variables. The lags of GOOGL and AMZN are the drivers 
of the return to the long-run equilibrium in the weakly exogenous variables.

The beta coefficients are the actual long-run relationship coefficients. The beta for FB is standardized at 1 for ease of 
interpretation of the other beta coefficients. The beta for GOOGL is -0.5516, which means a 1 dollar increase in GOOGL, 
leads to a 0.5516 dollar decrease in the closing price of FB in the long-run and 3.63% of this correction occurs within a 
day. Rearranging which beta is standardized is how you tell how the other stocks affect the stock with the standardized beta.
"""
from statsmodels.tsa.vector_ar.vecm import VECM
# estimates the VECM on the closing prices with 4 lags, 1 cointegrating relationship, and
# a constant within the cointegration relationship
fang_vecm = VECM(endog=df, k_ar_diff=8, coint_rank=1, deterministic="ci")
fang_vecm_fit = fang_vecm.fit()
print(fang_vecm_fit.summary())

# ----- Impulse Response Functions ----- #
"""
Impulse Response Functions (IRF) show what happens to one variable when you shock another (or the same variable) with 
an increase of 1 in the previous period. The blue curve shows the effect of the unit shock as the shock becomes less 
and less recent. The dotted lines represent the 95% confidence interval for the IRF.

A 1 dollar shock to the closing price of GOOGL leads to a drop in the closing price of FB, but the effect of that shock 
goes to zero over time. The effect of a 1 dollar shock to NFLX initially increases GOOGL's closing price by about 5 cents, but then goes to approximately zero. A unit shock to FB however, persists as do some others like NFLX on AMZN. If all the IRFs were like those, the system would be dynamically unstable, but luckily 0 is within the 95% confidence level of many of our IRFs.
"""
# stores and prints the impulse response functions
irf = fang_vecm_fit.irf(100)
irf.plot(orth=False)
plt.show()

# Below is just a graph of the closing stock prices with dynamic forecasting out roughly half a year with error bands.
fang_vecm_fit.plot_forecast(180)
plt.show()