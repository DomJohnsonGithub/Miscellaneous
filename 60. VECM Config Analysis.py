import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as pdr
from datetime import datetime
import warnings
import yfinance as yf

warnings.filterwarnings("ignore")

"""
- Cointegration describes a long-term relationship between two (or more) asset prices.
- Cointegration can be viewed as a measure of similarity of assets in terms of risk exposure profiles.
- The prices of cointegrated assets are tethered due to the stationarity of the spread.

- Correlation has no well-defined relationship with cointegration. Cointegrated series might have low 
  correlation, and highly correlated series might not be cointegrated at all.
- Correlation describes a short-term relationship between the returns.
- Cointegration describes a long-term relationship between the prices. Do not use correlation for prices!

It comes naturally that log prices fit this description better, for the difference of log prices is directly 
log returns, but the difference of raw prices is not percentage returns yet. However, according to (Alexander, 2002), 
“Since it is normally the case that log prices will be cointegrated when the actual prices are cointegrated, 
it is standard, but not necessary, to perform the cointegration analysis on log prices.” So it is OK to analyze 
raw prices, but log prices are preferable.
"""

# ----- VECTOR ERROR CORRECTION MODEL ----- #
"""
Error correction model (ECM)is important in time-series analysis to better understand long-run dynamics. 
ECM can be derived from auto-regressive distributed lag model as long as there is a cointegration 
relationship between variables. In that context, each equation in the vector auto regressive (VAR) model 
is an autoregressive distributed lag model; therefore, it can be considered that the vector error correction 
model (VECM) is a VAR model with cointegration constraints.

Cointegration relations built into the specification so that it restricts the long-run behavior of the 
endogenous variables to converge to their cointegrating relationships while allowing for short-run adjustment 
dynamics. This is known as the error correction term since the deviation from long-run equilibrium is corrected 
gradually through a series of partial short-run adjustments.
"""
# ----- Get Data ----- #
# Get Index Data
indices = ["^IBEX", "^GSPC", "^HSI", "^RUT", "^IXIC", "^NSEI", "^GDAXI"]
master_df = yf.download(indices, start=datetime(2009, 1, 1),
                        end=datetime(2018, 1, 1))["Adj Close"].rename(
    columns={"Adj Close": "close"})

# Fill in missing value using interpolate() method
master_df = master_df.interpolate()
print(master_df)

# Visualize the Time Series for the Indices - note: how they all seem to follow a similar pattern
plt.style.use("dark_background")
fig, ax = plt.subplots(len(master_df.columns), 1, figsize=(16, 2.5), sharex=True)
for col, i in dict(zip(master_df.columns, list(range(7)))).items():
    master_df[col].plot(ax=ax[i], legend=True, lw=1., c="r", sharex=True)
fig.suptitle("Historical trends of levels variables", fontsize=12, fontweight="bold")
plt.show()

# ----- Check for Stationarity ----- #
# n.b: we know due to the stochastic nature of these time series that they are non-stationary,
# however, we will perform an ADF Test for completeness.

# ADF Test
from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(series, signif=0.05, name="", verbose=False):
    r = adfuller(series, autolag="AIC")
    output = {"test_statistic": round(r[0], 4), "p_value": round(r[1], 4), "n_lags": round(r[2], 4), "n_obs": r[3]}
    p_value = output["p_value"]

    def adjust(val, length=7):
        return str(val).ljust(length)

    # Print Summary
    print(f"     ADF Test on '{name}'", " \n  ", '_' * 47)
    print(f" Null H0: Data has a unit root. Non-stationary.")
    print(f" Significance Level     = {signif}")
    print(f" Test Statistic         = {output['test_statistic']}")
    print(f" No. lags chosen        = {output['n_lags']}")

    for key, val in r[4].items():
        print(f" Critical Value {adjust(key)} = {round(val, 3)}")

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak Evidence to Reject Null Hypothesis")
        print(f" => Series is Non-Stationary.")


for name, column in master_df.iteritems():
    adf_test(column, name=column.name)
    print()


# KPSS Test - test for stationarity if p = 0.01 (or test statistic > critical value) then cannot
# reject the null. meaning the series is NOT stationary
def kpss_test(series, h0_type="c"):
    indexes = ["Test Statistic", "p-value", "# of Lags"]
    kpss_test = kpss(series, regression=h0_type, nlags='auto')
    results = pd.Series(kpss_test[0:3], index=indexes)
    for key, value in kpss_test[3].items():
        results[f'Critical Value ({key})'] = value
    return results


print('KPSS-IBEX:')
print(kpss_test(series=master_df["^IBEX"]))
print('___________________')
print('KPSS-GSPC:')
print(kpss_test(series=master_df["^GSPC"]))
print('___________________')
print('KPSS-HSI:')
print(kpss_test(series=master_df["^HSI"]))
print('___________________')
print('KPSS-RUT:')
print(kpss_test(series=master_df["^RUT"]))
print('___________________')
print('KPSS-IXIC:')
print(kpss_test(series=master_df["^IXIC"]))
print('___________________')
print('KPSS-NSEI:')
print(kpss_test(series=master_df["^NSEI"]))
print('___________________')
print('KPSS-GDAXI:')
print(kpss_test(series=master_df["^GDAXI"]))

# ----- Normality Test ----- #
"""
To extract maximum information from our data, it is important to have a normal or Gaussian 
distribution of the data. To check for that, we have done a normality test based on the Null 
and Alternate Hypothesis intuition.

These distribution gives us some intuition about the normal distribution of our data. Value 
close to 0 for Kurtosis indicates a Normal Distribution where asymmetrical nature is signified 
by a value between -0.5 and +0.5 for skewness. The tails are heavier for kurtosis greater than 
0 and vice versa. Moderate skewness refers to the value between -1 and -0.5 or 0.5 and 1.
"""
from scipy.stats import normaltest, skew, kurtosis, probplot

for col, val in master_df.iteritems():
    stat, p = normaltest(val)
    print("Statistics=%.3f, p=%.3f" % (stat, p))
    alpha = .05
    if p > alpha:
        print(f" {col} data looks Gaussian (fail to reject H0)")
        print(f"Skewness: {skew(val)}, Kurtosis = {kurtosis(val)}\n")
    else:
        print(f"{col} data looks Non-Gaussian (reject H0)")
        print(f"Skewness: {skew(val)}, Kurtosis = {kurtosis(val)}\n")

import statsmodels.api as sm


def hist_qq(series, bins, nrows):
    fig, axes = plt.subplots(nrows, ncols=2, figsize=(15, 5))
    for (col, val), i in zip(series.iteritems(), range(nrows)):
        axes[i, 0].hist(val, bins, density=True)
        axes[i, 0].set_title(f"{col}")
        sm.qqplot(val, ax=axes[i, 1], line="r")
        axes[i, 1].set_title("Probability Plot")
    plt.show()


hist_qq(series=master_df, bins=50, nrows=7)

# ----- CORRELATION AND CAUSATION ----- #
"""
Though correlation helps us determine the degree of relationship between the variables, it 
does not tell us about the cause & effect of the relationship. A high degree of correlation 
does not always necessarily mean a relationship of cause & effect exists between variables. 
Here, in this context, it can be noted that, correlation does not imply causation, although 
the existence of causation always implies correlation.

Value > 0.5 is considerred correlated, > 0.8 is highly correlated

Doing both is interesting because if you have S > P, that means that you have a correlation 
that is monotonic but not linear. Since it is good to have linearity in statistics (it is easier) 
you can try to apply a transformation on y (such a log).

Some quick rules of thumb to decide on Spearman vs. Pearson:

- The assumptions of Pearson's are constant variance and linearity (or something reasonably close to that), 
  and if these are not met, it might be worth trying Spearman's.
- The example above is a corner case that only pops up if there is a handful (<5) of data points. If there 
  is >100 data points, and the data is linear or close to it, then Pearson will be very similar to Spearman.
- If you feel that linear regression is a suitable method to analyze your data, then the output of Pearson's 
  will match the sign and magnitude of a linear regression slope (if the variables are standardized).
- If your data has some non-linear components that linear regression won't pick up, then first try to straighten 
  out the data into a linear form by applying a transform (perhaps log e). If that doesn't work, then Spearman 
  may be appropriate.
- I always try Pearson's first, and if that doesn't work, then I try Spearman's.
"""
# Compute the correlation matrices
corr_pearson = master_df.copy().corr(method="pearson")
corr_spearman = master_df.copy().corr(method="spearman")

# gen. a mask for the upper traingle of the matricies
mask1 = np.zeros_like(corr_pearson, dtype=bool)
mask1[np.triu_indices_from(mask1)] = True

mask2 = np.zeros_like(corr_spearman, dtype=bool)
mask2[np.triu_indices_from(mask2)] = True

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
sns.heatmap(corr_pearson, annot=True, fmt=".4f", mask=mask1, center=0, square=True, linewidths=0.5, ax=axes[0])
axes[0].set_title("Pearson Correlation Coefficients")
sns.heatmap(corr_spearman, annot=True, fmt=".4f", mask=mask2, center=0, square=True, linewidths=0.5, ax=axes[1])
axes[1].set_title("Spearman Rank Correlation Coefficients")
plt.show()

# !!! - going to remove IBEX due to its lack of correlation with the other random variables

master_df = master_df.drop(columns=["^IBEX"])
print(master_df)

# ----- GRANGER CAUSALITY TEST ----- #
"""
The basis behind Vector Auto-Regression is that each of the time series in the system influences each other. 
This way, we can predict the series with past values of itself along with other series in the system. We will 
use Granger’s Causality Test to test this relationship before building the model.

Null hypothesis (H0) = coefficients of past values in the regression equation is zero.

Below, we are checking Granger Causality of all possible combinations of the series. The rows are the response 
variable, columns are predictors. The values in the table are the P-Values.
"""
from statsmodels.tsa.stattools import grangercausalitytests

max_lag = 6
test = "ssr_chi2test"


def granger_causation_matrix(data, variables, test="ssr_chi2test", verbose=False):
    X = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c, r in zip(X.columns, X.index):
        test_result = grangercausalitytests(data[[r, c]], maxlag=max_lag, verbose=False)
        p_values = [round(test_result[i + 1][0][test][1], 5) for i in range(max_lag)]
        if verbose: print(f"Y = {r}, X = {c}, P Values = {p_values}")
        min_p_value = np.min(p_values)
        X.loc[r, c] = min_p_value
    X.columns = [v + "-x axis" for v in variables]
    X.index = [v + "-y axis" for v in variables]
    return X


print("--------------------------")
print("Granger Causation Matrix: ")
gcm = granger_causation_matrix(data=master_df, variables=master_df.columns)
sns.heatmap(gcm, annot=True, fmt=".5")
plt.show()
print(gcm)
print("--------------------------")

"""
P value is less than the significant level of 5%, which indicates the need to accept the null hypothesis, 
namely the existence of Granger cause.

We have seen earlier that all the series are unit root non-stationary, they may be co-integrated. This 
extension of unit root concept to multiple time series means that a liner combination of two or more 
series is stationary and hence, mean reverting. VAR model is not equipped to handle this case without 
differencing. So, we will use here Vector Error Correction Model (VECM). We will explore here cointegration 
because it can be leveraged for trading strategy.

However, the concept of an integrated multivariate series is complicated by the fact that, all the component 
series of the process may be individually integrated but the process is not jointly integrated in the sense 
that one or more linear combinations of the series exist that produce a new stationary series. To simplify, 
a combination of two co-integrated series has a stable mean to which this linear combination reverts. A 
multivariate series with this characteristics is said to be cointegrated.
"""

# ----- Test for Co-integration ----- #
"""
Here, all the series are unit root non-stationary, they may be co-integrated. This extension of unit root 
concept to multiple time series means that a liner combination of two or more series is stationary and hence, 
mean reverting. The first hypothesis, tests for the presence of cointegration.

When we are dealing with models based on nonstationary variables, we normally difference I(1) data and using 
OLS we create a dynamic mode. But, in this process long-term relationship is lost from the data. Dependencies 
between non-stationary variables which are sometimes stable in time are called co-integration relationships. 
There is a mechanism that brings the system back to equilibrium every time it is shocked away from it (Granger 
theorem).

Here, we are testing the order of integration using Johansen’s procedure. Let us determine the lag value by 
fitting a VECM model and passing a maximum lag as 10.
"""
from statsmodels.tsa.vector_ar.vecm import VECM, select_order
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.vecm import select_coint_rank
from statsmodels.tsa.vector_ar.vecm import CointRankResults

nobs = 21
train_ecm, test_ecm = master_df[0:-nobs], master_df[-nobs:]
print("Train ECM shape:", train_ecm.shape, "Test ECM shape:", test_ecm.shape)

# ORDER SELECTION
"""
Rule-of-thumb formula for maximum lag length:

(4 * (T/100))^(0.25)

where T sample size (Schwert, 2002)
"""
# ----- VECM Model Fitting ----- #
model = select_order(train_ecm, maxlags=5)
print(model.summary())

"""
Above AIC and BIC are both penalized-likelihood criteria. BIC penalizes model complexity more heavily. 
At large number of instances, AIC tends to pick somewhat larger models than BIC. Comparatively, BIC 
penalizes the number of parameters in the model to a greater extent than AIC. We will consider BIC 
(1: 41.10) for our use case.
"""

# ----- Johansen co-integration on level data ----- #
"""
Johansen test assesses the validity of a cointegrating relationship, 
using a maximum likelihood estimates (MLE) approach.

Two types of Johansen’s test:
- one uses trace (from linear algebra),
- the other a maximum eigenvalue approach (an eigenvalue is a special scalar; when we multiply a matrix 
  by a vector and get the same vector as an answer, along with a new scalar, the scalar is called an eigenvalue).
- Both forms of the test will determine if cointegration is present. The hypothesis is stated as:

Johansen Cointegration Test releases two statistics — Trace Statistic (from linear algebra) and Max-Eigen Statistic 
(an eigenvalue is a special scalar; when we multiply a matrix by a vector and get the same vector as an answer, 
along with a new scalar, the scalar is called an eigenvalue).

- H0 for both: no cointegrating equations.
- The difference is in the alternate hypothesis (H1): the trace test alternate hypothesis is simply that the 
  number of cointegrating relationships is at least one (shown by the number of linear combinations).
- Rejecting the H0 is basically stating there is only one combination of the non-stationary variables that
  gives a stationary process.
"""
"""definition of det_orderint:
-1 - no deterministic terms; 0 - constant term; 1 - linear trend"""
model = coint_johansen(endog=train_ecm, det_order=1, k_ar_diff=1)
print("Eigen Statistic: ")
print(model.eig)
print()
print("Critical Values: ")
d = pd.DataFrame(model.cvt)
d.rename(columns={0: "90%", 1: "95%", 2: "99%"}, inplace=True)
print(d);
print()
print("Trace Statistic: ")
print(pd.DataFrame(model.lr1))

"""
TS = Trace Statistic, CV = Critical Value
1. TS (118.12) > CV (116.98) @ 99% level
2. TS (79.88) < CV (87.775) @ 99% level
3. TS (47.70) < CV (55.25) @ 95% level
4. TS (22.66) < CV (35.01) @ 95% level
5. TS (11.27) < CV (18.40) @ 95% level
6. TS (1.98) < CV (3.8415) @ 95% level

- 1 is greater than all critical levels and 2 is only less than
  than 99% confidence level. 

So, we have a strong evidence to reject the H0 of no cointegration and H1 of cointegration 
exists are accepted. This makes it a good candidate for error correction model.

We can safely assume that, the series in question are related and therefore can be combined in 
a linear fashion. If there are shocks in the short run, which may affect movement in the individual 
series, they would converge with time (in the long run). We can estimate both long-run and short-run 
models here. The series are moving together in such a way that their linear combination results in a 
stationary time series and sharing an underlying common stochastic trend.

So, we see here that, cointegration analysis demonstrates that, the series is question do have long-run 
equilibrium relationships, but, in the short term, the series are in disequilibrium. The short-term 
imbalance and dynamic structure can be expressed as VECM.
"""

# ----- ECM Correction Model (ECM) ----- #
"""
ECM shows the long-run equilibrium relationships of variables by inducing a short-run dynamic adjustment 
mechanism that describes how variables adjust when they are out of equilibrium.

Let us identify the cointegration rank.
"""
rank1 = select_coint_rank(train_ecm, det_order=1, k_ar_diff=1, method="trace", signif=0.01)
print(rank1.summary())

"""
- 1st column in the table shows the rank which is the number of cointegrating relationships for 
  the dataset, while the 2nd reports the number of equations in total.
- λ trace statistics in the 3rd column, together with the corresponding critical values.
- In 1st row, we see that test statistic > critical values, so the null of at most no cointegrating vector is rejected.
- However, test statistic (79.88) at 2nd row does not exceed the critical value (87.77), so the null of at most 
  one cointegrating vector cannot be rejected.
"""
"""
Below test statistic on maximum eigen value:

Maximum-eigenvalue statistic assumes a given number of r cointegrating relations under the null hypothesis 
and tests this against the alternative that there are r + 1 cointegrating equations.
"""
rank2 = select_coint_rank(train_ecm, det_order=1, k_ar_diff=1, method="maxeig", signif=0.01)
print(rank2.summary())

# we see that the maxeig test stat is < CV on 1st row; therefore,
# so the null of at most one cointegrating vector cannot be rejected.

# ----- Model Fitting ----- #
# - now we have all the relevant parameters available for model fitting.
vecm = VECM(train_ecm, k_ar_diff=1, coint_rank=1, deterministic="ci")
"""
estimates the VECM on the price with 1 lag, and 1 cointegrating relationship
"""
vecm_fit = vecm.fit()
print(vecm_fit.summary())

# ----- Residual Auto-Correlation ------ #
# Durbin-Watson stat is a test for autocorrelation in the residuals from a regression analysis
# DW e (0, 4), DW = 2.0, no autocorrelation
from statsmodels.stats.stattools import durbin_watson

out = durbin_watson(vecm_fit.resid)
for col, val in zip(train_ecm.columns, out):
    print((col), ":", round(val, 2))

# ----- Impulse-response Function (IRF) ----- #
"""
In order to analyze dynamic effects of the model responding to certain shocks as well as how 
the effects are among the 6 variables, further analysis can be made through IRF, and the results 
for 20 periods can be obtained. IRF is adopted to reflect shock effect of a system on an internal 
variable.
"""
irf = vecm_fit.irf(periods=15)
# plt.style.use("ggplot")
irf.plot(orth=False)  # can set impulse arg to "name_of_index" to see for individual series
plt.show()

plt.style.use('ggplot')
irf.plot(impulse='^HSI')
plt.show()
"""
As shown the 2nd plot from top, after analysis of the effects of HANG SENG (^HSI) price shock, it is 
found that positive shock have some impact. HANG SENG INDEX prices decline after a positive shock, reach 
the lowest point in the 1st period, then rise slowly, reach the peak in the 2nd period, and then remain 
at a stable level. This suggests that positive shock of HANG SENG prices has considerable influence on its 
own increasing, and the considerable influence has relatively long sustained effectiveness. Here our period 
is in daily frequency.

Plot 1 (topmost) is the IRF diagram of S&P500 (^GSPC) changes caused by HANG SENG price shocks. As seen in 
the figure, the positive shock in the first period causes SP500 fluctuation and is the peak point. Then SP500 
quickly declines to the lowest point in the 2nd period, and after that returns to a stable condition. This shows 
that HANG SENG price shock can be shortly transferred to ^GSPC, and has relatively large impacts on SP500 in the 
short term, but SP500 becomes stable around 3rd or 4th period. HANG SENG price shock has the short-term promoting 
effect on SP500 fluctuation, and this effect tends to be gentle in the long term.
Likewise, we can plot and analyze all the variables against each one.
"""

# ----- Prediction ----- #
forecast, lower, upper = vecm_fit.predict(nobs, alpha=0.05)
print("Lower bounds of confidence intervals:")
print(pd.DataFrame(lower.round(2)))
print("\nPoint Forecasts:")
print(pd.DataFrame(forecast.round(2)))
print("\nUpper bounds of confidence intervals")
print(pd.DataFrame(upper.round(2)))

forecast = pd.DataFrame(forecast, index=test_ecm.index, columns=test_ecm.columns).rename(columns={
    "^GSPC": "S&P500_pred", "^HSI": "HANG_SENG_pred", "^RUT": "RUSSELL2000_pred", "^IXIC": "NASDAQ_pred",
    "^NSEI": "NIFTY50_pred", "^GDAXI": "DAX30_pred"
})
print(forecast)

# ----- Accuracy Metrics ----- #
from sklearn.metrics import mean_squared_error, mean_absolute_error

combine = pd.concat([test_ecm, forecast], axis=1)
pred = combine[["^GSPC", "S&P500_pred", "^HSI", "HANG_SENG_pred", "^RUT", "RUSSELL2000_pred", "^IXIC", "NASDAQ_pred",
                "^NSEI", "NIFTY50_pred", "^GDAXI", "DAX30_pred"]]

mae = mean_absolute_error(pred["^GSPC"], pred["S&P500_pred"])
mse = mean_squared_error(pred["^GSPC"], pred["S&P500_pred"])
rmse = np.sqrt(mse)

sum = pd.DataFrame(index=["MAE", "MSE", "RMSE"])
sum["Accuracy metrics:  S&P500"] = [mae, mse, rmse]

mae = mean_absolute_error(pred["^HSI"], pred["HANG_SENG_pred"])
mse = mean_squared_error(pred["^HSI"], pred["HANG_SENG_pred"])
rmse = np.sqrt(mse)
sum["HANG SENG"] = [mae, mse, rmse]

mae = mean_absolute_error(pred["^RUT"], pred["RUSSELL2000_pred"])
mse = mean_squared_error(pred["^RUT"], pred["RUSSELL2000_pred"])
rmse = np.sqrt(mse)
sum["RUSSELL 2000"] = [mae, mse, rmse]

mae = mean_absolute_error(pred["^IXIC"], pred["NASDAQ_pred"])
mse = mean_squared_error(pred["^IXIC"], pred["NASDAQ_pred"])
rmse = np.sqrt(mse)
sum["NASDAQ"] = [mae, mse, rmse]

mae = mean_absolute_error(pred["^NSEI"], pred["NIFTY50_pred"])
mse = mean_squared_error(pred["^NSEI"], pred["NIFTY50_pred"])
rmse = np.sqrt(mse)
sum["NIFTY 50"] = [mae, mse, rmse]

mae = mean_absolute_error(pred["^GDAXI"], pred["DAX30_pred"])
mse = mean_squared_error(pred["^GDAXI"], pred["DAX30_pred"])
rmse = np.sqrt(mse)
sum["DAX30"] = [mae, mse, rmse]

print(sum)

pred.to_excel("12_VECM_pred_actuals.xlsx")

"""
Key Takeaways:

Error correction model is a dynamic model in which the change of the variable in the current 
time period is related to the distance between its value in the previous period and its value 
in the long-run equilibrium. Cointegration relations built into the specification of ECM which 
is kind of a long-term relation between time-series and residuals although stationary may still 
have some short-term autocorrelation structure. ECM is used here to estimate a short-run dynamic 
relationship between cointegrated variables and their rate of adjustment to the long-run equilibrium 
relationship.

VECM enables to use non stationary data (but cointegrated) for interpretation. This helps retain the 
relevant information in the data which would otherwise get missed on differencing of the same.
"""
