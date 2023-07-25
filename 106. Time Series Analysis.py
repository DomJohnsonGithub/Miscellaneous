import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from pathlib import Path
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import signal
from statsmodels.tsa.filters.bk_filter import bkfilter
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.filters.cf_filter import cffilter
from pandas.plotting import autocorrelation_plot
import pmdarima.arima as arima
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import lag_plot
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.stattools import grangercausalitytests

plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})
idx = pd.IndexSlice


def calculate_returns(df):
    """
    Calculates returns and log returns for a given DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the "close" column.

    Returns:
    - df (pandas.DataFrame): The cleaned DataFrame with additional "returns" and "log_returns" columns.
    """

    # Calculate returns
    df["returns"] = df["close"].diff(1)
    # The "returns" column represents the difference between consecutive "close" values.

    # Calculate log returns
    df["log_returns"] = np.log(df["close"]).diff()
    # The "log_returns" column represents the logarithmic difference between consecutive "close" values.
    # We use the numpy log function to calculate the natural logarithm of the "close" column.

    # Drop NaN values
    df.dropna(inplace=True, how="any")
    # Remove rows with NaN values from the DataFrame.
    # This is necessary because calculating returns and log returns creates NaN values for the first row.

    return df


def plot_data(df):
    """
    Plots price close, returns, and logarithmic returns data from a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the necessary columns.

    Returns:
    - None
    """

    # Creating subplots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)

    # Plotting price close
    ax1.plot(df["close"].to_numpy())
    ax1.set_title("Price Close")

    # Plotting returns
    ax2.plot(df["returns"].to_numpy())
    ax2.set_title("Returns")

    # Plotting logarithmic returns
    ax3.plot(df["log_returns"].to_numpy())
    ax3.set_title("Logarithmic Returns")

    # Plotting histogram for price close
    ax4.hist(df["close"].to_numpy(), bins="rice")
    ax4.set_title("Histogram of Price Close")

    # Plotting histogram for returns
    ax5.hist(df["returns"].to_numpy(), bins="rice")
    ax5.set_title("Histogram of Returns")

    # Plotting histogram for logarithmic returns
    ax6.hist(df["log_returns"].to_numpy(), bins="rice")
    ax6.set_title("Histogram of Logarithmic Returns")

    # Adjusting subplot layout
    plt.tight_layout()


def calculate_monthly_log_returns(df):
    """
    Calculates the monthly mean of logarithmic returns for a given DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the "log_returns" column.

    Returns:
    - monthly_log_returns (pandas.Series): The monthly mean of logarithmic returns as a Series.
    """

    # Drop the "ticker" level if present
    df = df.droplevel(level="ticker")
    # If the DataFrame has a multi-index with a "ticker" level, this step drops that level.

    # Resample the log_returns column to monthly frequency and calculate the mean
    monthly_log_returns = df["log_returns"].resample("M").mean()
    # This step resamples the "log_returns" column to a monthly frequency and calculates the mean for each month.

    # Convert the result to a pandas Series
    monthly_log_returns = pd.Series(monthly_log_returns)
    # The result is converted to a pandas Series for convenience and ease of use.

    return monthly_log_returns


def plot_monthly_log_returns(monthly_log_returns):
    """
    Plots the monthly log returns data.

    Parameters:
    - monthly_log_returns (pandas.Series): The monthly mean of logarithmic returns.

    Returns:
    - None
    """

    # Plotting the monthly log returns
    plt.plot(monthly_log_returns)
    # Plots the data contained in the monthly_log_returns Series.

    # Adding a title, x-axis label, and y-axis label
    plt.title("Monthly Log Returns")
    plt.xlabel("Dates")
    plt.ylabel("Monthly Log Returns")

    # Displaying gridlines
    plt.grid(True)

    # Adjusting subplot layout
    plt.tight_layout()


def plot_seasonal_close_price(data, years):
    """
    Plots the seasonal plot of Walmart monthly close price series.

    Parameters:
    - data (pandas.DataFrame): The DataFrame containing the data with columns 'month', 'year', and 'close'.
    - years (list): The list of years to include in the plot.

    Returns:
    - None
    """

    # Prep Colors
    np.random.seed(100)
    mycolors = np.random.choice(list(plt.cm.colors.XKCD_COLORS.keys()), len(years), replace=False)
    # Randomly select colors from the XKCD color set based on the number of years.

    # Draw Plot
    plt.figure(figsize=(16, 12), dpi=80)
    for i, y in enumerate(years):
        if i > 0:
            plt.plot("month", "close", data=data.loc[data.year == y, :],
                     color=mycolors[i], label=y)
            # Plot the close prices for each year.

            plt.text(data.loc[data.year == y, :].shape[0] - .9,
                     data.loc[data.year == y, 'close'][-1:].values[0],
                     y, fontsize=12, color=mycolors[i])
            # Add text annotation for the closing price of each year.

    # Decoration
    plt.gca().set(ylabel='$Monthly Close Price$', xlabel='$Month$')
    plt.yticks(fontsize=12, alpha=.7)
    plt.title("Seasonal Plot of Monthly Close Price Series", fontsize=20)


def plot_box_plots(data):
    """
    Plots year-wise and month-wise box plots for the close prices.

    Parameters:
    - data (pandas.DataFrame): The DataFrame containing the data with columns 'year', 'month', and 'close'.

    Returns:
    - None
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), dpi=80)
    sns.boxplot(x="year", y='close', data=data, ax=axes[0])
    # Plot year-wise box plot of close prices.

    sns.boxplot(x='month', y='close', data=data.loc[~data.year.isin([2000, 2019]), :], ax=axes[1])
    # Plot month-wise box plot of close prices, excluding the years 2000 and 2019.

    # Set Title
    axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18)
    axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)


def perform_seasonal_decomposition(data):
    """
    Performs multiplicative and additive seasonal decomposition on the close prices.

    Parameters:
    - data (pandas.DataFrame): The DataFrame containing the data with a 'close' column.

    Returns:
    - result_mul (statsmodels.tsa.seasonal.DecomposeResult): The result of multiplicative seasonal decomposition.
    - result_add (statsmodels.tsa.seasonal.DecomposeResult): The result of additive seasonal decomposition.
    """

    # Multiplicative Decomposition
    result_mul = seasonal_decompose(x=data["close"], model="multiplicative", extrapolate_trend="freq")
    # Perform multiplicative seasonal decomposition on the 'close' column.

    # Additive Decomposition
    result_add = seasonal_decompose(x=data["close"], model="additive", extrapolate_trend="freq")
    # Perform additive seasonal decomposition on the 'close' column.

    return result_mul, result_add


def plot_seasonal_decomposition(result_mul, result_add):
    """
    Plots the seasonal decomposition results for both multiplicative and additive models.

    Parameters:
    - result_mul (statsmodels.tsa.seasonal.DecomposeResult): The result of multiplicative seasonal decomposition.
    - result_add (statsmodels.tsa.seasonal.DecomposeResult): The result of additive seasonal decomposition.

    Returns:
    - None
    """

    plt.rcParams.update({'figure.figsize': (10, 10)})

    # Plotting Multiplicative Decomposition
    result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
    # Plot the components (trend, seasonal, and residual) of the multiplicative decomposition.

    # Plotting Additive Decomposition
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    # Plot the components (trend, seasonal, and residual) of the additive decomposition.

    plt.show()

    # Plotting Residuals
    fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2)
    ax1.plot(result_add.resid)
    ax1.set_title('Additive Decompose Residuals', fontsize=20)
    # Plot the residuals of the additive decomposition.

    ax2.plot(result_mul.resid)
    ax2.set_title('Multiplicative Decompose Residuals', fontsize=20)
    # Plot the residuals of the multiplicative decomposition.

    plt.show()


def knn_mean(ts, n):
    out = np.copy(ts)
    for i, val in enumerate(ts):
        if np.isnan(val):
            n_by_2 = np.ceil(n / 2)
            lower = np.max([0, int(i - n_by_2)])
            upper = np.min([len(ts) + 1, int(i + n_by_2)])
            ts_near = np.concatenate([ts[lower:i], ts[i:upper]])
            out[i] = np.nanmean(ts_near)
    return out


def seasonal_mean(ts, n, lr=0.7):
    """
    Compute the mean of the corresponding seasonal periods
    ts: 1D array-like of the time series
    n: Seasonal window length of the time series
    """
    out = np.copy(ts)
    for i, val in enumerate(ts):
        if np.isnan(val):
            ts_seas = ts[i - 1::-n]  # prev. seasons only
            if np.isnan(np.nanmean(ts_seas)):
                ts_seas = np.concatenate([ts[i - 1::-n], ts[i::n]])  # previous and forward
            out[i] = np.nanmean(ts_seas) * lr
    return out


def ApEn(U, m, r):
    """Compute Approximate Entropy"""

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)
    return abs(_phi(m + 1) - _phi(m))


def SampEn(U, m, r):
    """Compute Sample entropy"""

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))]
        return sum(C)

    N = len(U)
    return -np.log(_phi(m + 1) / _phi(m))


def granger_causation_matrix(data, variables, test="ssr_chi2test", verbose=False):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables."""
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
            if verbose: print(f"Y = {r}, X = {c}, P Values = {p_values}")
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + "_x" for var in variables]
    df.index = [var + "-y" for var in variables]
    return df


# Path for retrieving stock DataFrame
DATA_STORE = Path("C:\\Users\\domin\\PycharmProjects\\End-to-end\\dow_stocks.h5")

# Import our Dow Jones Industrial Average DataFrame
with pd.HDFStore(DATA_STORE, "r") as store:
    df = store.get("DOW/stocks")

df = calculate_returns(df)  # calc. returns

# TIME SERIES ANALYSIS #

# Get Individual Stock Data
df = df.loc[idx["WMT", :], ["close", "returns", "log_returns"]]

# Plot Stock Data
plot_data(df)
plt.show()

# Monthly Frequency of Log Returns
monthly_log_returns = calculate_monthly_log_returns(df)

# Plot Monthly Log Returns
plot_monthly_log_returns(monthly_log_returns)
plt.show()

# Seasonal Plot of a Time Series
# - data preparation
month_close = df.close.droplevel(0).resample("M").last()
data = pd.DataFrame(month_close, index=month_close.index)
data.reset_index(inplace=True)

data["year"] = [d.year for d in data.date]
data["month"] = [d.strftime("%b") for d in data.date]
years = data["year"].unique()

plot_seasonal_close_price(data, years)
plt.show()

# Boxplot of Month-wise (Seasonal) & Year-wise (trend) Distribution
plot_box_plots(data)
plt.show()

# Patterns in a Time Series
# Additive and Multiplicative Decomkposition of Time-Series
# Time Series Decomposition
data.set_index("date", inplace=True)
result_mul, result_add = perform_seasonal_decomposition(data)

plot_seasonal_decomposition(result_mul, result_add)

# Extract the Components
# Actual Values = Seasonal x Trend x Residual
data_reconstructed = pd.concat([result_mul.seasonal, result_mul.trend,
                                result_mul.resid, result_mul.observed], axis=1)
data_reconstructed.columns = ["seasonal", "trend", "residuals", "actual_values"]
print(data_reconstructed.head())

# Stationary and Non-stationary Time Series
# Making a Time Series Stationary
"""
1. Differencing the Series (once or more)
2. Take the log of the series
3. Take the nth root of the series
4. Combination of the above
"""
# Option 1  - can clearly tell from plot that mean, variance and autocorrelation are not constant over time
df.close.plot()
plt.show()

# Option 2 - contiguous splitting of a time series and looking at the descriptive statistics
part_1 = df["close"].iloc[0:int(0.2 * (len(df)))]
part_2 = df["close"].iloc[int(0.2 * (len(df))):int(0.4 * (len(df)))]
part_3 = df["close"].iloc[int(0.4 * (len(df))):int(0.6 * (len(df)))]
part_4 = df["close"].iloc[int(0.6 * (len(df))):int(0.8 * (len(df)))]
part_5 = df["close"].iloc[int(0.8 * (len(df))):len(df)]

describe = {"1": part_1,
            "2": part_2,
            "3": part_3,
            "4": part_4,
            "5": part_5}

for key, value in describe.items():
    print(key, ":", value.describe())

# Option 3 - ADF Test
result = adfuller(data.close.values, autolag="AIC")
print(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
for key, value in result[4].items():
    print("Critical Values:");
    print(f"{key}, {value}")

# Option4 - KPSS Test
result = kpss(data.close.values, regression="C")
print("\nKPSS Statistic: %f" % result[0], "p-value: %f" % result[1])
for key, value in result[3].items():
    print("Critical Values:");
    print(f"{key}, {value}")

#  White Noise vs Stationary Series
randvals = np.random.randn(1000)
pd.Series(randvals).plot(title='Random White Noise', color='k')
plt.show()

# Detrending a Time Series

# Method 1: TS - Line of best fit
detrended = signal.detrend(data.close.values)
plt.plot(detrended)
plt.title("Walmart Stock Adj Close Price Series "
          "Detrended by Deducting the OLS Fit", fontsize=13)
plt.tight_layout()
plt.show()

# Method 2: Minus trend from TS using decomposition component
result_mul = seasonal_decompose(data.close, model="multiplicative", extrapolate_trend="freq")
result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
plt.show()

detrended = data.close.values - result_mul.trend
plt.plot(detrended)
plt.title("Walmart Stock Monthly Adj Close Price Series "
          "Detrended by Deducting the Trend Component", fontsize=13)
plt.show()

# Method 3 - Subtract the Mean
mu = df.close.describe()

detrended = data.close.values - mu[1]  # mu is the mean of the entire monthly price series
plt.plot(detrended)
plt.title("WMT Monthly Adj Close Detrended by Subtracting the Mean")
plt.show()

# Method 4 - Apply a Filter like HP or Baxter-King
detrended = bkfilter(data.close, low=18, high=96, K=36)  # returns the cyclical component of x
plt.plot(detrended)
plt.title("Baxter-King Filter (measures the cycles)")
plt.show()

cf_cycles, cf_trend = cffilter(data.close, low=18, high=96, drift=True)  # drift: whether or not to remove the trend
# returns; cycle: the features of x between the periodicities low and high
#          trend: the trend in the data with cycles removed
cf = pd.DataFrame([np.array(cf_cycles), np.array(cf_trend)])
cf = cf.T
cf.columns = ["cf_cycles", "cf_trend"]
cf.index = data.index
fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
ax1.plot(cf["cf_cycles"])
ax1.set_title("Christiano Fitzgerald asymmetric, random walk filter (cycle component)")
ax2.plot(cf["cf_trend"])
ax2.set_title("Christiano Fitzgerald asymmetric, random walk filter (trend component (minus cycles))")
plt.show()

cycle, trend = hpfilter(data.close, lamb=129600)
hpf = pd.DataFrame([cycle, trend])
hpf = hpf.T
hpf.index = data.index
hpf.columns = ["hp_cycles", "hp_trend"]
fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
ax1.plot(hpf["hp_cycles"])
ax1.set_title("Hodrick-Prescott filter (cycle component)")
ax2.plot(hpf["hp_trend"])
ax2.set_title("Hodrick-Prescott filter (trend component)")
plt.show()

frs = cf.merge(hpf, left_index=True, right_index=True)
frs = frs.merge(data.close, left_index=True, right_index=True)
print(frs)

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(nrows=3, ncols=2)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(frs["close"])
ax1.set_title("WMT Monthly Close Price Series")
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot((frs.close - frs.cf_trend))
ax2.set_title("WMT Month Close - CF-Trend Component")
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot((frs.close - frs.hp_trend))
ax3.set_title("WMT Month Close - HP-Trend Component")
ax4 = fig.add_subplot(gs[2, 0], sharex=ax2)
ax4.plot((frs.close - frs.cf_cycles))
ax4.set_title("WMT Month Close - CF-Business Cycles")
ax5 = fig.add_subplot(gs[2, 1], sharex=ax3)
ax5.plot((frs.close - frs.hp_cycles))
ax5.set_title("WMT Month Close - HF-Business Cycles")
plt.show()

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(nrows=3, ncols=2)

ax1 = fig.add_subplot(gs[0, 0])
detrended = signal.detrend(data.close.values)
ax1.plot(detrended)
ax1.set_title("Detrended WMT Month Close Price: OLS Line of Best Fit")

result_mul = seasonal_decompose(data.close, model="multiplicative", extrapolate_trend="freq")
detrended = data.close.values - result_mul.trend
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(detrended)
ax2.set_title("Detrended WMT Month Close Price: Subtract Decomposition Trend")

mu = df.close.describe()
detrended = data.close.values - mu[1]
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(detrended)
ax3.set_title("Detrended WMT Month Close Price: Subtract the Average")

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot((frs.close - frs.hp_trend))
ax4.set_title("Detrended WMT Month Close Price: Minus the HP Trend")

diff = pd.Series(data.close.diff())
diff.dropna(inplace=True)
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(diff)
ax5.set_title("Detrended WMT Month Close Price: Differenced")

ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(data.close, color="orange")
ax6.set_title("WMT Close Price, Not Detrended!")
plt.show()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
ax1.plot(data.close)
ax1.set_title("WMT Monthly Close Price")
ax2.plot(diff)
ax2.set_title("Differenced")
ax3.plot(detrended)
ax3.set_title("Detrended: Close - Average")

roll_avg = data.close.rolling(window=12).mean()
roll_avg.dropna(inplace=True)
ax4.plot((data.close[11:] - roll_avg))
ax4.set_title("Detrended: Close - MA")
plt.show()

# Deseasonalize a Time Series
# Method 1: done just above in the plot

# Time Series Decomposition
result_mul = seasonal_decompose(x=data.close, model="multiplicative", extrapolate_trend="freq")
# Deseasonalize
deseasonalized = data.close.values / result_mul.seasonal

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
ax1.plot(data.close)
ax1.set_title("Close Price")
ax2.plot(diff)
ax2.set_title("Differenced Price (detrended)")
ax3.plot(deseasonalized)
ax3.set_title("Deseasonalized Monthly Price Series of WMT")
plt.tight_layout()
plt.show()

# Testing for Seasonality in a Time Series
plt.rcParams.update({'figure.figsize': (9, 5), 'figure.dpi': 120})
autocorrelation_plot(data.close.tolist())
plt.show()

# Canonva-Hansen Test for Seasonal Differences
CHTest = arima.CHTest(m=12)
params = CHTest.get_params(deep=True)
CHTest = CHTest.estimate_seasonal_differencing_term(x=data.close)
print("CH Statistic is compared to a different critical value"
      ", and returns 1 if statistic is greater than the critical"
      " value, or 0 if not. Result:", CHTest)
if CHTest == 0:
    print("Statistic less than CV; fail to reject null hypothesis (e.g no effect)")
else:
    print("Statistic greater than CV; can declare statistical significance and reject"
          "the null hypothesis")
print(params)

# direct estimate of the number of seasonal differences
est_no = arima.nsdiffs(x=data.close, m=12, max_D=12, test="ch")
print("Estimation of the seasonal differencing term required to make a given TS stationary:")
if est_no == 0:
    print(f"Estimated Seasonal Differencing Term: {est_no}. This means the TS is constant.")
else:
    print(f"Estimated Seasonal Differencing Term: {est_no}.")

# Treating Missing Values in a Time Series
"""
Backward Fill
Linear Interpolation
Quadratic interpolation
Mean of nearest neighbors
Mean of seasonal couterparts
"""
# Add NaN values to the Data
np.random.seed(42)

df1 = data.copy()
df1["value"] = df1.close
df1.drop(columns=["year", "month"], inplace=True)

for col in df1.columns:
    df1.loc[df1.sample(frac=0.1).index, col] = np.nan

df1.drop(columns=["value"], inplace=True)

print("No. of null/nan values: ", df1.isna().sum())
print(df1)

fig, axes = plt.subplots(7, 1, sharex=True, figsize=(10, 12))
plt.rcParams.update({'xtick.bottom': False})

# 1. Actual ----------------------
data.close.plot(title="Actual", ax=axes[0], label="Actual", color="red", style=".-")
df1.close.plot(title="Actual", ax=axes[0], label="Actual", color="green", style=".-")
axes[0].legend(["Missing Data"])

# 2. Forward Fill ----------------
df_ffill = df1.ffill()
error = np.round(mean_squared_error(data.close, df_ffill.close), 2)
df_ffill.close.plot(title="Forward Fill (MSE: " + str(error) + ")", ax=axes[1], label="Forward Fill", style=".-")

# 3. Backward Fill ---------------
df_bfill = df1.bfill()
error = np.round(mean_squared_error(data.close, df_bfill.close), 2)
df_bfill.close.plot(title="Backward Fill (MSE: " + str(error) + ")", ax=axes[2],
                    label="Backward Fill", color="firebrick", style=".-")

# 4. Linear Interpolation --------
df1["rownum"] = np.arange(df1.shape[0])
df_nona = df1.dropna(subset=["close"])
f = interp1d(df_nona["rownum"], df_nona["close"])
df1["linear_fill"] = f(df1["rownum"])
error = np.round(mean_squared_error(data.close, df1["linear_fill"]), 2)
df1["linear_fill"].plot(title="Linear Fill (MSE: " + str(error) + ")", ax=axes[3],
                        label="Linear Fill", color="brown", style=".-")

# 5. Cubic Interpolation ---------
f2 = interp1d(df_nona["rownum"], df_nona.close, kind="cubic")
df1["cubic_fill"] = f2(df1["rownum"])
error = np.round(mean_squared_error(data.close, df1["cubic_fill"]), 2)
df1["cubic_fill"].plot(title="Cubic Fill (MSE: " + str(error) + ")", ax=axes[4],
                       label="Cubic Fill", color="red", style=".-")

# 6. Mean of "n" Nearest Past Neighbours -----
df1["knn_mean"] = knn_mean(df1.close.values, 8)
error = np.round(mean_squared_error(data.close, df1["knn_mean"]), 2)
df1["knn_mean"].plot(title="KNN Mean (MSE: " + str(error) + ")", ax=axes[5],
                     label="KNN Mean", color="tomato", alpha=0.5, style=".-")

# 7. Seasonal Mean
df1["seasonal_mean"] = seasonal_mean(df1.close, n=12, lr=1.25)
error = np.round(mean_squared_error(data.close, df1["seasonal_mean"]), 2)
df1["seasonal_mean"].plot(title="Seasonal Mean (MSE: " + str(error) + ")", ax=axes[6],
                          label="Seasonal Mean", color="blue", alpha=0.5, style=".-")
plt.show()

# Autocorrelation and Partial Autocorrelation Fn's

# Calculate ACF and PACF up to 50 lags
acf_50 = acf(data.close, nlags=50)
pacf_50 = pacf(data.close, nlags=50)

# Draw Plot
fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(16, 3), dpi=100)
plot_acf(data.close.tolist(), lags=50, ax=axes[0])
plot_pacf(data.close.tolist(), lags=50, ax=axes[1])
plt.show()

plt.rcParams.update({"ytick.left": False, "axes.titlepad": 10})

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 3), sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:5]):
    lag_plot(data.close, lag=i + 1, ax=ax, c="firebrick")
    ax.set_title("Lag " + str(i + 1))
fig.suptitle("Lag Plots of WMT Monthly Close Price (Points get "
             "wide and scattered with increasing lag -> lesser correlation))", y=1.05)
plt.show()

# How to Estimate Forecastability of a Time Series
rand_small = np.random.randint(0, 100, size=36)
rand_big = np.random.randint(0, 100, size=136)

print(f"Approximate Entropy WMT Monthly Stock Price: {ApEn(data.close, m=2, r=0.2 * np.std(data.close))}")
print(f"AE rand_small: {ApEn(rand_small, m=2, r=0.2 * np.std(rand_small))}")
print(f"AE rand_big: {ApEn(rand_big, m=2, r=0.2 * np.std(rand_big))}")

print(f"Sample Entropy WMT Monthly Stock Price: {SampEn(data.close, m=2, r=0.2 * np.std(data.close))}")
print(f"SE rand_small: {SampEn(rand_small, m=2, r=0.2 * np.std(rand_small))}")
print(f"SE rand_big: {SampEn(rand_big, m=2, r=0.2 * np.std(rand_big))}")

# Why and How to Smoothen A Time Series
"""
So how to smoothen a series?:
1. Take a moving average
2. Do a LOESS smoothing (Localized Regression)
3. Do a LOWESS smoothing (Locally Weighted Regression)
"""

# plt.rcParams({"xtick.bottom": False, "axes.titlepad": 5})

# 1. MA
close_ma = data.close.rolling(window=3, center=True, closed="both").mean()

# 2. LOESS Smoothing (5% and 15%)
close_loess_5 = pd.Series(lowess(endog=data.close, exog=np.arange(len(data.close)), frac=0.05)[:, 1], index=data.index,
                          name="close")

close_loess_15 = pd.Series(lowess(endog=data.close, exog=np.arange(len(data.close)), frac=0.15)[:, 1], index=data.index,
                           name="close")

# Plot
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(7, 7), sharex=True, dpi=120)
data.close.plot(ax=axes[0], color="k", title="Original WMT Month Close Price Series")
close_loess_5.plot(ax=axes[1], title="Loess Smoothed 5%")
close_loess_15.plot(ax=axes[2], title="Loess Smoothed 15%")
close_ma.plot(ax=axes[3], title="Moving Average (3)")
fig.suptitle("How to Smoothen a Time Series", y=0.95, fontsize=14)
plt.show()

# Using Granger Causality Test to Know if One Time Series is Helpful in Forecasting Another
dates = data.index
data["month_1"] = dates.month

grangercausalitytests(data[["close", "month_1"]], maxlag=2)

maxlag = 2
test = "ssr_chi2test"

train = data[["close", "month_1"]]
o = granger_causation_matrix(train, variables=train.columns)
print(o)

if o.iloc[0, 1] <= 0.05:
    print("P-Value less than 0.05:", o.iloc[0, 1] + ". Therefore, the variable X is useful for forecasting.")
else:
    print(f"P-Value is: {o.iloc[0, 1]}. Therefore, the lag of X is not useful.")
