import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr
from datetime import datetime
import yfinance as yf
from itertools import chain


# RANDOM MATRIX FILTERING IN FINANCE


def marchenko_pastur_pdf(x, Q, sigma=1):
    y = 1 / Q
    b = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)  # Largest eigenvalue
    a = np.power(sigma * (1 - np.sqrt(1 / Q)), 2)  # Smallest eigenvalue
    return (1 / (2 * np.pi * sigma * sigma * x * y)) * np.sqrt((b - x) * (x - a)) * (0 if (x > b or x < a) else 1)


def compare_eigenvalue_distribution(correlation_matrix, Q, sigma=1, set_autoscale=True, show_top=True):
    e, _ = np.linalg.eig(correlation_matrix)  # correlation matrix is Hermitian, so is faster than other variants of eig

    x_min = 0.0001 if np.power(sigma * (1 - np.sqrt(1 / Q)), 2) < 0.0001 else np.power(sigma * (1 - np.sqrt(1 / Q)), 2)
    x_max = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = 50
    if not show_top:
        # Clear top eigenvalue from plot
        e = e[e <= x_max + 1]
    ax.hist(e, density=True, bins=50)  # histogram of eigenvalues
    ax.set_autoscale_on(set_autoscale)

    # Plot the theoretical density
    f = np.vectorize(lambda x: marchenko_pastur_pdf(x, Q, sigma=sigma))

    x_min = .0001 if np.power(sigma * (1 - np.sqrt(1 / Q)), 2) < .0001 else np.power(sigma * (1 - np.sqrt(1 / Q)), 2)
    x_max = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)

    x = np.linspace(x_min, x_max, 5000)
    ax.plot(x, f(x), linewidth=4, color='r')
    plt.grid(True)
    plt.show()


# Create the correlation matrix and find the eigenvalues
N = 500
T = 1000
X = np.random.normal(0, 1, size=(N, T))
cor = np.corrcoef(X)
Q = T / N
compare_eigenvalue_distribution(cor, Q)

# CORRELATION FILTERING

# Import Data
DATA_STORE = "C:\\Users\\domin\\PycharmProjects\\Portfolio_Eigendecomposition\\russell2000.h5"
with pd.HDFStore(DATA_STORE, "r") as store:
    df = store.get("RUSSELL2000/stocks")["close"]

df = df.dropna(axis=1, thresh=len(df.index) / 2)  # drop columns with less than half of dataframe length

# Randomly sample 100 tickers
np.random.seed(42)
tickers = df.columns.copy()
print(len(tickers))
tickers = np.random.choice(tickers, size=100, replace=False)
print(tickers)
print(len(tickers))

# Get Data from 2010 onwards till 2015
df = df.loc["2010-01-01":"2015-01-01", df.columns.intersection(tickers)]
print(df)
print(len(df.columns))

print(sorted(df.columns))
print(sorted(tickers))


# Check for NaN Values
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


print_full(df.isnull().sum())

# # Remove any NaN Values via Imputation
# print(df[df["AVXL"].isna()])
# print(df.loc["2012-04-30":"2012-05-05", "AVXL"])
#
# df.interpolate(method="time", inplace=True)
# print(df.loc["2012-04-30":"2012-05-05", "AVXL"])
#
# # Replacing infinite with nan
# df.replace([np.inf, -np.inf], np.nan, inplace=True)
# # Dropping all the rows with nan values
# df.dropna(inplace=True)
# print(df)

clean_df = df.copy()
clean_df = clean_df.pct_change()  # calc. returns via pct_change() method
clean_df.dropna(inplace=True)

# Plot of a single ticker to see the distribution after calculating the returns from the adj close data
fig = plt.figure()
ax = plt.subplot(111)
ax.hist(clean_df["TISI"], bins=50, density=True, label="TISI Distrb'n after transform to returns", histtype="bar")
plt.legend(loc="best")
plt.show()

# Separate into in sample and out of sample data sets
nobs = 21
train = clean_df.iloc[:-nobs, :]
test = clean_df.iloc[-nobs:, :]
print("length of train: ", len(train))
print("length of test: ", len(test))

# Log Transformation
log_train = train.apply(lambda x: np.log(x + 1)).dropna()
print("Log in sample data:")
print(log_train)

# will need variance and standard deviation next:
variances = np.diag(log_train.cov().values)
standard_deviations = np.sqrt(variances)

# The Eigenvalues of the Correlation Matrix
T, N = clean_df.shape  # pandas does the reverse of what is written in the first section
Q = T / N
correlation_matrix = log_train.interpolate().corr()

compare_eigenvalue_distribution(correlation_matrix, Q, set_autoscale=True, show_top=False)

# let's see the eigenvalues larger than the largest theoretical eigenvalue
sigma = 1  # the variance for all of the standardized log returns is 1
max_theoretical_eval = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)
D, S = np.linalg.eigh(correlation_matrix)
print(D[D > max_theoretical_eval])
print(len(D[D > max_theoretical_eval]))

# Filter the eigenvalues out
D[D <= max_theoretical_eval] = 0
print(len(D[D <= max_theoretical_eval] == 0))

# Reconstruct the matrix
temp = np.dot(S, np.dot(np.diag(D), np.transpose(S)))

# Set the diagonal entries to 1
np.fill_diagonal(temp, 1)
filtered_matix = temp

fig = plt.figure()
ax = plt.subplot(121)
ax.imshow(correlation_matrix)
plt.title("Original")
ax = plt.subplot(122)
plt.title("Filtered")
a = ax.imshow(filtered_matix)
cbar = fig.colorbar(a, ticks=[-1, 0, 1])
plt.show()

# COMPARISON FOR CONSTRUCTING THE MINIMUM VARIANCE PORTFOLIO

# Reconstruct the filtered covariance matrix
covariance_matrix = train.cov()
inv_cov_mat = np.linalg.inv(covariance_matrix)

# Construct minimum variance weights
ones = np.ones(len(inv_cov_mat))
inv_dot_ones = np.dot(inv_cov_mat, ones)
min_var_weights = inv_dot_ones / np.dot(inv_dot_ones, ones)

fig = plt.figure()
ax = plt.subplot(121)
min_var_portfolio = pd.DataFrame(data=min_var_weights, columns=["Investment Weight"], index=tickers)
min_var_portfolio.plot(kind="bar", ax=ax)
plt.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")
plt.title("Minimum Variance")

# Reconstruct the filtered covariance matrix from the standard deviations and the filtered correlation matrix
filtered_cov = np.dot(np.diag(standard_deviations), np.dot(filtered_matix, np.diag(standard_deviations)))
filtered_inv_cov = np.linalg.pinv(filtered_cov)

# Construct minimum variance weights
ones = np.ones(len(filtered_inv_cov))
inv_dot_ones = np.dot(filtered_inv_cov, ones)
filt_min_var_weights = inv_dot_ones / np.dot(inv_dot_ones, ones)

ax = plt.subplot(122)
filt_min_var_portfolio = pd.DataFrame(filt_min_var_weights, columns=["Investment Weight"], index=tickers)
filt_min_var_portfolio.plot(kind="bar", ax=ax)
plt.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")
plt.title("Filtered Minimum Variance")
plt.show()

print(filt_min_var_portfolio.head())


# Plot Return over time. Since both contain short sales, remove short sales and redistribute their weight
def cum_rets(sample, weights):
    # Ignoring short sales
    weights[weights <= 0] = 0
    weights = weights / weights.sum()
    return (((1 + sample.values).cumprod(axis=0) - 1)).dot(weights)


cumulative_returns = cum_rets(sample=clean_df, weights=min_var_portfolio)
cumulative_returns_filt = cum_rets(sample=clean_df, weights=filt_min_var_portfolio)

cumulative_returns = pd.Series(list(chain(*cumulative_returns)))
cumulative_returns_filt = pd.Series(list(chain(*cumulative_returns_filt)))

fig = plt.figure(figsize=(16, 4))

ax = plt.subplot(131)
ax.plot(cumulative_returns[:-nobs], c="b")
ax.plot(cumulative_returns[-nobs:], c="r")
plt.title("Minimum Variance Portfolio")

ax = plt.subplot(132)
ax.plot(cumulative_returns_filt[:-nobs], c="b")
ax.plot(cumulative_returns_filt[-nobs:], c="r")
plt.title("Filtered Minimum Variance Portfolio")

ax = plt.subplot(133)
ax.plot(cumulative_returns, c="magenta")
ax.plot(cumulative_returns_filt, c="orange")
plt.title("Filtered (magenta) vs. Normal(orange)")
plt.show()
