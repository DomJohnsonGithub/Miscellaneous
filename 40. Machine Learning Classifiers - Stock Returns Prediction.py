import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import talib as ta
import pickle
import numba
from deeptime.decomposition import DMD

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.base import BaseEstimator, TransformerMixin

sns.set_style("darkgrid")

idx = pd.IndexSlice


class BlockedTimeSeriesPurgedSplit:
    def __init__(self, n_splits, purge_gap, train_percentage):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.train_pct = train_percentage

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def get_purge_gap(self):
        return self.purge_gap

    def get_train_percentage(self):
        return self.train_pct

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            begin = i * k_fold_size
            stop = begin + k_fold_size
            mid = int(self.train_pct * (stop - begin)) + begin

            yield indices[begin: mid], indices[mid + self.purge_gap: stop]


@numba.njit
def lorentzian_distance(x1, x2):
    return np.sqrt(np.sum(np.log(1 + (np.abs(x1 - x2) ** 2))))


class DynamicModeDecompositionTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.n_components = int
        self.modes = None

    def fit(self, X, y=None):
        U, S, V = np.linalg.svd(X)
        cum_sum_variance = np.cumsum(np.square(S) / (np.sum(np.square(S))))
        self.n_components = int(min(np.argwhere(cum_sum_variance >= 0.95)))
        dmd = DMD(mode="exact", rank=self.n_components)
        fit = dmd.fit((X[:-1], X[1:])).fetch_model()
        self.modes = fit.modes

    def transform(self, X, y=None):
        transformed_data = (X @ self.modes.T).real
        unique_data = np.unique(transformed_data.T, axis=0)
        return unique_data.T

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)


@numba.njit
def generate_weight_combinations(n):
    possible_values = np.round(np.arange(0, 1.05, 0.05), 2)
    num_possible_values = len(possible_values)

    valid_combos = []
    for i in range(num_possible_values ** n):
        combo = np.zeros(n)
        carry = i
        for j in range(n):
            combo[j] = possible_values[carry % num_possible_values]
            carry = carry // num_possible_values

        if np.sum(combo) <= 1:
            valid_combos.append(list(combo))

    return valid_combos[1:]


# Get Data
symbol_names = ["^GSPC", "EURUSD=X", "CL=F", "^TNX", "TLT", "WPC"]
df = yf.download(symbol_names, start=datetime(1999, 1, 1), end=datetime.now() - timedelta(1)).drop(
    columns=["Adj Close", "Volume"]).dropna().stack(1).swaplevel(0).sort_index(level=0)
df.index.names = ["Ticker", "Date"]

# Feature Engineering
lags = [1, 2, 3, 4, 5]

# Returns
for lag in lags:
    df[f"returns_{lag}"] = df.groupby(level="Ticker").Close.pct_change(lag)

# Lag Returns
for lag in lags:
    df[f"returns_1_t-{lag}"] = df.groupby(level="Ticker")["returns_1"].shift(lag)

# Returns Momentum
for lag1 in lags:
    for lag2 in lags[1:]:
        if lag2 > lag1:
            df[f"momentum_{lag2}_{lag1}"] = df[f"returns_{lag2}"] - df[f"returns_{lag1}"]


# Ranges
def close_to_open_returns(data):
    return data.Close / data.Open.shift() - 1

def high_low_diff(data):
    return (data.High - data.Low).diff()

def open_close_diff(data):
    return (data.Open - data.Close).diff()

df["co_returns"] = df.groupby(level="Ticker", group_keys=False).apply(close_to_open_returns)
df["high_low_diff"] = df.groupby(level="Ticker", group_keys=False).apply(high_low_diff)
df["open_close"] = df.Open - df.Close
df["open_close_diff"] = df.groupby(level="Ticker", group_keys=False).apply(open_close_diff)


# Sum of Returns
def sum_rets(data, n):
    return data["returns_1"].rolling(n).sum()


for i in symbol_names:
    rets = df.loc[idx[f"{i}", :], "returns_1"].droplevel(0)
    rets = pd.DataFrame(rets, index=rets.index, columns=["returns_1"])
    for window in lags[1:]:
        df.loc[idx[f"{i}", :], f"sum_returns_{window}"] = rets["returns_1"].rolling(window).sum().values


# Standard Deviation, Skewness, Kurtosis
def sd(data):
    return data.rolling(5).std().diff()


def skewness(data):
    return data.rolling(5).skew().diff()


def kurtosis(data):
    return data.rolling(5).kurt().diff()


df["standard_deviation"] = df.groupby(level="Ticker", group_keys=False)["returns_1"].apply(sd)
df["skewness"] = df.groupby(level="Ticker", group_keys=False)["returns_1"].apply(skewness)
df["kurtosis"] = df.groupby(level="Ticker", group_keys=False)["returns_1"].apply(kurtosis)


# Momentum and Trend Indicators
def rsi(data):
    return ta.RSI(data, timeperiod=5)


def cci(data):
    return ta.CCI(data.High, data.Low, data.Close, timeperiod=5)


def stoch(data):
    return ta.STOCHF(data.High, data.Low, data.Close, fastk_period=5, fastd_period=3, fastd_matype=0)[0]


def macd(data):
    return ta.MACD(data, fastperiod=3, slowperiod=5, signalperiod=2)[0]


def aroon(data):
    return ta.AROONOSC(data.High, data.Low, timeperiod=5)


def bop(data):
    return ta.BOP(data.Open, data.High, data.Low, data.Close)


def adx(data):
    return ta.ADX(data.High, data.Low, data.Close, timeperiod=5)


df["RSI"] = df.groupby(level="Ticker", group_keys=False).Close.apply(rsi)
df["CCI"] = df.groupby(level="Ticker", group_keys=False).apply(cci)
df["STOCH"] = df.groupby(level="Ticker", group_keys=False).apply(stoch)
df["AROON"] = df.groupby(level="Ticker", group_keys=False).apply(aroon)
df["ASOI"] = df.iloc[:, -4:].mean(axis=1)
df["BOP"] = df.groupby(level="Ticker", group_keys=False).apply(bop)
df["MACD"] = df.groupby(level="Ticker", group_keys=False).Close.apply(macd)
df["ADX"] = df.groupby(level="Ticker", group_keys=False).apply(adx).diff()
df = df.drop(columns=["RSI", "CCI", "STOCH", "AROON"])

# Cyclical Features
df = df.unstack(0)
index = df.index
quarter, month, week, day = index.quarter, index.month, index.isocalendar().week, index.day
df = df.stack(1).swaplevel(0).sort_index(level=0)

for transform in ["sin", "cos"]:
    for freq, frequency, num in zip(["quarter", "month", "week", "day"],
                                    [quarter, month, week, day], [4, 12, 52, 365]):
        if transform == "sin":
            df[f"{transform}_{freq}"] = np.tile(np.sin(2 * np.pi * frequency / num), len(symbol_names))
        else:
            df[f"{transform}_{freq}"] = np.tile(np.cos(2 * np.pi * frequency / num), len(symbol_names))

sin, cos = "sin", "cos"
for freq in ["quarter", "month", "week", "day"]:
    df[f"{freq}"] = df[f"{sin}_{freq}"] - df[f"{cos}_{freq}"]
df = df.drop(
    columns=[f"{transform}_{freq}" for transform in ["sin", "cos"] for freq in ["quarter", "month", "week", "day"]])

# Remove non-stationary variables and drop NaN values
df = df.iloc[:, 4:].dropna()

# Create Target Variable
df["target"] = np.where(df.groupby(level="Ticker").returns_1.shift(-1) >= 0, 1, -1)

# Remove last day due to target variable creation
df = df.sort_index(level=1).iloc[:-len(symbol_names), :].sort_index(level=0)

# Separate data into list of dataframes for ML algo's
data = [df.loc[idx[f"{symbol}", :], :].droplevel(0) for symbol in symbol_names]
X_data = [data[i].drop(columns="target") for i, symbol in enumerate(data)]
y_data = [data[i]["target"] for i, symbol in enumerate(data)]

train_data_pct = 0.8
X_train = [X_data[i].iloc[:int(len(X_data[i]) * train_data_pct), :] for i, symbol in enumerate(X_data)]
X_test = [X_data[i].iloc[int(len(X_data[i]) * train_data_pct):, :] for i, symbol in enumerate(X_data)]
y_train = [y_data[i][:int(len(y_data[i]) * train_data_pct)] for i, symbol in enumerate(y_data)]
y_test = [y_data[i][int(len(y_data[i]) * train_data_pct):] for i, symbol in enumerate(y_data)]

# Cross Validator
cv = BlockedTimeSeriesPurgedSplit(n_splits=7, train_percentage=0.7, purge_gap=1)

# Transformers
continuous_transformer = Pipeline(steps=[("yeo_johnson_sc", PowerTransformer(standardize=True))])
decomposition_transformer = Pipeline(steps=[
    ("yeo_johnson_sc", PowerTransformer(standardize=True)),
    ("dynamic_mode_decomp", DynamicModeDecompositionTransformer())])

# Column Transformer
column_transformer = ColumnTransformer(transformers=[
    ("returns", decomposition_transformer, np.arange(0, 5)),
    ("lag_returns", decomposition_transformer, np.arange(5, 10)),
    ("momentum", decomposition_transformer, np.arange(10, 20)),
    ("ranges", decomposition_transformer, np.arange(20, 26)),
    ("sum_returns", decomposition_transformer, np.arange(26, 30)),
    ("other", continuous_transformer, np.arange(31, 41))],
    n_jobs=11, remainder="passthrough")

# Models
classifiers = [
    LogisticRegression(penalty="elasticnet", class_weight="balanced", max_iter=10_000,
                       fit_intercept=True, solver="saga", n_jobs=11),
    KNeighborsClassifier(n_jobs=11),
    SVC(kernel="rbf", max_iter=-1, shrinking=False,
        class_weight="balanced", probability=True),
    ExtraTreesClassifier(criterion="entropy", class_weight="balanced_subsample",
                         n_jobs=11, bootstrap=False),
    RandomForestClassifier(criterion="gini", bootstrap=True, n_jobs=11,
                           class_weight="balanced_subsample"),
    GaussianNB()
]

# Parameter Grid
param_grid = [
    {"lr__C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
     "lr__l1_ratio": np.round(np.arange(0, 1.1, 0.1), 1)},
    {"knn__n_neighbors": np.arange(5, 33, 2),
     "knn__weights": ["distance", "uniform"],
     "knn__metric": ["manhattan", "euclidean", "cosine", lorentzian_distance,
                     "correlation", "braycurtis", "canberra", "chebyshev"]},
    {"svm__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
     "svm__gamma": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, "scale", "auto"]},
    {"et__n_estimators": np.arange(100, 900, 100),
     "et__max_depth": np.arange(2, 11, 1)},
    {"rf__n_estimators": np.arange(100, 900, 100),
     "rf__max_depth": np.arange(2, 11, 1)}
]

model_names = ["lr", "knn", "svm", "et", "rf", "nb"]
pipelines = [Pipeline(steps=[("ct", continuous_transformer), (f"{name}", classifiers[i])]) for i, name in
             enumerate(model_names)]

# Grid Search Hyperparameter and Cross-validation Optimization
models_for_training = [GridSearchCV(pipelines[i], param_grid[i], scoring="accuracy", n_jobs=11, refit=True, cv=cv,
                                    verbose=3) for i in range(len(model_names) - 1)] + [
                          pipelines[range(len(model_names))[-1]]]

# # Fit Models for each dataset (symbol/ticker)
# for symbol, num in zip(symbol_names, range(len(symbol_names))):
#
#     trained_models = [model.fit(X_train[num], y_train[num]) for model in models_for_training]
#
#     # Prepare Estimators for Voting Classifier
#     estimators = [Pipeline([("ct", continuous_transformer), (f"{i}", untrained.set_params(
#         **{key.replace(f'{i}__', ''): value for key, value in
#            model.best_params_.items()}))]) if i != "nb" else Pipeline(
#         [("ct", continuous_transformer), (f"{i}", untrained)]) for i, model, untrained in
#                   zip(model_names, trained_models, classifiers)]
#
#     # Ensemble Voting Classifiers
#     vclf = VotingClassifier(estimators=[(f"clf{i}", j) for i, j in enumerate(estimators)], voting="soft", n_jobs=11)
#     vclf.fit(X_train[num], y_train[num])
#
#     # Optimising Soft Vote Weights - randomly select rows from all possible weights
#     weights = np.array(generate_weight_combinations(n=6))
#     num_rows = 10
#     random_matrix = weights[np.random.choice(weights.shape[0], size=num_rows, replace=False)]
#
#     accuracies = []
#     for j in range(np.shape(random_matrix)[0]):
#         vclf.weights = random_matrix[j, :]
#         accuracies.append(accuracy_score(y_test[num], np.where(vclf.predict_proba(X_test[num])[:, 1] >= 0.5, 1, -1)))
#
#     index_best_acc = np.argmax(accuracies)
#     vclf.weights = random_matrix[index_best_acc, :]
#
#     with open(f"voting_classifier_{symbol}", "wb") as fp:
#         pickle.dump(vclf, fp)


plt.plot()
plt.show()



final_classifier_per_symbol = []
for ticker in symbol_names:
    with open(f"voting_classifier_{ticker}", "rb") as fp:
        final_classifier_per_symbol.append(pickle.load(fp))

print(final_classifier_per_symbol)
