from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as pdr
import math
import talib as ta
from pykalman import KalmanFilter
import yfinance as yf

sns.set_style("darkgrid")

stock = "^GSPC"
df = yf.download(
    stock,
    start=datetime(2000, 1, 1),
    end=datetime.now() - timedelta(1)).drop(
    columns=["Adj Close"])

kf = KalmanFilter(transition_matrices=[1],
                  observation_matrices=[1],
                  initial_state_mean=0,
                  initial_state_covariance=1,
                  observation_covariance=1,
                  transition_covariance=.0001)

# plt.plot(df.Close)
# plt.plot(df.Close.index, kf.filter(df.Close)[0])
# plt.show()

df["returns"] = np.log1p(df.Close.pct_change())
df.dropna(inplace=True)
df["kf_close"] = kf.filter(df.Close)[0]
df = df.iloc[100:, :]
df["kf_returns"] = np.log1p(df.kf_close.pct_change())

df.dropna(inplace=True)
print(df)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

std_sc = StandardScaler()
norm_sc = MinMaxScaler()

X = df.loc[:, ["kf_returns"]]

X = norm_sc.fit_transform(std_sc.fit_transform(X))

from hmmlearn import hmm
from matplotlib import cm


def train_plot_hmm(X):
    model = hmm.GaussianHMM(n_components=2, n_iter=10_000, covariance_type="full")
    fit = model.fit(X)
    print(model.score(X))
    prediction = model.predict(X)
    print(prediction)

    print(model.means_)
    print(model.covars_)

    fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
    colours = cm.rainbow(np.linspace(0, 1, model.n_components))
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = prediction == i
        ax.plot_date(
            df.index[mask],
            df["Close"][mask],
            c=colour, markersize="1")
        ax.set_title("Hidden State #%s" % i)
    plt.show()


train_plot_hmm(X)

from pyts.image import MarkovTransitionField
import tsia

# --------------------
# Markov chain - probability of bull or bear next day
X_binned, bin_edges = tsia.markov.discretize(df.returns, n_bins=2)
print(X_binned)
print(bin_edges)
X_mtm = tsia.markov.markov_transition_matrix(X_binned)
X_mtm = tsia.markov.markov_transition_probabilities(X_mtm)
print(np.round(X_mtm, 2))
print([0, 1] * X_mtm)
# -------------------------

# -------------------------
# Markov Transition Field
x = df.returns.values
X = np.array([x])
mtf = MarkovTransitionField(n_bins=10)
X_mtf = mtf.fit_transform(X)[0]

# denoise image
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, dpi=50)
axes[0].imshow(X_mtf)
axes[1].imshow(denoise_tv_chambolle(X_mtf, weight=0.1, channel_axis=-1))
plt.show()

COLORMAP = 'jet'


def get_mtf_map(timeseries,
                mtf,
                step_size=0,
                colormap=COLORMAP,
                reversed_cmap=False):
    image_size = mtf.shape[0]
    mtf_min = np.min(mtf)
    mtf_max = np.max(mtf)
    mtf_range = mtf_max - mtf_min
    mtf_colors = (np.diag(mtf, k=step_size) - mtf_min) / mtf_range

    # Define the color map:
    if reversed_cmap == True:
        colormap = plt.cm.get_cmap(colormap).reversed()
    else:
        colormap = plt.cm.get_cmap(colormap)

    mtf_map = []
    sequences_width = timeseries.shape[0] / image_size
    for i in range(image_size - step_size):
        c = colormap(mtf_colors[i])
        # start = int(i * sequences_width)
        # end = int((i + 1) * sequences_width - 1) + 1
        # data = timeseries.iloc[start:end, :]

        # current_map = dict()
        # current_map.update({
        #     'color': c,
        #     'slice': data
        # })
        # mtf_map.append(current_map)
        mtf_map.append(c)

    # for i in range(image_size - step_size, image_size):
    #     c = '#DDDDDD'
    #     start = int(i * sequences_width)
    #     end = int((i + 1) * sequences_width - 1)
    #     data = timeseries.iloc[start:end, :]
    #
    #     current_map = dict()
    #     current_map.update({
    #         'color': c,
    #         'slice': data
    #     })
    #     mtf_map.append(current_map)
    #     print("-:", c)

    return mtf_map


mtf_map = get_mtf_map(df, X_mtf, reversed_cmap=True)

plt.scatter(df.index, df.returns, c=mtf_map)
plt.show()
# -----------------------------------

# Markov Transition Matrix
data = df.drop(columns="returns").copy()
print(data)
data["rets"] = data.Close.pct_change()
data["log_rets"] = np.log1p(data.Close.pct_change())
data["empirical_dist"] = data.log_rets.rank(method="average") / len(df)

data.dropna(inplace=True)

data["rets_bin"] = pd.qcut(data.log_rets, q=2, labels=[1, 2]).astype(float)
data["empirical_bin"] = pd.qcut(data.empirical_dist, q=2, labels=[1, 2]).astype(float)

data = data.loc["2015-01-01": "2020-01-01", :]

X_binned1, bin_edges = tsia.markov.discretize(data.log_rets, n_bins=2)
data["states"] = X_binned1 + 1

X_mtm1 = tsia.markov.markov_transition_matrix(X_binned=X_binned1)
print(X_mtm1)

X_mtm1 = tsia.markov.markov_transition_probabilities(X_mtm1)
print(X_mtm1)
print(np.sum(X_mtm1, axis=1))
print(X_mtm1 @ X_mtm1 @ X_mtm1 @ X_mtm1 @ X_mtm1 @ X_mtm1 @ X_mtm1)

data["signal"] = 0
for i in range(len(data)):
    if data.iloc[i, 10] == 1:
        data.iloc[i + 1, 11] = data.iloc[i + 1, 5]

print(data[["rets", "states", "signal"]].head(10))

# print((np.product(1 + data.rets) ** (1 / 5) - 1) * 100)
# print((np.product(1 + data.signal) ** (1 / 5) - 1) * 100)
