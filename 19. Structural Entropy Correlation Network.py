import pandas as pd
import pandas_datareader.data as pdr
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import requests
import re
from itertools import chain
from pathlib import Path
from tqdm import tqdm
import networkx as nx
from sklearn.decomposition import PCA
from scipy.sparse.csgraph import connected_components


# Define Utility Functions for Structural Entropy Calc.
def get_timespan(df, start, end):
    return df.iloc[:, start:end].T


def corr_matrix(df, start, end):
    seq = get_timespan(df, start, end)
    corr_seq = seq.corr().values
    return corr_seq


def structural_entropy(df, sequence_length, t):
    structural_entropy = {"timestamp": [], "structural_entropy": []}

    for d in tqdm(range(sequence_length, df.shape[1])):
        _corr = corr_matrix(df, d - sequence_length, d)

        _corr = (np.abs(_corr) > t).astype(int)
        _, _labels = connected_components(_corr)

        _, _count = np.unique(_labels, return_counts=True)
        _countnorm = _count / _count.sum()
        _entropy = -(_countnorm * np.log2(_countnorm)).sum()

        structural_entropy["timestamp"].append(df.columns[d])
        structural_entropy["structural_entropy"].append(_entropy)

    structural_entropy = pd.Series(structural_entropy["structural_entropy"],
                                   index=structural_entropy["timestamp"])
    return structural_entropy


def create_graph_corr(df, id_, sequence_length, thresh_cluster, thresh_edge):
    # Utility Function to Plot Correlation Network
    _corr = corr_matrix(df, id_ - sequence_length, id_)
    _pca = PCA(n_components=2, random_state=42).fit_transform(_corr)

    clusters = (np.abs(_corr) >= thresh_cluster).astype(int)
    _, _labels = connected_components(clusters)

    results = dict()

    results["edges"] = [(x, y) for x, y in zip(*np.where(np.abs(_corr) >= thresh_edge))]
    results["pos"] = {i: (_pca[i, 0], _pca[i, 1]) for i in range(len(_labels))}
    results["node_color"] = _labels
    results["nodelist"] = range(len(_labels))

    return results


# ----- Get RUSSELL 2000 Stock Tickers ----- #
# url = "https://bullishbears.com/russell-2000-stocks-list/"
# req = requests.get(url)
# soup = BeautifulSoup(req.text, "html.parser")
#
# table_ac = soup.find_all("table")[0]
# table_dk = soup.find_all("table")[2]
# table_lr = soup.find_all("table")[4]
# table_sz = soup.find_all("table")[6]
#
# td_0 = [i for i in table_ac]
# td_1 = [i for i in table_dk]
# td_2 = [i for i in table_lr]
# td_3 = [i for i in table_sz]
#
# td_0 = list(chain(*td_0))
# td_1 = list(chain(*td_1))
# td_2 = list(chain(*td_2))
# td_3 = list(chain(*td_3))
#
# stocks = []
# for i, j, k, l in zip(td_0, td_1, td_2, td_3):
#     ticker_0 = re.sub("<[^<]+?>", '', str(i))
#     ticker_1 = re.sub("<[^<]+?>", '', str(j))
#     ticker_2 = re.sub("<[^<]+?>", '', str(k))
#     ticker_3 = re.sub("<[^<]+?>", '', str(l))
#     stocks.append(ticker_0)
#     stocks.append(ticker_1)
#     stocks.append(ticker_2)
#     stocks.append(ticker_3)
#
# stocks.sort()
# tickers = [i.replace(".", "-") for i in stocks]
# print(tickers)

# ----- Get Stock Data ----- #
# start, end = datetime(2000, 1, 1), datetime(2020, 1, 1)
# master_df = pdr.DataReader(tickers, "yahoo", start=start, end=end).drop("Close", axis="columns").rename(columns=
#                                                                                                         {"Open": "open",
#                                                                                                          "Low": "low",
#                                                                                                          "High": "high",
#                                                                                                          "Adj Close": "close",
#                                                                                                          "Volume": "volume"})

DATA_STORE = Path("C:\\Users\\domin\\PycharmProjects\\Miscellaneous\\russell2000.h5")
# with pd.HDFStore(DATA_STORE, "w") as store:
#     store.put("RUSSELL2000/stocks", master_df)

# ----- Retrieve Stock Data ----- #
with pd.HDFStore(DATA_STORE, "r") as store:
    master_df = store.get("RUSSELL2000/stocks")

# A lot of failure with reading the symbols from Yahoo and converted to Nan
# Therefore, need to delete columns based on a null percentage
percentage = float(0.9 * 100)
min_count = int(((100 - percentage) / 100) * master_df.shape[0] + 1)
master_df = master_df.dropna(axis=1, thresh=min_count)

master_df = master_df[master_df.index >= "2009-01-01"]
percentage = float(0.2 * 100)
min_count = int(((100 - percentage) / 100) * master_df.shape[0] + 1)
master_df = master_df.dropna(axis=1, thresh=min_count)

master_df.dropna(axis=1, inplace=True)
idx = pd.IndexSlice

master_df = master_df.stack().swaplevel(0)
master_df = master_df.loc[idx[:, :], "close"].unstack("Date")
print(master_df)

master_df = master_df.sample(n=30, replace=False, random_state=42)
print(master_df.info())

# Create Log Returns - makes stationary and normally distributed
log_returns = np.log(master_df.copy().T).diff().dropna().T
print(log_returns)

# Define window length as a hyperparameter
seq_len = 200

# Plot Raw Log Return Statistics
fig = plt.figure(figsize=(10, 8))
sns.heatmap(log_returns.T.corr())
plt.xticks(range(log_returns.shape[0]), log_returns.index, rotation=90)
plt.yticks(range(log_returns.shape[0]), log_returns.index)
plt.show()

log_returns.T.plot.hist(bins=100)
plt.title("logarithmic returns distributions")
plt.show()

# Plot Log Return Sliding Statistics
fig = plt.figure(figsize=(18, 6), dpi=120)
plt.subplot(121)
log_returns.T.rolling(seq_len).mean().plot(legend=False, c="b", alpha=0.3, ax=plt.gca(), title="logret sliding mean")
log_returns.T.rolling(seq_len).mean().median(axis=1).plot(c="r", lw=3., ax=plt.gca())

plt.subplot(122)
log_returns.T.rolling(seq_len).std().plot(legend=False, c="b", alpha=0.3, ax=plt.gca(), title="logret sliding std")
log_returns.T.rolling(seq_len).std().median(axis=1).plot(c="r", lw=3., ax=plt.gca())
plt.show()

# ----- Structural Entropy ----- #
# Calculate Structural Entropy with Various Threshold to create Adjacent Matrixes
structural_entropy_03 = structural_entropy(df=log_returns, sequence_length=seq_len, t=0.3)
structural_entropy_05 = structural_entropy(df=log_returns, sequence_length=seq_len, t=0.5)
structural_entropy_06 = structural_entropy(df=log_returns, sequence_length=seq_len, t=0.6)
structural_entropy_07 = structural_entropy(df=log_returns, sequence_length=seq_len, t=0.7)
structural_entropy_08 = structural_entropy(df=log_returns, sequence_length=seq_len, t=0.8)

# Plot the structural entropy with various threshold to create adjacent matrices
plt.figure(figsize=(14, 6))
structural_entropy_03.plot(label="TH: 0.3")
structural_entropy_05.plot(label="TH: 0.5")
structural_entropy_06.plot(label="TH: 0.6")
structural_entropy_07.plot(label="TH: 0.7")
structural_entropy_08.plot(label="TH: 0.8")

plt.ylabel("structural entropy");
plt.legend(loc="best")
plt.show()

# Explore Edge Cases in Structural Entropy Calculation
reference_entropy = structural_entropy_06.copy()

id_max = np.random.choice(
    np.where(reference_entropy == reference_entropy.max())[0]
) + seq_len

id_mean = np.random.choice(
    np.where(reference_entropy.round(1) == round((reference_entropy.max() +
                                                  reference_entropy.min()) / 2, 1))[0]
)

id_min = np.random.choice(
    np.where(reference_entropy == reference_entropy.min())[0]
) + seq_len

# log_returns.columns[id_min], log_returns.columns[id_mean], log_returns[id_max]

# Compare Structural Entropy and Volatility
plt.figure(figsize=(14, 6))

reference_entropy.plot(label="entropy", c="orange")
plt.ylabel("structural entropy")
plt.legend(loc="upper right")
plt.twinx()
log_returns.T.rolling(seq_len).std().median(axis=1).plot(label="std", c="r")
plt.ylabel("standard deviation")

plt.legend(loc="lower right")
plt.show()

# Use our Utility Function to Plot Correlation Network
# Plot A Network with Maximum Structural Entropy
graph_param = create_graph_corr(df=log_returns, id_=id_max, sequence_length=seq_len, thresh_cluster=0.6,
                                thresh_edge=0.7)
G = nx.Graph()
G.add_edges_from(graph_param["edges"])
del graph_param["edges"]

plt.figure(figsize=(8, 6))
nx.draw_networkx(G, **graph_param, cmap="plasma")
plt.title("Max Structural Entropy")
plt.show()

# Plot a Network with Medium Structural Entropy
graph_param = create_graph_corr(df=log_returns, id_=id_mean, sequence_length=seq_len,
                                thresh_cluster=0.6, thresh_edge=0.7)
G = nx.Graph()
G.add_edges_from(graph_param['edges'])
del graph_param['edges']

plt.figure(figsize=(8, 6))
nx.draw_networkx(G, **graph_param, cmap='plasma')
plt.title('medium structural entropy')
plt.show()

# Plot a Network with Min Structural Entropy
graph_param = create_graph_corr(df=log_returns, id_=id_min, sequence_length=seq_len,
                                thresh_cluster=0.6, thresh_edge=0.7)
G = nx.Graph()
G.add_edges_from(graph_param['edges'])
del graph_param['edges']

plt.figure(figsize=(8, 6))
nx.draw_networkx(G, **graph_param, cmap='plasma')
plt.title('minimum structural entropy')
plt.show()
