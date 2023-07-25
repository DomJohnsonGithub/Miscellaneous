import numpy as np
import pandas as pd
from datetime import datetime
import pandas_datareader.data as pdr
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from tqdm import tqdm
from itertools import accumulate
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.isotonic import IsotonicRegression
from polyfit import PolynomRegressor, Constraints
from scipy.stats import mode
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import norm, uniform, kendalltau, t, genextreme, genlogistic, kstest, cauchy, laplace
import pickle
import statsmodels.api as sm
import yfinance as yf
sns.set_style("darkgrid")


# pickle_off = open ("datafile.pkl", "rb")
# df = pickle.load(pickle_off)
start, end = datetime(2012, 10, 1), datetime(2014, 5, 31)
df = yf.download(["ICICIBANK.NS", "AXISBANK.NS"], start, end)["Adj Close"]
df.dropna(inplace=True)
print(df)

df = np.log1p(df.pct_change())[1:]

plt.plot(np.cumsum(df.iloc[:, 0]), c="blue", label="ICICIBANK")
plt.plot(np.cumsum(df.iloc[:, 1]), c="red", label="AXISBANK")
plt.legend()
plt.show()

sc = StandardScaler()
norm_diff_cum_rets = np.cumsum(df.iloc[:, 1]) - np.cumsum(df.iloc[:, 0])
norm_diff_cum_rets = sc.fit_transform(np.reshape(norm_diff_cum_rets.values, (-1, 1)))
z_score = (norm_diff_cum_rets - np.mean(norm_diff_cum_rets))/np.std(norm_diff_cum_rets)
plt.plot(z_score, c="blue")
plt.show()

# a better spread was normalize each cumsum before then take difference

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principal_component = pca.fit_transform(sc.fit_transform(df))
print(pca.explained_variance_ratio_)


X = np.linspace(-0.1, 0.1)
Y1 = X*pca.components_[0, 0]
Y2 = -X*pca.components_[0, 1]
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c="blue", s=10)
plt.plot(X, Y1, c="red")
plt.plot(X, Y2, c="lightgreen")
plt.show()

np.set_printoptions(formatter={'float_kind':'{:f}'.format})
print(sm.tsa.acf(principal_component[:, 0]))

pc1 = pd.DataFrame((principal_component[:, 0] - np.mean(principal_component[:, 0])) / np.std(principal_component[:, 0]), index=df.index, columns=["Zscore_PC1"])
pc1["Signal"] = pc1.rolling(4).sum()
pc1.dropna(inplace=True)

entry_threshold = 4

plt.bar(np.array(pc1.index).flatten(), np.array(pc1["Zscore_PC1"]).flatten(), width=1, color="blue")
plt.plot(pc1.Signal, c="red")
plt.axhline(y=entry_threshold, c="lightgreen")
plt.axhline(y=-entry_threshold, c="lightgreen")
plt.axhline(y=np.mean(pc1["Zscore_PC1"]), c="purple")
plt.show()








