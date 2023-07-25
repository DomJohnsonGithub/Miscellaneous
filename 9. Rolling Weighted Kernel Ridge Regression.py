import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import time
import pymysql
from sqlalchemy import create_engine
import talib as ta
import warnings

sns.set_style("darkgrid")
idx = pd.IndexSlice

tickers = sorted(
    {"TSCO.L", "JD.L", "GSK.L", "RR.L", "BATS.L", "BA.L", "UU.L", "SSE.L", "SPX.L", "SMT.L", "NXT.L", "SDR.L", "AAL.L",
     "ABF.L", "BARC.L", "VOD.L", "UTG.L", "STAN.L", "SMIN.L", "SGRO.L", "SBRY.L", "BKG.L", "BP.L", "HSX.L", "SGE.L",
     "RIO.L", "REL.L", "RKT.L", "PRU.L", "PSN.L", "NWG.L", "INF.L", "KGF.L", "PSON.L", "WPP.L", "WTB.L", "WEIR.L",
     "STJ.L", "SN.L", "SVT.L", "RTO.L", "NG.L", "LLOY.L", "LGEN.L", "LAND.L", "KGF.L", "JMAT.L", "INF.L", "IMB.L",
     "HSBA.L", "HLMA.L", "DGE.L", "DCC.L", "CRDA.L", "CPG.L", "CNA.L", "BNZL.L", "BP.L", "BKG.L", "BDEV.L", "BARC.L",
     "AZN.L", "ABF.L", "AHT.L", "ANTO.L", "III.L"})

contract_size_100 = sorted(["NWG.L", "BARC.L", "III.L", "BA.L", "BP.L", "BDEV.L", "SBRY.L", "STAN.L",
                            "TSCO.L", "VOD.L", "RR.L", "GSK.L", "HSBA.L", "LGEN.L", "LLOY.L", "NXT.L", "PRU.L"])
contract_size_10 = sorted(list(set(tickers) - set(contract_size_100)))

four_dec_places = ["BP.L", "BARC.L", "HSBA.L", "LLOY.L", "TSCO.L", "VOD.L"]
three_dec_places = ["BA.L", "BDEV.L", "GSK.L", "LGEN.L", "PRU.L", "RR.L",
                    "SBRY.L", "III.L", "STAN.L", "RIO.L", "NG.L",
                    "WEIR.L", "SPX.L", "HSX.L", "HLMA.L", "NWG.L", "BKG.L",
                    "KGF.L", "CNA.L", "ANTO.L", "CPG.L", "REL.L", "BATS.L", "UTG.L", "JD.L", "DGE.L"]
two_dec_places = ["NXT.L", "AHT.L", "BNZL.L", "CRDA.L", "DCC.L", "INF.L", "PSON.L", "PSN.L", "RTO.L",
                  "SSE.L", "SGE.L", "UU.L", "SMIN.L", "SN.L", "SVT.L", "SMT.L", "RKT.L", "LAND.L",
                  "JMAT.L", "IMB.L", "ABF.L", "AAL.L", "WTB.L", "WPP.L", "SDR.L", "STJ.L", "AZN.L",
                  "SGRO.L"]

user = "root"
password = "Dom23129"
host = '127.0.0.1'
db = 'av_dataframe'

ALPHA_VANTAGE_API_KEY = "C9KLZ7L6S0BUWEW6"

# data_obj = ALPHA_VANTAGE_DATA(symbols=tickers, api_key=ALPHA_VANTAGE_API_KEY)
# data_obj.retrieve_data()
# combined_df = data_obj.combine_dataframes()
#
# # Manipulate MultiIndex for MySQL
# df = combined_df.dropna().swaplevel(0).reset_index()

# Store in MySQL
connection = pymysql.connect(host=host, user=user, password=password, db=db)
engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{db}')
# df.to_sql(name='av_ftse_stocks', con=engine, index=True, if_exists='replace')
df = pd.read_sql("SELECT * FROM av_ftse_stocks", connection).set_index(["Ticker", "Date"]).drop(columns="index")
connection.close()

# Change values for CFD Trading
df.loc[contract_size_100 + contract_size_10, ["Open", "High", "Low", "Close"]] /= 100
df.loc[contract_size_100 + contract_size_10, "Volume"] /= 10e5
df.loc[idx[four_dec_places, :], ["Open", "High", "Low", "Close"]] = df.loc[
    idx[four_dec_places, :], ["Open", "High", "Low", "Close"]].round(4)
df.loc[idx[three_dec_places, :], ["Open", "High", "Low", "Close"]] = df.loc[
    idx[three_dec_places, :], ["Open", "High", "Low", "Close"]].round(3)
df.loc[idx[two_dec_places, :], ["Open", "High", "Low", "Close"]] = df.loc[
    idx[two_dec_places, :], ["Open", "High", "Low", "Close"]].round(2)

aal = df.loc[idx["AAL.L", :], :].droplevel(0)

aal["returns"] = np.sign(aal.Close.pct_change())
aal["ema_50"] = np.sign(ta.EMA(aal.Close, timeperiod=50).diff())
aal["std"] = np.sign(aal.returns.rolling(20).std().diff())
aal["rsi"] = np.sign(ta.RSI(aal.Close, timeperiod=5).diff())
aal["volume"] = np.sign(ta.EMA(aal.Volume, timeperiod=10).diff())
aal["skew"] = np.sign(aal.returns.rolling(20).skew().diff())
aal["kurt"] = np.sign(aal.returns.rolling(20).kurt().diff())

aal = aal.drop(columns=["Open", "High", "Low", "Volume", "Close"])

aal = aal.resample("5B").last()

aal["target"] = np.where(aal["returns"].shift(-1) > 0, 1, -1)


aal = aal.iloc[:-1, :].dropna()

print(aal)
print(pd.DataFrame(aal).corr())

X_train, y_train = aal.drop(columns=["target"]).iloc[:int(0.75 * len(aal)), :], aal.iloc[:int(0.75 * len(aal)), 7]
X_test, y_test = aal.drop(columns=["target"]).iloc[int(0.75 * len(aal)):, :], aal.iloc[int(0.75 * len(aal)):, 7]

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import mode
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]


def fitGNB(X, y):
    svc = SVC(shrinking=False, class_weight="balanced", C=1)
    rf = RandomForestClassifier(n_jobs=11, class_weight="balanced_subsample")
    cnb = GaussianNB()
    lr = LogisticRegression(fit_intercept=True, class_weight="balanced", max_iter=10_000, n_jobs=11, solver="saga")
    knn = KNeighborsClassifier(n_jobs=11, weights="distance", n_neighbors=11)

    svc.fit(X, y)
    rf.fit(X, y)
    cnb.fit(X, y)
    lr.fit(X, y)
    knn.fit(X, y)

    return svc, rf, cnb, lr, knn


splitter = BlockingTimeSeriesSplit(n_splits=3)

models = []
scs = []
for train_index, test_index in splitter.split(X_train):
    X_train1 = X_train.iloc[train_index, :]
    y_train1 = y_train[train_index]
    sc = StandardScaler()
    models.append(fitGNB(X_train1, y_train1))

preds = np.array([sub_model.predict(X_test) for model in models for sub_model in model]).T
modes = np.array([mode(preds[i])[0][0] for i in range(np.shape(preds)[0])])
print(accuracy_score(y_test, modes))

plt.plot(y_test.values)
plt.plot(modes)
plt.show()







