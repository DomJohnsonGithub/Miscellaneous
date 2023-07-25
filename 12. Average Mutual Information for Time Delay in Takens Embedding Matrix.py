import pandas as pd
import numpy as np
from scipy.signal import argrelmin
from sklearn.metrics import mutual_info_score
from tqdm import tqdm


class AmiOptimalTimeDelay:
    def __init__(self, lags):
        self.lags = lags
        self.bins = None
        self.average_mi = None
        self.first_minima = None

    def fit(self, data):
        dataframe = pd.DataFrame(data)
        for i in range(1, self.lags + 1):
            dataframe[f"Lag_{i}"] = dataframe.iloc[:, 0].shift(i)
        dataframe.dropna(inplace=True)

        self.bins = int(np.round(2 * (len(dataframe)) ** (1 / 3), 0))

        def calc_mi(x, y, bins):
            c_xy = np.histogram2d(x, y, bins)[0]
            mi = mutual_info_score(None, None, contingency=c_xy)
            return mi

        def mutual_information(dataframe, lags, bins):
            mutual_information = []
            for i in tqdm(range(1, lags + 1)):
                mutual_information.append(calc_mi(dataframe.iloc[:, 0], dataframe[f"Lag_{i}"], bins=bins))

            return np.array(mutual_information)

        self.average_mi = mutual_information(dataframe, self.lags, self.bins)
        self.first_minima = argrelmin(self.average_mi)[0][0]

    def get_average_mi(self):
        return self.average_mi

    def get_first_minima(self):
        return self.first_minima
