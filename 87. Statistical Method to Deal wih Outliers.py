import pandas as pd
import numpy as np
import talib as ta


class OutlierHandler:
    def __init__(self, lookback, n, method):
        self.lookback = lookback
        self.n = n
        self.method = method
        self.ma = None

    def fit_transform(self, data):
        ma = pd.DataFrame(index=data.index)
        for i, j in data.items():
            ma[f"{i}"] = ta.SMA(j.values, timeperiod=self.lookback)

        res = data - ma

        Q1 = res.quantile(0.25)
        Q3 = res.quantile(0.75)
        IQR = Q3 - Q1

        lw_bound = Q1 - (self.n * IQR)
        up_bound = Q3 + (self.n * IQR)

        res[res <= lw_bound] = np.nan
        res[res >= up_bound] = np.nan

        res = res.interpolate(method=self.method)

        df = pd.DataFrame((res + ma))
        df.dropna(inplace=True)

        self.ma = ma

        return df

    def get_moving_averages(self):
        return self.ma