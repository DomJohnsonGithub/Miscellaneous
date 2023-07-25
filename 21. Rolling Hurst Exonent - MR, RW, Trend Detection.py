import numpy as np
import pandas as pd
from pathlib import Path
from hurst import compute_Hc as hc


def calculate_rolling_hurst(data, window_size):
    hurst_values = np.zeros(len(data))

    for i in range(window_size, len(data)):
        hurst_values[i] = hc(data.iloc[i - window_size:i], kind="price")[0]

    return hurst_values


DATA_SOURCE = Path("C:\\Users\\domin\\PycharmProjects\\Miscellaneous\\ASSETS.h5")
with pd.HDFStore(DATA_SOURCE, "r") as store:
    prices = store.get("ASSET/data")

idx = pd.IndexSlice
prices = prices.loc[idx[:, "KO"], ["Close", "Volume"]].droplevel(level=1)

prices["HURST_EXPONENT"] = 0
prices = prices.drop(columns={"Volume"})
print(prices)

# Rolling Hurst Exponent
prices["HURST_EXPONENT"] = calculate_rolling_hurst(prices["Close"], window_size=100)
print(prices)
