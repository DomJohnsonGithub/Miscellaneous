import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

df = yf.download("^GSPC", datetime(2000, 1, 1), datetime.now() - timedelta(1)).dropna()
df["returns"] = df.Close.pct_change()
df = df.dropna()

returns = df["returns"] * 100

counts1 = 0
counts2 = 0

counts3 = 0
counts4 = 0
for i in range(len(returns) - 3):
    first_rets = returns[i]
    second_rets = returns[i + 1]
    third_rets = returns[i + 2]

    if first_rets >= 0.01 and second_rets >= 0.01:
        if third_rets > 0:
            counts1 += 1
        else:
            counts2 += 1

    elif first_rets <= 0.01 and second_rets <= 0.01:
        if third_rets < 0:
            counts3 += 1
        else:
            counts4 += 1

probability_positive_returns = counts1 / (counts1 + counts2)
print(probability_positive_returns)

probability_negative_returns = counts3 / (counts3 + counts4)
print(probability_negative_returns)

