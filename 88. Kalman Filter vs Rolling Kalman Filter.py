import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from joblib import Parallel, delayed

sns.set_style("darkgrid")

df = yf.download("^GSPC", datetime(2000, 1, 1), datetime.now() - timedelta(1)).dropna()
print(df)


def kalman(data, trans_cov=0.01):
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=data[0],
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=trans_cov)

    return kf.filter(data)[0]


def kalman1(data, trans_cov=0.01, window_size=10):
    def kalman_filter_subsequence(sub_data, initial_state_mean):
        kf = KalmanFilter(transition_matrices=[1],
                          observation_matrices=[1],
                          initial_state_mean=initial_state_mean,
                          initial_state_covariance=1,
                          observation_covariance=1,
                          transition_covariance=trans_cov)

        return kf.filter(sub_data)[0][-1]

    results = Parallel(n_jobs=-1)(
        delayed(kalman_filter_subsequence)(data[i:i + window_size], data[i - 1])
        for i in range(0, len(data) - window_size + 1))

    return np.array((window_size - 1) * [np.nan] + list(np.concatenate(results)))


# Estimate hidden state mean
df["kalman_0.001"] = kalman(data=df.Close, trans_cov=0.001)
df["kalman_0.01"] = kalman(data=df.Close, trans_cov=0.01)
df["kalman_0.1"] = kalman(data=df.Close, trans_cov=0.1)
df["kalman_1"] = kalman(data=df.Close, trans_cov=1.0)

# Rolling KF
filtered_data = kalman1(df.Close, trans_cov=2, window_size=100)
df["roll_kf"] = filtered_data

plt.plot(df.Close, c="black", label="Close")
plt.plot(df["kalman_0.001"], c="purple", label="kf_0.001")
plt.plot(df["kalman_0.01"], c="orange", label="kf_0.01")
plt.plot(df["kalman_0.1"], c="red", label="kf_0.1")
plt.plot(df["kalman_1"], c="pink", label="kf_1")
plt.plot(df["roll_kf"], c="seagreen", label="roll_kf_2")
plt.legend(loc="best")
plt.show()
