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
from tqdm import tqdm
from scipy.special import logit, expit
from sklearn.metrics import mean_squared_error

sns.set_style("darkgrid")


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def viz_results(y_obs, y_pred, title):
    fig, axes = plt.subplots(nrows=2, ncols=1, dpi=50, sharex=True)
    axes[0].plot(y_pred, c="red", label="Predicted")
    axes[0].plot(y_obs, c="blue", label="Actual")
    axes[0].legend(loc="best")
    axes[1].plot(y_obs - y_pred, c="seagreen", label="Residual")
    axes[1].legend(loc="best")
    plt.suptitle(f"{title}")
    plt.show()


class GBM:
    """
    Geometric Brownian Motion
    """
    def simulate(self):
        while (self.total_time > 0):
            dS = self.current_price * self.drift * self.time_period + self.current_price * self.volatility * np.random.normal(
                0, math.sqrt(self.time_period))
            self.prices.append(self.current_price + dS)
            self.current_price += dS
            self.total_time -= self.time_period

    def __init__(self, initial_price, drift, volatility, time_period, total_time):
        # Initialize fields
        self.initial_price = initial_price
        self.current_price = initial_price
        self.drift = drift
        self.volatility = volatility
        self.time_period = time_period
        self.total_time = total_time
        self.prices = []
        # Simulate the diffusion process
        self.simulate()  # Simulate the diffusion proces


class ProcessCEV:
    """
    Constant Elasticity of Variance
    """
    def __init__(self, mu, sigma, gamma):
        self._mu = mu
        self._sigma = sigma
        self._gamma = gamma

    def Simulate(self, T=1, dt=0.001, S0=1.):
        n = round(T / dt)

        mu = self._mu
        sigma = self._sigma
        gamma = self._gamma

        gaussian_increments = np.random.normal(size=n - 1)
        res = np.zeros(n)
        res[0] = S0
        S = S0
        sqrt_dt = dt ** 0.5
        for i in range(n - 1):
            S = S + S * mu * dt + sigma * \
                (S ** gamma) * gaussian_increments[i] * sqrt_dt
            res[i + 1] = S

        return res


def generate_heston_paths(S, T, r, kappa, theta, v_0, rho, xi,
                          steps, Npaths, return_vol=False):
    dt = T / steps
    size = (Npaths, steps)
    prices = np.zeros(size)
    sigs = np.zeros(size)
    S_t = S
    v_t = v_0
    for t in range(steps):
        WT = np.random.multivariate_normal(np.array([0, 0]),
                                           cov=np.array([[1, rho],
                                                         [rho, 1]]),
                                           size=paths) * np.sqrt(dt)

        S_t = S_t * (np.exp((r - 0.5 * v_t) * dt + np.sqrt(v_t) * WT[:, 0]))
        v_t = np.abs(v_t + kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * WT[:, 1])
        prices[:, t] = S_t
        sigs[:, t] = v_t

    if return_vol:
        return prices, sigs

    return prices



stock = "GS"
df = yf.download(
    stock,
    start=datetime(2000, 1, 1),
    end=datetime.now() - timedelta(1)).drop(
    columns=["Adj Close"])

train = df.iloc[:int(len(df) * 0.8), :]
test = df.iloc[int(len(df) * 0.8):, :]

data = [train, test]
for dataframe in data:
    dataframe["returns"] = np.log1p(dataframe.Close.pct_change())
    dataframe["target"] = np.sign(dataframe["returns"].shift(-1))
    dataframe["target"] = [clamp(i, 1e-5, 1 - 1e-5) for i in dataframe.target]
    dataframe["next_day"] = dataframe.Close.shift(-1)
    dataframe.dropna(inplace=True)
    dataframe["target_log_odds"] = logit(dataframe.target)

train, test = data[0], data[1]

X_train, y_train = train.Close, train.next_day
X_test, y_test = test.Close, test.next_day

# Geometric Brownian Motion
n = 1000
drift = 0.001
volatility = 0.01
time_period = 1
total_time = 1

# Attempt to predict using GBM Paths
sims = []
for i in tqdm(range(n)):
    simulations = []
    for j in range(0, len(X_test)):
        simulations.append(GBM(X_test[j], drift, volatility, time_period, total_time).prices)  # .iloc[j, 1]
    sims.append(np.array(simulations))

sims = np.reshape(np.array(sims), (n, 1183)).T

y_test = pd.DataFrame(y_test.values, index=y_test.index, columns=["Actual"])
for i in range(n):
    y_test[f"GBM_{i}"] = sims[:, i]
y_test["Mean_GBM"] = y_test.iloc[:, 1:].mean(axis=1)
y_test["Pred"] = y_test.Mean_GBM

viz_results(y_obs=y_test.Actual, y_pred=y_test.Pred, title="GBM")

print((y_test.Actual - y_test.Pred).describe())
mean_resid = (y_test.Actual - y_test.Pred).describe()[1]

# Attempt to improve by mean of residual
viz_results(y_obs=y_test.Actual, y_pred=(y_test.Pred + mean_resid), title="GBM with adding Mean Residual")

print("\nRMSE: ", np.sqrt(mean_squared_error(y_true=y_test.Actual, y_pred=y_test.Pred)))
print("RMSE corrected: ", np.sqrt(mean_squared_error(y_true=y_test.Actual, y_pred=y_test.Pred + mean_resid)))

plt.plot(y_test.iloc[:, 1:])
plt.title("GBM Paths")
plt.show()

# -------------------------------------------------
# Constant Elasticity of variance
n = 3
T = 1
dt = 1 / (len(X_test) + 1)
mu = 0.001
sigma = 0.01
gamma = 0.5

simulation = []
for i in range(len(X_test)):
    simulation.append(ProcessCEV(mu, sigma, gamma).Simulate(T, dt))
simulation = np.array(simulation).T

plt.plot(np.mean(simulation, axis=1)[1:] - 1 + X_test.values, c="green")
plt.plot(np.mean(simulation, axis=1)[1:] * X_test.values, c="yellow")
plt.plot(y_test.values, c="red")
plt.title("CEV Model")
plt.show()

print("RMSE1: ",
      np.sqrt(mean_squared_error(y_true=y_test.Actual.values, y_pred=np.mean(simulation, axis=1)[1:] - 1 + X_test.values)))
print("RMSE2: ",
      np.sqrt(mean_squared_error(y_true=y_test.Actual.values, y_pred=np.mean(simulation, axis=1)[1:] * X_test.values)))

hello = []
for j in range(len(X_test)):
    simulation = []
    for i in tqdm(range(n)):
        simulation.append(ProcessCEV(mu, sigma, gamma).Simulate(T, dt))
    simulation = np.mean(np.array(simulation).T[1])

    hello.append(X_test[j] * simulation)

plt.plot(hello, c="red")
plt.plot(y_test.values, c="blue")
plt.title("CEV Model Test")
plt.show()

print("RMSE: ", np.sqrt(mean_squared_error(y_test.Actual.values, hello)))

# --------------------------------------------------------------------
# Heston Model
kappa = 3
theta = 0.02
v_0 = 0.02
xi = 0.9
r = 0.02
S = X_test[0]
paths = len(X_test)
steps = 20000
T = 1

prices_pos = generate_heston_paths(S, T, r, kappa, theta,
                                   v_0, rho=0.9, xi=xi, steps=steps, Npaths=paths,
                                   return_vol=False)[:, -1]
prices_neg = generate_heston_paths(S, T, r, kappa, theta,
                                   v_0, rho=-0.9, xi=xi, steps=steps, Npaths=paths,
                                   return_vol=False)[:, -1]

plt.plot(prices_pos, c="red")
plt.plot(prices_neg, c="blue")
plt.plot(y_test.values, c="seagreen")
plt.show()


