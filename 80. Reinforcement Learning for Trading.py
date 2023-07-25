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

df["returns"] = np.log1p(df.Close.pct_change())
df.dropna(inplace=True)

import gym
import gym_anytrading
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.a2c import A2C
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
import quantstats as qs
from gym_anytrading.envs import StocksEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.dqn import DQN
from stable_baselines3.ddpg import DDPG
from stable_baselines3.sac import SAC
from stable_baselines3.td3 import TD3

data = df.iloc[:, :-1].copy()
X_train, X_test = data.iloc[:int(len(data) * 0.8), :], data.iloc[int(len(data) * 0.8):, :]

X = [X_train, X_test]
for dataframe in X:
    for i in [8, 14, 21]:
        dataframe.loc[:, f"RSI_{i}"] = ta.RSI(dataframe.loc[:, "Close"], timeperiod=i)
        dataframe.loc[:, f"ADX_{i}"] = ta.ADX(dataframe.loc[:, "High"], dataframe.loc[:, "Low"],
                                              dataframe.loc[:, "Close"], timeperiod=i)
        dataframe.loc[:, f"AROON)_{i}"] = ta.AROONOSC(dataframe.loc[:, "High"], dataframe.loc[:, "Low"], timeperiod=i)
        dataframe.loc[:, f"CCI_{i}"] = ta.ADX(dataframe.loc[:, "High"], dataframe.loc[:, "Low"],
                                              dataframe.loc[:, "Close"], timeperiod=i)
        dataframe[f"WILLR_{i}"] = ta.WILLR(dataframe.loc[:, "High"], dataframe.loc[:, "Low"], dataframe.loc[:, "Close"],
                                           timeperiod=i)
    for i in [1, 2, 3, 4, 5, 6, 10, 21, 42, 63]:
        dataframe[f"log_rets{i}d"] = np.log1p(dataframe.Close.pct_change(i))
    dataframe["rets"] = dataframe.Close.pct_change()
    dataframe["APO"] = ta.APO(dataframe.loc[:, "Close"], fastperiod=12, slowperiod=26, matype=0)
    dataframe["BOP"] = ta.BOP(dataframe.loc[:, "Open"], dataframe.loc[:, "High"], dataframe.loc[:, "Low"],
                              dataframe.loc[:, "Close"])
    dataframe["macd"], macdsignal, macdhist = ta.MACD(dataframe.loc[:, "Close"], fastperiod=12, slowperiod=26,
                                                      signalperiod=9)
    dataframe["PPO"] = ta.PPO(dataframe.loc[:, "Close"], fastperiod=12, slowperiod=26, matype=0)
    dataframe["ULTOSC"] = ta.ULTOSC(dataframe.loc[:, "High"], dataframe.loc[:, "Low"], dataframe.loc[:, "Close"],
                                    timeperiod1=7, timeperiod2=14, timeperiod3=28)
    for i in [1, 2, 3, 4, 5, 6, 10, 21, 42, 63]:
        dataframe.loc[:, f"MOM_{i}"] = ta.MOM(dataframe.loc[:, "Close"], timeperiod=i)

X0 = X[0]
X1 = X[1]

sc = StandardScaler()
X_train = pd.DataFrame(sc.fit_transform(X0), index=X0.index, columns=X0.columns)
X_test = pd.DataFrame(sc.transform(X1), index=X1.index, columns=X1.columns)
X_train.dropna(inplace=True)
X_test.dropna(inplace=True)

print(X_train.columns)


# Customised Data for the Model
def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, "Close"].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['High', 'Low', 'Open', 'Close', 'Volume', 'RSI_8', 'ADX_8', 'AROON)_8',
                                     'CCI_8', 'WILLR_8', 'RSI_14', 'ADX_14', 'AROON)_14', 'CCI_14',
                                     'WILLR_14', 'RSI_21', 'ADX_21', 'AROON)_21', 'CCI_21', 'WILLR_21',
                                     'log_rets1d', 'log_rets2d', 'log_rets3d', 'log_rets4d', 'log_rets5d',
                                     'log_rets6d', 'log_rets10d', 'log_rets21d', 'log_rets42d',
                                     'log_rets63d', 'rets', 'APO', 'BOP', 'macd', 'PPO', 'ULTOSC', 'MOM_1',
                                     'MOM_2', 'MOM_3', 'MOM_4', 'MOM_5', 'MOM_6', 'MOM_10', 'MOM_21',
                                     'MOM_42', 'MOM_63']].to_numpy()[start:end]
    return prices, signal_features


class MyCustomEnv(StocksEnv):
    _process_data = add_signals


# # -------------------------------------------
# Model Parameters
window_size = 10
start_index = window_size
end_index = len(X_train)

# Build Environment and Learn
env_maker = lambda: MyCustomEnv(df=X_train, window_size=window_size, frame_bound=(start_index, int(len(X_train)*0.8)))
env = DummyVecEnv([env_maker])

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10_000)
model.save(path="C:\\Users\\domin\\PycharmProjects\\Miscellaneous\\reinforcement_model")
# print(model.get_parameters())

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# Evaluate Model
env = MyCustomEnv(df=X_train, window_size=window_size, frame_bound=(int(len(X_train)*0.8), end_index))
observation = env.reset()

actions = []
rewards = []
while True:
    observation = observation[np.newaxis, ...]
    action, _states = model.predict(observation)
    print(action)
    actions.append(action)
    observation, reward, done, info = env.step(action)
    rewards.append(reward)

    if done:
        print("info:", info)
        break

actions = np.array(actions)
rewards = np.array(rewards)
print(env.signal_features)

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, dpi=50)
axes[0].plot(rewards)
axes[1].plot(np.cumsum(rewards))
plt.show()

plt.figure(figsize=(16, 6))
env.render_all()
plt.show()

qs.extend_pandas()

net_worth = pd.Series(env.history['total_profit'], index=X_train.index[int(len(X_train)*0.8)+1:len(X_train)])
returns = net_worth.pct_change().iloc[1:]

qs.reports.full(returns, match_dates=True)
qs.reports.html(returns, output='C:\\Users\\domin\\PycharmProjects\\Miscellaneous\\a2c_quantstats.html')

plt.plot(np.cumsum(returns), c="red")
plt.plot(np.cumprod(1 + returns) - 1, c="blue")
plt.show()
# # -------------------------------------------

# Multi-Models #
# Model Parameters
window_size = 50
start_index = window_size
end_index = len(X_train)

# Build Environment and Learn
env_maker = lambda: MyCustomEnv(df=X_train, window_size=window_size, frame_bound=(start_index, end_index))
env = DummyVecEnv([env_maker])

model1 = A2C('MlpPolicy', env, verbose=1)
model1.learn(total_timesteps=10_000)

model2 = PPO("MlpPolicy", env, verbose=1)
model2.learn(total_timesteps=10_000)

model3 = DQN("MlpPolicy", env, verbose=1)
model3.learn(total_timesteps=10_000)

# Evaluate Model
env = MyCustomEnv(df=X_train, window_size=window_size, frame_bound=(int(len(X_train) * 0.8), end_index))
observation = env.reset()

from collections import Counter


def find_majority(votes):
    vote_count = Counter(votes)
    top_two = vote_count.most_common(2)
    if len(top_two) > 1 and top_two[0][1] == top_two[1][1]:
        # It is a tie
        return 0
    return top_two[0][0]


actions = []
rewards = []
while True:
    observation = observation[np.newaxis, ...]

    action1, _states1 = model1.predict(observation, )
    action2, _states2 = model2.predict(observation)
    action3, _states3 = model3.predict(observation)
    action = np.array([find_majority([np.asscalar(action1),
                                      np.asscalar(action2),
                                      np.asscalar(action3)
                                      ])])

    # actions.append([action1, action2])
    observation, reward, done, info = env.step(action)
    rewards.append(reward)

    if done:
        print("info:", info)
        break

actions = np.array(actions)
rewards = np.array(rewards)
print(env.signal_features)

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, dpi=50)
axes[0].plot(rewards)
axes[1].plot(np.cumsum(rewards))
plt.show()

plt.figure(figsize=(16, 6))
env.render_all()
plt.show()

qs.extend_pandas()

net_worth = pd.Series(env.history['total_profit'], index=X_train.index[int(len(X_train) * 0.8) + 1:len(X_train)])
returns = net_worth.pct_change().iloc[1:]

qs.reports.full(returns, match_dates=True)
qs.reports.html(returns, output='C:\\Users\\domin\\PycharmProjects\\Miscellaneous\\a2c_quantstats.html')

plt.plot(np.cumsum(returns), c="red")
plt.plot(np.cumprod(1 + returns) - 1, c="blue")
plt.show()
