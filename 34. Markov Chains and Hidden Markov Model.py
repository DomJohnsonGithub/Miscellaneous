from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pandas_datareader.data as pdr
import talib as ta
from matplotlib.widgets import MultiCursor
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
import yfinance as yf
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

# SPY ETF replicates the S&P500
ticker, source, start, end = "EA", "yahoo", datetime(2000, 1, 1), datetime.now()
df = yf.download(ticker, start, end).drop(columns=["Adj Close"]).dropna()
df["Returns"] = df.Close.pct_change()  # Daily Returns
df.dropna(inplace=True)

# MARKOV CHAIN
# Identify the states we want to model. Price moves up, down or is unchanged
df["State"] = df.Returns.apply(lambda x: "Up" if (x > 0.001) else ("Down" if (x < -0.001) else "Flat"))

# Analyze the Transitions in the prior day's price to today's price
df["PriorState"] = df.State.shift(1)

# Build Frequency Distribution Matrix  -  freq. distrb'n of th transitions
states = df[["PriorState", "State"]].dropna()
states_mat = states.groupby(["PriorState", "State"]).size().unstack()

print("------------------------------")
print("Frequency Distribution Matrix")
print(states_mat)

# Build Initial Probability Matrix or Transition Matrix at time t0
transition_matrix = states_mat.apply(lambda x: x / float(x.sum()), axis=1)
print("------------------------------")
print("Initial Transition/Probability Matrix")
print(transition_matrix)

# Build Markov Chain by multiplying transition matrix by itself to obtain the probability matrix in t1 which allows
# us to make one-day forecasts
t0 = transition_matrix.copy()
t1 = round(t0.dot(t0), 4)
print("------------------------------")
print("Probability Matrix in t1")
print(t1)

# Obtaining the next probabilities at time t2
t2 = round(t0.dot(t1), 4)
print("------------------------------")
print("Probability Matrix in t2")
print(t2)

# Obtaining the next probabilities at time t3 and so on until we find the equilibrium matrix where the
# probabilities do not change and therefore we cannot continue evolving the prediction.
t3 = round(t0.dot(t2), 4)
print("------------------------------")
print("Probability Matrix in t3")
print(t3)

# Interestingly can get same results by raising the initial transition matrix to "n" days
print("------------------------------")
print("Probability Matrix in t3 using Linear Algebra Matrix Power  (same as above ^)")
print(pd.DataFrame(np.linalg.matrix_power(t0, 4)))

# Finding the Equilibrium Matrix
# iterative process until probabilities do NOT change
print("------------------------------")
print("Finding equilibrium matrix")

i = 1
a = t0.copy()
b = t0.dot(t0)
while not (a.equals(b)):
    print("Iteration numbers: " + str(i))
    i += 1
    a = b.copy()
    b = b.dot(t0)

print("")
print("Equilibrium Matrix")
print(np.round(b, 4))

# HIDDEN MARKOV MODEL
sc = StandardScaler()
input_data = sc.fit_transform(np.reshape(df.Returns.values, (-1, 1)))

hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=10_000)
hmm_model.fit(input_data)
hidden_states = hmm_model.predict(input_data)
print(hidden_states)

print("------------------------------")
print("Hidden Markov Model Transition Matrix")
print("")
print(hmm_model.transmat_)

print("")
print("Mean and Variances of each Hidden State")
for i in range(hmm_model.n_components):
    print(f"{i}th Hidden State")
    print("Mean = ", hmm_model.means_[i])
    print("Var = ", np.diag(hmm_model.covars_[i]))

fig, axes = plt.subplots(hmm_model.n_components, sharex=True, sharey=True)
colours = cm.rainbow(np.linspace(0, 1, hmm_model.n_components))
for i, (ax, colour) in enumerate(zip(axes, colours)):
    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
    ax.plot_date(df.index[mask], df.Close[mask], ".-", c=colour)
    ax.set_title("{0}th hidden state".format(i))

    # Format the ticks.
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())

    ax.grid(True)

plt.show()

# Sampling from our HMM
n = 100
sample = np.zeros((n, 1))

# Get the parameters of the last observed hidden state
last_state_params = hmm_model.means_[hidden_states[-1]], hmm_model.covars_[hidden_states[-1]]

# Generate future returns based on the last state parameters
for i in range(n):
    sample[i] = np.random.normal(last_state_params[0], np.sqrt(last_state_params[1]))
print(sample)

