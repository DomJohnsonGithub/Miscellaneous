import pandas as pd
import numpy as np
from fbm import MBM
import math
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import interpolate
from fractalmarkets.mmar.brownian_motion_multifractal_time import BrownianMotionMultifractalTime

# fractal markets package
bmmt = BrownianMotionMultifractalTime(9, x=0.35, y=0.68, randomize_segments=True, randomize_time=True, M=[0.6, 0.4])
data = bmmt.simulate()  # [ [x, y], ..., [x_n, y_n]]
f = interpolate.interp1d(data[:, 0], data[:, 1])
y = f(np.arange(0, 1, .0005))
x = pd.date_range(datetime(2010, 1, 1), periods=len(y))
df = pd.DataFrame(y, index=x)

ipo = 50.00
df += ipo

fig, axes = plt.subplots(nrows=2, ncols=1)
axes[0].plot(df)
axes[1].plot(df.rolling(window=20).std().dropna())
plt.grid(True)
plt.show()


# fbm package
def h(t):
    return 0.25 * math.sin(20 * t) + 0.5


m = MBM(n=2000, hurst=h, length=1, method="riemannliouville")
mbm_dataframe = pd.DataFrame()
for i in range(10):
    mbm_dataframe[f"Sample_{i}"] = m.mbm()

ipo = 50.00  # arbitrary starting value
mbm_dataframe += ipo
datelist = pd.date_range(start=datetime(2010, 1, 1), periods=len(mbm_dataframe))
mbm_dataframe.index = datelist

fig, axes = plt.subplots(ncols=1, nrows=2)
axes[0].plot(mbm_dataframe)
axes[0].grid(True)
axes[1].plot(mbm_dataframe.rolling(window=10).std().dropna())
plt.grid(True)
plt.show()
