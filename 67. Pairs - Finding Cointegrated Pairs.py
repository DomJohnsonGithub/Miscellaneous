import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
import seaborn
import yfinance as yf
from pandas_datareader import data

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

"""
We will load the financial data by using the panda
data reader. This time, we load many symbols at the same time. In this example,
we use SPY (this symbol reflects market movement), APPL (technology), ADBE
(technology), LUV (airlines), MSFT (technology), SKYW (airline industry), QCOM
(technology), HPQ (technology), JNPR (technology), AMD (technology), and IBM
(technology).

Since the goal of this trading strategy is to find co-integrated symbols, we narrow
down the search space according to industry. This function will load the data of a
file from the Yahoo finance website if the data is not in the
multi_data_large.pkl file:
"""

symbolsIds = ['SPY', 'AAPL', 'ADBE', 'LUV', 'MSFT', \
              'SKYW', 'QCOM',
              'HPQ', 'JNPR', 'AMD', 'IBM']


def load_financial_data(symbols, start_date, end_date, output_file):
    try:
        df = pd.read_pickle(output_file)
        print('File data found...reading symbols data')
    except FileNotFoundError:
        print('File not found...downloading the symbols data')
        df = data.DataReader(symbols, start_date, end_date)
        df.to_pickle(output_file)
    return df


data = load_financial_data(symbolsIds, start_date='2001-01-01',
                           end_date='2018-01-01',
                           output_file='multi_data_large.pkl')

"""
Create a function establishing cointegration between pairs, as
shown in the following code. This function takes as inputs a list of financial
instruments and calculates the cointegration values of these symbols. The values
are stored in a matrix. We will use this matrix to display a heatmap:
"""


def find_cointegrated_pairs(data):
    n = data.shape[1]
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            result = coint(data[keys[i]], data[keys[j]])
            pvalue_matrix[i, j] = result[1]
            if result[1] < 0.02:
                pairs.append((keys[i], keys[j]))
    return pvalue_matrix, pairs


"""
We call the load_financial_data function, we will then call the
find_cointegrated_pairs function:
"""
pvalues, pairs = find_cointegrated_pairs(data['Adj Close'])
print(pairs)

"""
We will use the seaborn package to draw the heatmap. The code calls the
heatmap function from the seaborn package. Heatmap will use the list of
symbols on the x and y axes. The last argument will mask the p-values higher
than 0.98:

This code will return the following map as an output. This map shows the pvalues of the return of the coin:
-If a p-value is lower than 0.02, this means the null hypothesis is rejected.
-This means that the two series of prices corresponding to two different
 symbols can be co-integrated.
-This means that the two symbols will keep the same spread on average. On
 the heatmap, we observe that the following symbols have p-values lower
 than 0.02:
"""
seaborn.heatmap(pvalues, xticklabels=symbolsIds,
                yticklabels=symbolsIds, cmap='RdYlGn_r',
                mask=(pvalues >= 0.98))
plt.show()
print(pairs)
"""
This screenshot represents the heatmap measuring the cointegration between a
pair of symbols. If it is red, this means that the p-value is 1, which means that the
null hypothesis is not rejected. Therefore, there is no significant evidence that the
pair of symbols is co-integrated. After selecting the pairs we will use for trading,
let's focus on how to trade these pairs of symbols.
"""

print(data.head(3))
