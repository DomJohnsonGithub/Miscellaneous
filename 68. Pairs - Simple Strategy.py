import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt

# Set a seed value to make the experience reproducible
np.random.seed(123)

import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
import seaborn

from pandas_datareader import data

symbolsIds = ['SPY', 'AAPL', 'ADBE', 'LUV', 'MSFT', 'SKYW', 'QCOM',
              'HPQ', 'JNPR', 'AMD', 'IBM']


def load_financial_data(symbols, start_date, end_date, output_file):
    try:
        df = pd.read_pickle(output_file)
        print('File data found...reading symbols data')
    except FileNotFoundError:
        print('File not found...downloading the symbols data')
        df = data.DataReader(symbols, 'yahoo', start_date, end_date)
        df.to_pickle(output_file)
    return df


data = load_financial_data(symbolsIds, start_date='2001-01-01',
                           end_date='2018-01-01',
                           output_file='multi_data_large.pkl')

"""
We will use MSFT and JNPR to implement the strategy based on real symbols. We
will replace the code to build Symbol 1 and Symbol 2 with the following code.
The following code will get the real prices for MSFT and JNPR:

The following screenshot shows the MSFT and JNPR prices. We observe
similarities of movement between the two symbols:
"""
Symbol1_prices = data['Adj Close']['MSFT']
Symbol1_prices.plot(figsize=(15, 7))
plt.show()
Symbol2_prices = data['Adj Close']['JNPR']
Symbol2_prices.name = 'JNPR'
plt.title("MSFT and JNPR prices")
Symbol1_prices.plot()
Symbol2_prices.plot()
plt.legend()
plt.show()


def zscore(series):
    return (series - series.mean()) / np.std(series)


score, pvalue, _ = coint(Symbol1_prices, Symbol2_prices)
print(pvalue)
ratios = Symbol1_prices / Symbol2_prices
plt.title("Ration between Symbol 1 and Symbol 2 price")

ratios.plot()
plt.show()

# plt.axhline(ratios.mean())
# plt.legend([' Ratio'])


zscore(ratios).plot()
plt.title("Z-score evolution")
plt.axhline(zscore(ratios).mean(), color="black")
plt.axhline(1.0, color="red")
plt.axhline(-1.0, color="green")
plt.show()

ratios.plot()
buy = ratios.copy()
sell = ratios.copy()
buy[zscore(ratios) > -1] = 0
sell[zscore(ratios) < 1] = 0
buy.plot(color="g", linestyle="None", marker="^")
sell.plot(color="r", linestyle="None", marker="v")
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, ratios.min(), ratios.max()))
plt.legend(["Ratio", "Buy Signal", "Sell Signal"])
plt.show()

symbol1_buy = Symbol1_prices.copy()
symbol1_sell = Symbol1_prices.copy()
symbol2_buy = Symbol2_prices.copy()
symbol2_sell = Symbol2_prices.copy()

Symbol1_prices.plot()
symbol1_buy[zscore(ratios) > -1] = 0
symbol1_sell[zscore(ratios) < 1] = 0
symbol1_buy.plot(color="g", linestyle="None", marker="^")
symbol1_sell.plot(color="r", linestyle="None", marker="v")

"""
We will create a data frame, pair_correlation_trading_strategy, in the
code. This contains information relating to orders and position and we will use
this data frame to calculate the performance of this pair correlation trading
strategy:
"""
pair_correlation_trading_strategy = pd.DataFrame(index=Symbol1_prices.index)
pair_correlation_trading_strategy['symbol1_price'] = Symbol1_prices
pair_correlation_trading_strategy['symbol1_buy'] = np.zeros(len(Symbol1_prices))
pair_correlation_trading_strategy['symbol1_sell'] = np.zeros(len(Symbol1_prices))
pair_correlation_trading_strategy['symbol2_buy'] = np.zeros(len(Symbol1_prices))
pair_correlation_trading_strategy['symbol2_sell'] = np.zeros(len(Symbol1_prices))

"""
We will limit the number of orders by reducing the position to one share. This
can be a long or short position. For a given symbol, when we have a long
position, a sell order is the only one that is allowed. When we have a short
position, a buy order is the only one that is allowed. When we have no position,
we can either go long (by buying) or go short (by selling). We will store the price
we use to send the orders. For the paired symbol, we will do the opposite. When
we sell Symbol 1, we will buy Symbol 2, and vice versa:
"""
position = 0
for i in range(len(Symbol1_prices)):
    s1price = Symbol1_prices[i]
    s2price = Symbol2_prices[i]
    if not position and symbol1_buy[i] != 0:
        pair_correlation_trading_strategy['symbol1_buy'][i] = s1price
        pair_correlation_trading_strategy['symbol2_sell'][i] = s2price
        position = 1
    elif not position and symbol1_sell[i] != 0:
        pair_correlation_trading_strategy['symbol1_sell'][i] = s1price
        pair_correlation_trading_strategy['symbol2_buy'][i] = s2price
        position = -1
    elif position == -1 and (symbol1_sell[i] == 0 or i == len(Symbol1_prices) - 1):
        pair_correlation_trading_strategy['symbol1_buy'][i] = s1price
        pair_correlation_trading_strategy['symbol2_sell'][i] = s2price
        position = 0
    elif position == 1 and (symbol1_buy[i] == 0 or i == len(Symbol1_prices) - 1):
        pair_correlation_trading_strategy['symbol1_sell'][i] = s1price
        pair_correlation_trading_strategy['symbol2_buy'][i] = s2price
        position = 0

Symbol2_prices.plot()
symbol2_buy[zscore(ratios) < 1] = 0
symbol2_sell[zscore(ratios) > -1] = 0
symbol2_buy.plot(color="g", linestyle="None", marker="^")
symbol2_sell.plot(color="r", linestyle="None", marker="v")

x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, Symbol1_prices.min(), Symbol2_prices.max()))
plt.legend(["Symbol1", "Buy Signal", "Sell Signal", "Symbol2"])
plt.show()

Symbol1_prices.plot()
pair_correlation_trading_strategy['symbol1_buy'].plot(color="g", linestyle="None", marker="^")
pair_correlation_trading_strategy['symbol1_sell'].plot(color="r", linestyle="None", marker="v")
Symbol2_prices.plot()
pair_correlation_trading_strategy['symbol2_buy'].plot(color="g", linestyle="None", marker="^")
pair_correlation_trading_strategy['symbol2_sell'].plot(color="r", linestyle="None", marker="v")
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, Symbol1_prices.min(), Symbol2_prices.max()))
plt.legend(["Symbol1", "Buy Signal", "Sell Signal", "Symbol2"])
plt.show()

pair_correlation_trading_strategy['symbol1_buy'].head()

"""
We will now write the code that calculates the profit and loss of the pair
correlation strategy. We make a subtraction between the vectors containing the
Symbol 1 and Symbol 2 prices. We will then add these positions to create a
representation of the profit and loss:
"""
pair_correlation_trading_strategy['symbol1_position'] = \
    pair_correlation_trading_strategy['symbol1_buy'] - pair_correlation_trading_strategy['symbol1_sell']

pair_correlation_trading_strategy['symbol2_position'] = \
    pair_correlation_trading_strategy['symbol2_buy'] - pair_correlation_trading_strategy['symbol2_sell']

pair_correlation_trading_strategy['symbol1_position'].cumsum().plot()
pair_correlation_trading_strategy['symbol2_position'].cumsum().plot()

pair_correlation_trading_strategy['total_position'] = \
    pair_correlation_trading_strategy['symbol1_position'] + pair_correlation_trading_strategy['symbol2_position']
pair_correlation_trading_strategy['total_position'].cumsum().plot()
plt.title("Symbol 1 and Symbol 2 positions")
plt.legend()
plt.show()

"""
^ This code will return the following output. In the plot, the blue line represents the
profit and loss for Symbol 1, and the orange line represents the profit and loss for
Symbol 2. The green line represents the total profit and loss:
"""

"""
Until this part, we traded only one share. In regular trading, we will trade
hundreds/thousands of shares. Let's analyze what can happen when we use a
pair-correlation trading strategy.

we usually hedge our positions by
investing in something that will move on the opposite side of our positions. In the
example of a pair trading correlation, we should aim to have a neutral position by
investing the same notional in Symbol 1 and in Symbol 2. By taking the example
of having a Symbol 1 price that is markedly different to the Symbol 2 price, we
cannot use the hedge of Symbol 2 if we invest the same number of shares as we
invest in Symbol 1.
"""

pair_correlation_trading_strategy['symbol1_price'] = Symbol1_prices
pair_correlation_trading_strategy['symbol1_buy'] = np.zeros(len(Symbol1_prices))
pair_correlation_trading_strategy['symbol1_sell'] = np.zeros(len(Symbol1_prices))
pair_correlation_trading_strategy['symbol2_buy'] = np.zeros(len(Symbol1_prices))
pair_correlation_trading_strategy['symbol2_sell'] = np.zeros(len(Symbol1_prices))
pair_correlation_trading_strategy['delta'] = np.zeros(len(Symbol1_prices))

position = 0
s1_shares = 1000000
for i in range(len(Symbol1_prices)):
    s1positions = Symbol1_prices[i] * s1_shares
    s2positions = Symbol2_prices[i] * int(s1positions / Symbol2_prices[i])
    print(Symbol1_prices[i], Symbol2_prices[i])
    delta_position = s1positions - s2positions
    if not position and symbol1_buy[i] != 0:
        pair_correlation_trading_strategy['symbol1_buy'][i] = s1positions
        pair_correlation_trading_strategy['symbol2_sell'][i] = s2positions
        pair_correlation_trading_strategy['delta'][i] = delta_position
        position = 1
    elif not position and symbol1_sell[i] != 0:
        pair_correlation_trading_strategy['symbol1_sell'][i] = s1positions
        pair_correlation_trading_strategy['symbol2_buy'][i] = s2positions
        pair_correlation_trading_strategy['delta'][i] = delta_position
        position = -1
    elif position == -1 and (symbol1_sell[i] == 0 or i == len(Symbol1_prices) - 1):
        pair_correlation_trading_strategy['symbol1_buy'][i] = s1positions
        pair_correlation_trading_strategy['symbol2_sell'][i] = s2positions
        position = 0
    elif position == 1 and (symbol1_buy[i] == 0 or i == len(Symbol1_prices) - 1):
        pair_correlation_trading_strategy['symbol1_sell'][i] = s1positions
        pair_correlation_trading_strategy['symbol2_buy'][i] = s2positions
        position = 0

pair_correlation_trading_strategy['symbol1_position'] = \
    pair_correlation_trading_strategy['symbol1_buy'] - pair_correlation_trading_strategy['symbol1_sell']

pair_correlation_trading_strategy['symbol2_position'] = \
    pair_correlation_trading_strategy['symbol2_buy'] - pair_correlation_trading_strategy['symbol2_sell']

pair_correlation_trading_strategy['symbol1_position'].cumsum().plot()
pair_correlation_trading_strategy['symbol2_position'].cumsum().plot()

pair_correlation_trading_strategy['total_position'] = \
    pair_correlation_trading_strategy['symbol1_position'] + pair_correlation_trading_strategy['symbol2_position']
pair_correlation_trading_strategy['total_position'].cumsum().plot()
plt.title("Symbol 1 and Symbol 2 positions")
plt.legend()
plt.show()

"""
The code displays the delta position. The maximum amount is $25. Because this
amount is too low, we don't need to hedge this delta position:
"""
pair_correlation_trading_strategy['delta'].plot()
plt.title("Delta Position")
plt.show()

"""
This chart reveals a large quantity of orders. The pair correlation strategy without
limitation sends too many orders. We can limit the number of orders in the same
way we did previously:
-Limiting positions
-Limiting the number of orders
-Setting a higher Z-score threshold

In this section, we focused on when to enter a position, but we have not
addressed when to exit a position. While the Z-score value is above or below the
threshold limits (in this example, -1 or +1), a Z-score value within the range
between the threshold limits denotes an improbable change of spread between
the two symbol prices. Therefore, when this value is within this limit, this can be
regarded as an exit signal.

"""
