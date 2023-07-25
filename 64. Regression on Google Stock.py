import pandas as pd
import pandas_datareader.data as pdr
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf


def load_financial_data(start_date, end_date, output_file):
    try:
        df = pd.read_pickle(output_file)
        print('File data found...reading GOOG data')
    except FileNotFoundError:
        print('File not found...downloading the GOOG data')
        df = yf.download('GOOG', start_date, end_date)
        df.to_pickle(output_file)
    return df


def create_classification_trading_condition(df):
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low
    df = df.dropna()
    X = df[['Open-Close', 'High-Low']]
    Y = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)
    return (X, Y)


def create_regression_trading_condition(df):
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low
    df = df.dropna()
    X = df[['Open-Close', 'High-Low']]
    Y = df['Close'].shift(-1) - df['Close']

    return (X, Y)


def create_train_split_group(X, Y, split_ratio=0.8):
    return train_test_split(X, Y, shuffle=False, train_size=split_ratio)


if __name__ == "__main__":

    start = start_date = datetime(2001, 1, 1)
    end = end_date = datetime(2018, 1, 1)

    goog_data = load_financial_data(start, end, output_file="goog_data_large.pkl")
    print(goog_data)
    goog_data1 = goog_data["Close"]

    X, Y = create_regression_trading_condition(goog_data)

    X = X[0:-1]
    Y.dropna(inplace=True)

    print("X Values:\n", X)
    print("Y Values: \n", Y)
    print(len(X))
    print(len(Y))

    goog_data = pd.concat([X, Y], axis=1)
    goog_data.set_axis(["Open-Close", "High-Low", "Target"], axis=1, inplace=True)
    goog_data.dropna(inplace=True)
    print(goog_data)

    pd.plotting.scatter_matrix((goog_data[["Open-Close", "High-Low", "Target"]]), grid=True, diagonal="kde")
    plt.show()

    X_train, X_test, y_train, y_test = create_train_split_group(X, Y, split_ratio=0.8)

    #    ols = linear_model.LinearRegression(fit_intercept=True, normalize=True)
    #    ols.fit(X_train, y_train)

    # Fit the Model: OLS Regression
    try:
        ols = linear_model.LinearRegression(fit_intercept=True, copy_X=True).fit(X_train, y_train)
    except:
        ols = linear_model.LinearRegression(fit_intercept=True, copy_X=True).fit(X_train, y_train)

    # The Coefficients
    print('Coefficients: \n', ols.coef_)

    # -- Metrics -- #

    # The mean squared error
    print("Train Mean squared error: %.2f" % mean_squared_error(y_train, ols.predict(X_train)))

    # Explained variance score: 1 is perfect prediction
    print('Train Variance score: %.2f' % r2_score(y_train,
                                                  ols.predict(X_train)))

    # The mean squared error
    print("Test Mean squared error: %.2f" % mean_squared_error(y_test, ols.predict(X_test)))

    # Explained variance score: 1 is perfect prediction
    print('Test Variance score: %.2f' % r2_score(y_test, ols.predict(X_test)))

    # -- Predict and Calculate Strategy Returns -- #
    goog_data['Predicted_Signal'] = ols.predict(X)
    goog_data['GOOG_Returns'] = np.log(goog_data1 /
                                       goog_data1.shift(1))


    def calculate_return(df, split_value, symbol):
        cum_goog_return = df[split_value:]['%s_Returns' %
                                           symbol].cumsum() * 100
        df['Strategy_Returns'] = df['%s_Returns' % symbol] * df['Predicted_Signal'].shift(1)
        return cum_goog_return


    def calculate_strategy_return(df, split_value, symbol):
        cum_strategy_return = df[split_value:]['Strategy_Returns'].cumsum() * 100
        return cum_strategy_return


    cum_goog_return = calculate_return(goog_data,
                                       split_value=len(X_train), symbol='GOOG')
    cum_strategy_return = calculate_strategy_return(goog_data,
                                                    split_value=len(X_train), symbol='GOOG')


    def plot_chart(cum_symbol_return, cum_strategy_return, symbol):
        plt.figure(figsize=(10, 5))
        plt.plot(cum_symbol_return, label='%s Returns' % symbol)
        plt.plot(cum_strategy_return, label='Strategy Returns')
        plt.legend()
        plt.show()


    plot_chart(cum_goog_return, cum_strategy_return, symbol='GOOG')


    def sharpe_ratio(symbol_returns, strategy_returns):
        strategy_std = strategy_returns.std()
        sharpe = (strategy_returns - symbol_returns) / strategy_std
        return sharpe.mean()


    print("Sharpe Ratio: ", sharpe_ratio(cum_strategy_return, cum_goog_return))

    """
    Here, we can observe that the simple linear regression model using only the two features,
    Open-Close and High-Low, returns positive returns. However, it does not outperform the
    Google stock's return because it has been increasing in value since inception. But since that
    cannot be known ahead of time, the linear regression model, which does not assume/expect
    increasing stock prices, is a good investment strategy.
    """

    # -- LASSO Regression -- #
    # Fit the model
    lasso = linear_model.Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)

    # The coefficients
    print('Coefficients: \n', lasso.coef_)
    """
    Increasing the regularization parameter will shrink the coefficients further.

    If the regularization parameter is increased to 0.6, the coefficients shrink much further to [
    0. -0.00540562], and the first predictor gets assigned a weight of 0, meaning that predictor
    can be removed from the model. L1 regularization has this additional property of being
    able to shrink coefficients to 0, thus having the extra advantage of being useful for feature
    selection, in other words, it can shrink the model size by removing some predictors.
    """

    # -- Ridge Regression -- #
    ridge = linear_model.Ridge(alpha=10000)
    ridge.fit(X_train, y_train)

    # The coefficients
    print('Coefficients: \n', ridge.coef_)

    # -- Elastic Net -- #
    elastic_nt = linear_model.ElasticNet(alpha=2)
    elastic_nt.fit(X_train, y_train)

    print('Coefficients: \n', ridge.coef_)
