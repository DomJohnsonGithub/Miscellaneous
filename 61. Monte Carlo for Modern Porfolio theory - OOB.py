import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
from pandas_datareader._utils import RemoteDataError
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf


# import seaborn as sns
# from scipy.optimize import minimize


# sns.set_style("darkgrid")


class IMPORT_DATA:
    def __init__(self, tickers, start_date, end_date):
        """
        Initialisation of properties to import data.
        :param tickers: symbols of the assets you wish to analyse
        :param source: where to get the data from
        :param start_date: beginning date of the data
        :param end_date: last date of the data
        """
        self.tickers = tickers
        self.start = start_date
        self.end = end_date
        self.df = pd.DataFrame()

    def load_data(self):
        """
        This functions allows the user to load in data and
        will attempt it max 5 times if a problem does occur.
        """
        if len(self.tickers) <= 1:
            raise ValueError(
                "This Monte Carlo Simulation requires more than one ticker. Please include at least one more symbol.")
        for i in range(5):
            try:
                self.df = yf.download(self.tickers, start=self.start, end=self.end)
                break
            except:
                print("Trying to establish a connection...try: ", i)
                print("Are you connected to the internet.")
                if i == 4:
                    raise RemoteDataError("Cannot connect to Yahoo Finance.")

    def create_close_df(self, col):
        """
        This function creates a dataframe of the prices of the assets.
        :param col: column which to use from the multi-index dataframe.
        """
        idx = pd.IndexSlice
        try:
            print(self.df)
            self.df = self.df.loc[idx[:], [col]]
            print(self.df)
        except NameError:
            print(f"'{col}' column does not exist in the dataframe.")
            self.df = self.df.loc[idx[:], ["Close"]]

    def create_log_returns(self):
        """
        This functions creates the log returns of the data.
        """
        if np.sum(self.df.isna().sum().values) >= 1:
            raise ValueError("NaN values exist in the dataframe. Consequently, erroneous results will occur.")
        else:
            self.df = np.log(1 + self.df.pct_change())
            self.df.dropna(inplace=True)

        if self.df.empty == True:
            raise Exception(
                "Sorry, you seem to have an empty dataframe. Go back a step to check for any NaN values that are causing this.")

    def get_df(self):
        """
        A getter function to obtain the final dataframe.
        :return: the final dataframe.
        """
        return self.df


class MONTE_CARLO_OPTIMAL_PORTFOLIO:
    def __init__(self, rets_df, simulations, seed=False):
        """
        This function performs the monte carlo simulation on various weights of the portfolios.
        :param rets_df: this will be the log returns.
        :param simulations: how many simulations to run
        :param seed: if we want to keep the simulations the same or random.
        """
        self.df = rets_df
        self.monte_carlo_sims = simulations
        self.seed = seed

        self.port_exp_returns = []
        self.port_risk = []
        self.port_weights = []

    @classmethod
    def number_simulations(cls, simulations):
        print("Simulations:", simulations)

    def monte_carlo_optimal_portfolio(self):
        """
        This is the Monte Carlo simulation.
        """
        if self.seed == False:
            asset_exp_return = self.df.mean().values

            port_exp_returns = []
            port_risk = []
            port_weights = []
            if self.monte_carlo_sims > 100000:
                raise ValueError("Too many simulations for monte carlo, may cause instability issues.")
            try:
                for i in range(self.monte_carlo_sims):
                    weights = np.random.random(len(self.df.columns))
                    weights /= np.sum(weights)

                    port_weights.append(weights)
                    port_exp_returns.append(np.dot(asset_exp_return, weights) * 252)
                    port_risk.append(np.sqrt(np.dot(weights, np.dot(self.df.cov().values * 252, weights.T))))
            except MemoryError:
                print("Not enough RAM available to complete the Monte Carlo Simulation")
            except KeyboardInterrupt:
                print("Accidentally pressed something to interrupt the program you are executing.")
                print("Re-run Simulation!")
            except ValueError:
                print("Shape of one or both matrices do not allow for matrix multiplication.")
            except RuntimeError:
                print("No. of simulations should not be a string, must be an integer.")

            self.port_exp_returns = np.array(port_exp_returns)
            self.port_risk = np.array(port_risk)
            self.port_weights = np.vstack(port_weights)

        else:
            np.random.seed(42)

            asset_exp_return = self.df.mean().values

            port_exp_returns = []
            port_risk = []
            port_weights = []
            if self.monte_carlo_sims > 100000:
                raise ValueError("Too many simulations for monte carlo, may cause instability issues.")
            try:
                for i in range(self.monte_carlo_sims):
                    weights = np.random.random(len(self.df.columns))
                    weights /= np.sum(weights)

                    port_weights.append(weights)
                    port_exp_returns.append(np.dot(asset_exp_return, weights) * 252)
                    port_risk.append(np.sqrt(np.dot(weights, np.dot(self.df.cov().values * 252, weights.T))))
            except MemoryError:
                print("Not enough RAM to complete the Monte Carlo Simulation")
            except KeyboardInterrupt:
                print("Accidentally pressed something to interupt the program you are executing.")
                print("Re-run simulation")
            except ValueError:
                print("Shape of one or both matrices do not allow for matrix multiplication.")
            except RuntimeError:
                print("No. of simulations should not be a string, must be an integer.")

            self.port_exp_returns = np.array(port_exp_returns)
            self.port_risk = np.array(port_risk)
            self.port_weights = np.vstack(port_weights)

    def get_portfolio_expected_returns(self):
        """
        A getter function to obtain the different portfolio expected returns.
        :return: portfolio expected returns.
        """
        return self.port_exp_returns

    def get_portfolio_risks(self):
        """
        A getter function to obtain the different portfolio volatilities.
        :return: portfolio risk.
        """
        return self.port_risk

    def get_portfolio_weights(self):
        """
        A getter function to obtain the different portfolio weights.
        :return: various portfolio weights which do sum to 1.
        """
        return self.port_weights


class SHARPE_RATIO:
    def __init__(self, risk_free_rate, port_exp_rets_array, port_risk_array, port_weights_array):
        """
        These our the properties required for calculating the sharpe ratio.
        :param risk_free_rate: use a proxy for the risk-free rate. Usually government debt.
        :param port_exp_rets_array: an array of portfolio returns
        :param port_risk_array: an array of portfolio volatility
        :param port_weights_array: an array of various weights that can be invested in.
        """
        self.rf = risk_free_rate
        self.port_exp_returns = port_exp_rets_array
        self.port_risk = port_risk_array
        self.port_weights = port_weights_array

    def calc_sharpe_ratios(self):
        """
        Function to calculate all sharpe ratios.
        :return: different sharpe ratios.
        """
        try:
            self.sharpe_ratio = (self.port_exp_returns - self.rf) / self.port_risk
        except ZeroDivisionError:
            print("Denominator in the formula for the Sharpe Ratio is zero. Cannont divide by zero.")
        except RuntimeError:
            print("Risk free rate should not be a string, must be a float.")

        return self.sharpe_ratio

    def calc_annualised_sharpe_ratios(self):
        """
        Calculates the annualised Sharpe ratios.
        """
        try:
            self.annualised_sharpe_ratio = ((self.port_exp_returns - self.rf) / self.port_risk) * (252 ** 0.5)
        except ZeroDivisionError:
            print("Denominator in the formula for the Sharpe Ratio is zero. Cannont divide by zero.")
        except RuntimeError:
            print("Risk free rate should not be a string, must be a float.")

    def find_max_sharpe_ratio(self):
        """
        Finds the location where we have the max sharpe ratio.
        """
        try:
            self.max_sharpe = self.sharpe_ratio[self.sharpe_ratio.argmax()]
        except IndexError:
            print("Cannot find the position where the max sharpe ratio exists.")

    def find_max_annualised_sharpe_ratio(self):
        """
        Finds the location where we have the max annualised sharpe ratio.
        """
        try:
            self.max_annualised_sharpe = self.annualised_sharpe_ratio[self.annualised_sharpe_ratio.argmax()]
        except IndexError:
            print("Cannot find the position where the maximised annualised sharpe ratio exists")

    def find_portfolio_maximising_sharpe(self):
        """
        Find the return and risk where sharpe ratio is maximised.
        """
        try:
            self.max_sr_vol = self.port_risk[self.sharpe_ratio.argmax()]
            self.max_sr_ret = self.port_exp_returns[self.sharpe_ratio.argmax()]
        except IndexError:
            print(
                "Cannot find the position of where the portfolio expected returns and volatility, maximising the sharpe, exist.")

    def print_summary(self):
        """
        Prints a summary of the measurements of the best portfolio.
        """
        print("\n-----------------------------------------------")
        print("Max Sharpe: ", self.max_sharpe)
        print("Max Annualised Sharpe: ", self.max_annualised_sharpe)
        print("Max Sharpe Return: ", self.max_sr_ret)
        print("Max Sharpe Volatility: ", self.max_sr_vol)
        print("-----------------------------------------------")

    def get_max_sharpe_ratio(self):
        """
        Function to get the largest sharpe ratio.
        :return: maximum sharpe ratio
        """
        return self.max_sharpe

    def get_max_annualised_sharpe(self):
        """
        Function to get the max annualised sharpe ratio.
        :return: maximum annualised sharpe ratio
        """
        return self.max_annualised_sharpe

    def get_max_sr_vol_ret(self):
        """
        Function enusures that portfolio returns and risk are not below zero.
        It also allows user to get these items.
        :return: returns for max sharpe, and risk for max sharpe
        """
        if self.max_sr_ret <= 0:
            raise ValueError("Cannot have negative portfolio expected returns")
        if self.max_sr_vol <= 0:
            raise ValueError("Cannot have negative portfolio volatility.")
        return self.max_sr_vol, self.max_sr_ret

    def get_best_weights_for_max_sharpe(self):
        """
        Ensures that portfolio weights sum to 1.0
        and returns the best portfolio weights.
        :return: best portfolio weights
        """
        if np.round(np.sum(self.port_weights[self.sharpe_ratio.argmax()]), 5) != 1.0:
            raise ValueError("Portfolio weighting should equal 1.")
        return self.port_weights[self.sharpe_ratio.argmax()]


class EQUAL_WEIGTHED_PORTFOLIO:
    def __init__(self, rf):
        """
        This class helps to build the metrics for an equal
        weighted portfolio to compare against the max sharpe ratio.
        :param rf: risk free rate (use a proxy such as government debt)
        """
        self.rf = rf

        self.equal_weights = []
        self.equal_port_exp_return = []
        self.equal_port_risk = []
        self.sharpe_ratio = []

    @staticmethod
    def _expected_returns(df):
        """
        Computes the portfolio expected returns. Checks to see if
        dataframe is empty.
        :param df: dataframe which will be composed of log returns
        :return: portfolio asset returns.
        """
        if df.empty == True:
            raise Exception("Empty dataframe. Cannot compute expected returns.")
        else:
            return df.mean().values

    @staticmethod
    def _returns_cov(df):
        """
        Function to return portfolio covariance.
        :param df: dataframe which will be composed of log returns
        :return: portfolio covariance.
        """
        if df.empty == True:
            raise Exception("Empty dataframe. Cannot compute expected returns.")
        else:
            return df.cov().values

    def generate_equal_weights(self, df):
        """
        Function generates equal weights. Ensures it sums to 1.
        :param df: takes in dataframe of log returns
        """
        self.equal_weights = np.array([1 / len(df.columns) \
                                       for i in range(len(df.columns))])
        if np.sum(self.equal_weights) != 1:
            raise ValueError("Sum of portfolio weights does not equal 1. They are not equal weighted.")

    def calc_equal_port_return_risk(self, df):
        """
        This functions calculates the equal weighted portfolio risk and returns.
        :param df: dataframe of log returns.
        """
        asset_exp_returns = EQUAL_WEIGTHED_PORTFOLIO._expected_returns(df)
        cov = EQUAL_WEIGTHED_PORTFOLIO._returns_cov(df)

        try:
            self.equal_port_exp_return = np.dot(asset_exp_returns, self.equal_weights) * 252
        except ValueError:
            print("Shape of one or both of the matrices do not allow for matrix multiplication.")
        try:
            self.equal_port_risk = np.sqrt(np.dot(self.equal_weights, np.dot(cov * 252, self.equal_weights.T)))
        except ValueError:
            print("Shape of one of matrices or all do not allow for matrix multiplication.")

    def calc_equal_port_sharpe_ratios(self):
        """
        Calculates the sharpe ratio of an equal weighted portfolio.
        """
        try:
            self.sharpe_ratio = (self.equal_port_exp_return - self.rf) / self.equal_port_risk
        except ZeroDivisionError:
            print("Cannont divide by zero. Equal Portfolio Risk is zero in Sharpe calculation.")
        except RuntimeError:
            print("Risk free rate should not be a string, must be a float.")

    def get_equal_port_risk_return(self):
        """
        Gets the equal weighted portfolio risk and return
        :return: risk and returns of equal weighted portfolio
        """
        return self.equal_port_risk, self.equal_port_exp_return

    def get_equal_port_sharpe(self):
        """
        Obtains the sharpe ratio for equal weigthed portfolio.
        :return: sharpe ratio of equal weighted portfolio
        """
        return self.sharpe_ratio


class VISUALISE_EFFICIENT_FRONTIER:
    def __init__(self, port_risk, rf, max_sharpe,
                 max_sr_vol, max_sr_ret, equal_port_risk,
                 equal_port_ret, sharpe_ratios, port_returns):
        """
        This class allows us to calculate the capital allocation line, min variance portfolio
        and visualize the efficient frontier from the simulated portfolios from monte carlo.
        :param port_risk: volatilies of the portfolios
        :param rf: risk free rate
        :param max_sharpe: maximum sharpe ratio
        :param max_sr_vol: risk for max sharpe ratio
        :param max_sr_ret: return for max sharpe ratio
        :param equal_port_risk: risk for equal weighted portfolio
        :param equal_port_ret: return for 1/N portfolio
        :param sharpe_ratio: sharpe ratios generated from monte carlo
        :param port_returns: portfolio returns from monte carlo
        """
        self.port_risk = port_risk
        self.rf = rf
        self.max_sharpe = max_sharpe
        self.max_sr_vol = max_sr_vol
        self.max_sr_ret = max_sr_ret
        self.equal_port_risk = equal_port_risk
        self.equal_port_ret = equal_port_ret
        self.sharpe_ratios = sharpe_ratios
        self.port_returns = port_returns

        self._cal_x = []
        self._cal_y = []
        self.min_var = []
        self.return_for_min_var = []

    def calc_capital_allocation_line(self):
        """
        Function to calculate the capital allocation line.
        """
        self.cal_x = np.linspace(0, np.max(self.port_risk), 200)
        self.cal_y = self.rf + self.cal_x * self.max_sharpe

    def calc_min_variance_portfolio(self):
        """
        Function to calculate the minimum variance portfolio.
        """
        try:
            where = self.port_risk.argmin()
            self.min_var = self.port_risk[where]
            self.return_for_min_var = self.port_returns[where]
        except IndexError:
            print("Position in array does not exist!")

    def visualize_efficient_frontier(self):
        """
        This function allows us to take all information
        previously computed to visualize the efficient frontier.
        """
        plt.figure(dpi=80)
        plt.scatter(self.port_risk, self.port_returns, c=self.sharpe_ratios, s=1, cmap="viridis")
        plt.colorbar(label="Sharpe Ratio")

        plt.scatter(self.max_sr_vol, self.max_sr_ret, c="black", s=20, label="Max Sharpe Ratio Portfolio")
        plt.scatter(self.equal_port_risk, self.equal_port_ret, c="pink", s=20, label="Equal Weighted Portfolio")
        plt.scatter(self.min_var, self.return_for_min_var, c="orange", s=20, label="Min Risk Portfolio")

        plt.plot(self.cal_x, self.cal_y, c="red", lw=1.0, label="Capital Allocation Line")

        plt.xlabel("Portfolio Risk (Volatility)")
        plt.ylabel("Portfolio Return")
        plt.suptitle("Modern Portfolio Theory")
        plt.title("Omni Efficient Frontier")
        plt.legend(loc="best")
        plt.show()


def main():
    # Create import data object
    tickers = ["AAPL", "GS", "SEB", "KO"]
    df = IMPORT_DATA(tickers=tickers,
                     start_date=datetime(2000, 1, 1),
                     end_date=datetime(2022, 3, 29))

    df.load_data()  # load in the data
    df.create_close_df(col="Close")  # reassemble closing prices into one df
    df.create_log_returns()  # convert price into log prices
    df = df.get_df()  # get the dataframe

    # Perform Monte Carlo Simulation to generate Portfolio weights, risk and returns
    monte_carlo_sims = 10000
    monte_carlo_portfolio = MONTE_CARLO_OPTIMAL_PORTFOLIO(rets_df=df, simulations=monte_carlo_sims, seed=False)
    monte_carlo_portfolio.monte_carlo_optimal_portfolio()
    monte_carlo_portfolio.number_simulations(simulations=monte_carlo_sims)

    port_exp_returns = monte_carlo_portfolio.get_portfolio_expected_returns()  # portfolio expected returns
    port_risk = monte_carlo_portfolio.get_portfolio_risks()  # portfolio risks
    port_weights = monte_carlo_portfolio.get_portfolio_weights()  # portfolio weights

    # Create sharpe ratio object
    rf = 0.02
    sharpe = SHARPE_RATIO(risk_free_rate=rf,
                          port_exp_rets_array=port_exp_returns,
                          port_risk_array=port_risk,
                          port_weights_array=port_weights)

    sharpe_ratio = sharpe.calc_sharpe_ratios()  # calc all sharpe ratios
    sharpe.calc_annualised_sharpe_ratios()  # calc annualised sharpe ratios

    sharpe.find_max_sharpe_ratio()  # find the max sharpe ratio
    sharpe.find_max_annualised_sharpe_ratio()  # find the max annualised sharpe ratio
    sharpe.find_portfolio_maximising_sharpe()  # find the risk and return that maximise sharpe
    sharpe.print_summary()  # show summary of stats
    best_weights = sharpe.get_best_weights_for_max_sharpe()  # find the optimal weights
    print(f"Best Weights for Max Sharpe Portfolio {tickers}: ", best_weights)  # show best weights

    max_sharpe_ratio = sharpe.get_max_sharpe_ratio()  # get the max sharpe ratio
    max_sr_vol, max_sr_ret = sharpe.get_max_sr_vol_ret()  # risk and return that maximise sr

    # Create an equal weighted portfolio object
    equal_weighted_port = EQUAL_WEIGTHED_PORTFOLIO(rf=rf)  # rf = risk free rate
    equal_weighted_port.generate_equal_weights(df=df)  # generate equal weights
    equal_weighted_port.calc_equal_port_return_risk(df=df)  # calc portfolio risk and return
    equal_weighted_port.calc_equal_port_sharpe_ratios()  # calc the equal portfolio sharpe

    equal_port_risk, equal_port_exp_returns = equal_weighted_port.get_equal_port_risk_return()  # risk and return

    # Visualise the Efficient Frontier
    viz = VISUALISE_EFFICIENT_FRONTIER(port_risk=port_risk, rf=rf,
                                       sharpe_ratios=sharpe_ratio,
                                       max_sharpe=max_sharpe_ratio,
                                       max_sr_vol=max_sr_vol,
                                       max_sr_ret=max_sr_ret,
                                       equal_port_risk=equal_port_risk,
                                       equal_port_ret=equal_port_exp_returns,
                                       port_returns=port_exp_returns)

    viz.calc_capital_allocation_line()  # calc the capital allocation line
    viz.calc_min_variance_portfolio()  # calc the min variance portfolio
    viz.visualize_efficient_frontier()  # show the portfolios and efficient frontier


if __name__ == "__main__":
    for i in range(3):
        main()
