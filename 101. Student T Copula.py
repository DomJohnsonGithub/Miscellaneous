from scipy.optimize import brentq
from scipy.optimize import minimize_scalar
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


class Copula:
    '''
    Parent class for copulas
    '''

    # parameters for calculating probabilities numerically
    prob_sample_size = 10000000  # sample size
    prob_band = 0.005  # band size
    prob_sample = np.array([])  # sample

    def log_likelihood(self, u, v):
        '''
        Compute log likelihood for copula

        Args:
            u (numpy.ndarray): input (uniform) data with shape (n_samples,)
            v (numpy.ndarray): input (uniform) data with shape (n_samples,)

        Returns:
            numpy.ndarray: log likelihood (1,)
        '''
        return np.log(self.pdf(u, v)).sum()

    def cdf_u_given_v(self, u, v):
        '''
        Compute numerically conditional CDF P(U<=u|V=v)

        Args:
            u (numpy.ndarray): input (uniform) data with shape (1,)
            v (numpy.ndarray): input (uniform) data with shape (1,)

        Returns:
            numpy.ndarray: conditional CDF (1,)
        '''
        # generate sample if it does not exist
        if len(self.prob_sample) == 0:
            self.prob_sample = self.sample(size=self.prob_sample_size)
        # calculate conditional CDF
        s_u = self.prob_sample[:, 0]
        s_v = self.prob_sample[:, 1]
        condition = (s_v <= v + self.prob_band / 2) & (s_v >= v - self.prob_band / 2)
        sample_size = len(s_u[condition])
        prob = (s_u[condition] <= u).sum() / sample_size
        return prob

    def cdf_v_given_u(self, u, v):
        '''
        Compute numberically conditional CDF P(V<=v|U=u)

        Args:
            u (numpy.ndarray): input (uniform) data with shape (1,)
            v (numpy.ndarray): input (uniform) data with shape (1,)

        Returns:
            numpy.ndarray: conditional CDF (1,)
        '''
        # generate sample if it does not exist
        if len(self.prob_sample) == 0:
            self.prob_sample = self.sample(size=self.prob_sample_size)
        # calculate conditional CDF
        s_u = self.prob_sample[:, 0]
        s_v = self.prob_sample[:, 1]
        condition = (s_u <= u + self.prob_band / 2) & (s_u >= u - self.prob_band / 2)
        sample_size = len(s_v[condition])
        prob = (s_v[condition] <= v).sum() / sample_size
        return prob

    def plot_sample(self, size=10000):
        '''
        Generate and plot a sample from copula

        Args:
            size (int): sample size
        '''

        # generate sample
        sample = self.sample(size=size)
        u = sample[:, 0]
        v = sample[:, 1]
        # plot
        fig, ax = plt.subplots(figsize=(18, 4), nrows=1, ncols=3)
        ax[0].hist(u, density=True)
        ax[0].set(title='Historgram of U')
        ax[1].hist(v, density=True)
        ax[1].set(title='Historgram of V')
        ax[2].scatter(u, v, alpha=0.2)
        ax[2].set(title='Scatterplot of U and V', xlabel='U', ylabel='V')


class StudentTCopula(Copula):
    """
    Class for bivariate t copula
    """
    name = "StudentT"
    num_params = 2

    def __init__(self, rho=0, nu=1):
        """
        Initialize Student T copula object
        :param rho: copula parameter rho
        :param nu: degrees of freedom
        """
        self.rho = rho
        self.cov = np.array([[1, rho], [rho, 1]])
        self.nu = nu


    def cdf(self, u, v):
        """
        Compute CDF for Student T copula.
        :param u: input (uniform) data with shape (n_samples,)
        :param v: input (uniform) data with shape (n_samples,)
        :return: numpy.ndarray: cumulative probability (n_samples,)
        """
        data = np.vstack([stats.norm.ppf(u, self.nu), stats.norm.ppf(v, self.nu)]).T

        return stats.multivariate_t(shape=self.cov, df=self.nu).cdf(data)


    def pdf(self, u, v):
        """
        Compute PDF for Student-T copula.
        :param u: input (uniform) data with shape (n_samples,)
        :param v: input (uniform) data with shape (n_samples,)
        :return: numpy.ndarray: probability density (n_samples,)
        """
        from scipy.stats import t
        from scipy.special import gamma

        a = t.ppf(u, self.nu)  # degrees of freedom
        b = t.ppf(v, self.nu)

        K_2 = 0.5 * (gamma(self.nu / 2) / gamma(0.5 + self.nu / 2)) ** 2 * self.nu * (1 - self.rho) ** (-0.5)

        return K_2 * ((1 + a**2/self.nu)*(1 + b**2/self.nu))**((self.nu+1)/2) * \
               (1 + (a**2 - 2*self.rho*a*b + b**2)/((1 - self.rho**2)*self.nu))**(-1-self.nu/2)


    def fit(self, u, v):
        """
        Fit Gaussian copula to data
        :param u: input (uniform) data with shape (n_samples,)
        :param v: input (uniform) data with shape (n_samples,)
        """
        tau = stats.kendalltau(u, v)[0]
        self.rho = np.sin(tau * np.pi / 2)
        self.cov = np.array([[1, self.rho], [self.rho, 1]])


    def sample(self, size=1):
        """
        Generate sample from Student T copula.
        :param size: sample size (int)
        :return: sample (size,2) (np.ndarray)
        """
        smp = stats.multivariate_t(shape=self.cov).rvs(size=size)
        u = stats.t.cdf(smp[:, 0], self.nu)
        v = stats.t.cdf(smp[:, 1], self.nu)
        sample = np.vstack([u, v]).T

        return sample