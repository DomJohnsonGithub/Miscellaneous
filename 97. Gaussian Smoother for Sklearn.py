from tsmoothie.smoother import GaussianSmoother
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class GaussianSmootherWrap(TransformerMixin, BaseEstimator):

    def __init__(self, sigma, n_knots, df=True):
        self.gs = None
        self.index_ = None
        self.feature_names_ = None
        self._is_fitted = None
        self.smoother = None
        self.sigma = sigma
        self.n_knots = n_knots
        self.df = df

    def fit(self, X, y=None):
        self._is_fitted = True
        if self.df:
            self.feature_names_ = X.columns
            self.index_ = X.index
        return self

    def transform(self, X, y=None):
        self.gs = GaussianSmoother(sigma=self.sigma, n_knots=self.n_knots)
        self.gs.smooth(X.copy().T)
        return pd.DataFrame(self.gs.smooth_data.T,
                            index=self.index_,
                            columns=self.feature_names_) \
            if self.df == True else self.smoother.smooth_data.T

    def fit_transform(self, X, y=None, **kwargs):
        return super().fit_transform(X)
