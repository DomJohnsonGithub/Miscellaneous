import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin


class Winsorizer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(self, lower_percentile=1, upper_percentile=99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bound = None
        self.upper_bound = None

    def fit(self, X, y=None):
        self.lower_bound = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_bound = np.percentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X):
        X_transformed = np.copy(X)
        for i in range(X.shape[1]):
            X_transformed[:, i] = np.clip(X_transformed[:, i], self.lower_bound[i], self.upper_bound[i])
        return X_transformed

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X)
        return self.transform(X)