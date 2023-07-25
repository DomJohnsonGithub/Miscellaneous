import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor
from joblib import Parallel, delayed
import warnings


class MulticollinearityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vif_threshold=5):
        self.vif_threshold = vif_threshold
        self.remaining_features = []

    def fit(self, X, y=None):
        # Add Intercept
        X = X.assign(const=1)

        # Apply VIF until threshold is reached
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            vif = pd.DataFrame({"Features": X.columns,
                                "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})

            while vif['VIF'].max() > self.vif_threshold:
                feature_to_drop = vif.loc[vif['VIF'].idxmax(), 'Features']
                X = X.drop(feature_to_drop, axis=1)

                # Parallelize VIF computation
                vif = pd.DataFrame(
                    Parallel(n_jobs=-1)(
                        delayed(variance_inflation_factor)(X.values, i)
                        for i in range(X.shape[1])
                    ),
                    columns=["VIF"]
                )
                vif["Features"] = X.columns

            self.remaining_features = list(vif.Features[:-1])

        return self

    def transform(self, X, y=None):
        # Select the same features as in fit method
        X = X[self.remaining_features]
        return X

    def get_feature_names(self):
        return self.remaining_features