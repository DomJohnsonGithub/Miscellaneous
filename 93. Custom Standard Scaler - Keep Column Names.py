import pandas as pd
from sklearn.preprocessing import StandardScaler


class AnotherStandardScaler(StandardScaler):
    def fit(self, X, y=None, **kwargs):
        self.feature_names_ = X.columns
        return super().fit(X, y, **kwargs)

    def transform(self, X, **kwargs):
        return pd.DataFrame(data=super().transform(X, **kwargs),
                            columns=self.feature_names_)
