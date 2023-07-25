import numpy as np
from sklearn.calibration import CalibratedClassifierCV


class IsotonicCalibrationClassifierCV(CalibratedClassifierCV):
    def __init__(self, estimator=None, method="isotonic", cv="prefit", n_jobs=11):
        super().__init__(estimator=estimator, method=method, cv=cv, n_jobs=n_jobs)


class SigmoidCalibrationClassifierCV(CalibratedClassifierCV):
    def __init__(self, estimator=None, method="sigmoid", cv="prefit", n_jobs=11):
        super().__init__(estimator=estimator, method=method, cv=cv, n_jobs=n_jobs)


class CombinedCalibrationClassifierCV(IsotonicCalibrationClassifierCV, SigmoidCalibrationClassifierCV):
    def __init__(self, estimator):
        self.estimator = estimator
        super().__init__(estimator=self.estimator)
        self.iso = IsotonicCalibrationClassifierCV(estimator=self.estimator)
        self.sig = SigmoidCalibrationClassifierCV(estimator=self.estimator)
        self.y = np.array

    def fit(self, X, y, sample_weight=None, **fit_params):
        self.iso.fit(X, y, sample_weight, **fit_params)
        self.sig.fit(X, y, sample_weight, **fit_params)
        self.y = y
        return self

    def predict_proba(self, X):
        iso_probs = self.iso.predict_proba(X)
        sig_probs = self.sig.predict_proba(X)
        return np.column_stack((np.mean([iso_probs[:, 0], sig_probs[:, 0]], axis=0),
                                np.mean([iso_probs[:, 1], sig_probs[:, 1]], axis=0)))

    def predict(self, X):
        vals = np.unique(self.y, return_counts=True)[0]
        return np.where(self.predict_proba(X)[:, 1] >= 0.5, max(vals), min(vals))