import numpy as np
from sklearn.decomposition import KernelPCA

class CustomKernelPCA(KernelPCA):
    """
    Select n_components as same as number of features, this then looks at the cumulative sum
    of explained variance and based on the threshold, it will keep only KPCA's that cumsum
    to that threshold.
    """
    def __init__(self, threshold=0.9, n_components=None, kernel='rbf', gamma=None,
                 degree=3, coef0=1, kernel_params=None, alpha=1.0,
                 fit_inverse_transform=False, eigen_solver='auto',
                 tol=0, max_iter=None, remove_zero_eig=False,
                 random_state=None, copy_X=True, n_jobs=-1):
        super().__init__(n_components=n_components, kernel=kernel, gamma=gamma,
                         degree=degree, coef0=coef0, kernel_params=kernel_params,
                         alpha=alpha, fit_inverse_transform=fit_inverse_transform,
                         eigen_solver=eigen_solver, tol=tol, max_iter=max_iter,
                         remove_zero_eig=remove_zero_eig, random_state=random_state,
                         copy_X=copy_X, n_jobs=n_jobs)
        self.threshold = threshold
        self.cumulative_explained_variance = None
        self.n_components_to_keep = int

    def fit(self, X, y=None):
        super().fit(X, y)
        self.cumulative_explained_variance = np.cumsum(self.eigenvalues_ / np.sum(self.eigenvalues_))

    def transform(self, X):
        principal_components = super().transform(X)
        self.n_components_to_keep = np.argmax(self.cumulative_explained_variance >= self.threshold) + 1
        return principal_components[:, :self.n_components_to_keep]

    def fit_transform(self, X, y=None, **params):
        self.fit(X, y)
        return self.transform(X)