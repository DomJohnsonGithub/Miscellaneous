import numpy as np


class BlockedTimeSeriesPurgedSplit:
    def __init__(self, n_splits, purge_gap, train_percentage):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.train_pct = train_percentage

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def get_purge_gap(self):
        return self.purge_gap

    def get_train_percentage(self):
        return self.train_pct

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            begin = i * k_fold_size
            stop = begin + k_fold_size
            mid = int(self.train_pct * (stop - begin)) + begin

            yield indices[begin: mid], indices[mid + self.purge_gap: stop]