import numpy as np


class StandardScaler:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, x):
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        return self

    def transform(self, x):
        return (x - self.mean_) / self.std_

    def fit_transform(self, x):
        return self.fit(x).transform(x)


def chronological_split(n, train_fraction=0.7, val_fraction=0.15):
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


def remap_event_ids(event_id):
    out = np.full_like(event_id, -1)
    ids = [i for i in np.unique(event_id) if i >= 0]
    for new, old in enumerate(ids):
        out[event_id == old] = new
    return out
