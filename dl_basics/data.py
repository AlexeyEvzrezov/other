import numpy as np
import torch


def load_data(X, y, batch_size, shuffle=True):
    n_samples = len(X)
    ids = list(range(n_samples))
    if shuffle:
        np.random.shuffle(ids)
    for i in range(0, n_samples, batch_size):
        batch_ids = torch.tensor(ids[i:min(i + batch_size, n_samples)])
        yield X[batch_ids], y[batch_ids]


def synth_linear(w, b, n_samples):
    """Generate y = Xw + b + Gaussian noise."""
    X = torch.normal(0, 1, (n_samples, len(w)))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)