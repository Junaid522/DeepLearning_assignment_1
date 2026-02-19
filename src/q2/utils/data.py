import numpy as np


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = y.astype(int)
    out = np.zeros((len(y), num_classes), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


def make_synthetic(N=600, d=5, k=4, seed=0):
    """
    Simple synthetic classification dataset.
    This is just to reproduce convergence behaviour and compare models.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(N, d))

    # create labels with a mild nonlinear rule
    Wtrue = rng.normal(0, 1, size=(k, d))
    logits = X @ Wtrue.T
    y = np.argmax(logits + 0.3 * rng.normal(0, 1, size=logits.shape), axis=1)

    return X.astype(np.float64), y.astype(int)


def train_val_split(X, y, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)

    n_val = int(len(X) * val_ratio)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]
