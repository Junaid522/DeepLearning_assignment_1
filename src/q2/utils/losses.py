import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / (np.sum(ez, axis=1, keepdims=True) + 1e-12)


def cross_entropy_onehot(probs: np.ndarray, y_onehot: np.ndarray) -> float:
    # y_onehot: (N, k)
    return float(-np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1)))
