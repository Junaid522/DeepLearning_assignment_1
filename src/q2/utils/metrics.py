import numpy as np


def accuracy_from_probs(probs: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.argmax(probs, axis=1)
    y_true = y_true.astype(int)
    return float(np.mean(y_pred == y_true))
