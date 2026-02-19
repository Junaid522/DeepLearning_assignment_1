import numpy as np
from ..utils.losses import softmax, cross_entropy_onehot
from ..utils.metrics import accuracy_from_probs


class SharedBiasMLP:
    """
    Two-layer MLP with a single shared bias vector b used in BOTH affine transforms:
        z1 = W x + b
        h1 = relu(z1)
        z2 = U h1 + b
        yhat = softmax(z2)
    Requirement: hidden_dim == output_dim so the same b can be added in both layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 0):
        if hidden_dim != output_dim:
            raise ValueError("Shared bias requires hidden_dim == output_dim (because b is reused in both layers).")

        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.1, size=(hidden_dim, input_dim))
        self.U = rng.normal(0, 0.1, size=(output_dim, hidden_dim))
        self.b = np.zeros(hidden_dim, dtype=np.float64)

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def relu_deriv(z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(np.float64)

    def forward(self, X: np.ndarray):
        """
        X: (N, d)
        Returns: probs (N, k), cache for backward
        """
        z1 = X @ self.W.T + self.b  # (N, m)
        h1 = self.relu(z1)          # (N, m)

        z2 = h1 @ self.U.T + self.b  # (N, k) and k==m
        probs = softmax(z2)          # (N, k)

        cache = {
            "X": X,
            "z1": z1,
            "h1": h1,
            "z2": z2,
            "probs": probs,
        }
        return probs, cache

    def loss(self, probs: np.ndarray, y_onehot: np.ndarray) -> float:
        return cross_entropy_onehot(probs, y_onehot)

    def backward(self, cache, y_onehot: np.ndarray):
        """
        Implements:
          delta2 = dL/dz2 = (probs - y)/N
          delta1 = (delta2 U) ⊙ relu'(z1)
          db = sum(delta1) + sum(delta2)    [shared bias]
        """
        X = cache["X"]
        z1 = cache["z1"]
        h1 = cache["h1"]
        probs = cache["probs"]

        N = X.shape[0]
        delta2 = (probs - y_onehot) / N                 # (N, k)
        dU = delta2.T @ h1                              # (k, m)

        delta1 = (delta2 @ self.U) * self.relu_deriv(z1)  # (N, m)
        dW = delta1.T @ X                                # (m, d)

        # shared bias accumulates gradients from both occurrences
        db = np.sum(delta1, axis=0) + np.sum(delta2, axis=0)  # (m,)

        grads = {"dW": dW, "dU": dU, "db": db}
        return grads

    def step(self, grads, lr: float):
        self.W -= lr * grads["dW"]
        self.U -= lr * grads["dU"]
        self.b -= lr * grads["db"]

    def evaluate(self, X: np.ndarray, y_true: np.ndarray):
        probs, _ = self.forward(X)
        acc = accuracy_from_probs(probs, y_true)
        return acc
