import numpy as np
from ..utils.losses import softmax, cross_entropy_onehot
from ..utils.metrics import accuracy_from_probs


class IndependentBiasMLP:
    """
    Two-layer MLP with independent biases:
        z1 = W x + b1
        h1 = relu(z1)
        z2 = U h1 + b2
        yhat = softmax(z2)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.1, size=(hidden_dim, input_dim))
        self.U = rng.normal(0, 0.1, size=(output_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.b2 = np.zeros(output_dim, dtype=np.float64)

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def relu_deriv(z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(np.float64)

    def forward(self, X: np.ndarray):
        z1 = X @ self.W.T + self.b1
        h1 = self.relu(z1)

        z2 = h1 @ self.U.T + self.b2
        probs = softmax(z2)

        cache = {"X": X, "z1": z1, "h1": h1, "z2": z2, "probs": probs}
        return probs, cache

    def loss(self, probs: np.ndarray, y_onehot: np.ndarray) -> float:
        return cross_entropy_onehot(probs, y_onehot)

    def backward(self, cache, y_onehot: np.ndarray):
        X = cache["X"]
        z1 = cache["z1"]
        h1 = cache["h1"]
        probs = cache["probs"]

        N = X.shape[0]
        delta2 = (probs - y_onehot) / N
        dU = delta2.T @ h1
        db2 = np.sum(delta2, axis=0)

        delta1 = (delta2 @ self.U) * self.relu_deriv(z1)
        dW = delta1.T @ X
        db1 = np.sum(delta1, axis=0)

        return {"dW": dW, "dU": dU, "db1": db1, "db2": db2}

    def step(self, grads, lr: float):
        self.W -= lr * grads["dW"]
        self.U -= lr * grads["dU"]
        self.b1 -= lr * grads["db1"]
        self.b2 -= lr * grads["db2"]

    def evaluate(self, X: np.ndarray, y_true: np.ndarray):
        probs, _ = self.forward(X)
        acc = accuracy_from_probs(probs, y_true)
        return acc
