import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
from src.q1.utils import softmax, one_hot, accuracy

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def d_sigmoid(a):
    return a * (1.0 - a)

def relu(z):
    return np.maximum(0.0, z)

def d_relu(z):
    return (z > 0.0).astype(np.float64)

@dataclass
class NumpyMLP:
    in_dim: int
    h1: int
    h2: int
    out_dim: int
    activation: str = "relu"
    seed: int = 42

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)

        # basic init
        if self.activation == "relu":
            self.W1 = rng.normal(0, np.sqrt(2/self.in_dim), size=(self.in_dim, self.h1))
            self.W2 = rng.normal(0, np.sqrt(2/self.h1), size=(self.h1, self.h2))
        else:
            self.W1 = rng.normal(0, np.sqrt(1/self.in_dim), size=(self.in_dim, self.h1))
            self.W2 = rng.normal(0, np.sqrt(1/self.h1), size=(self.h1, self.h2))

        self.W3 = rng.normal(0, np.sqrt(1/self.h2), size=(self.h2, self.out_dim))

        self.b1 = np.zeros((1, self.h1))
        self.b2 = np.zeros((1, self.h2))
        self.b3 = np.zeros((1, self.out_dim))

    def _act(self, z):
        return relu(z) if self.activation == "relu" else sigmoid(z)

    def _d_act(self, z, a):
        return d_relu(z) if self.activation == "relu" else d_sigmoid(a)

    def forward(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        Z1 = X @ self.W1 + self.b1
        A1 = self._act(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = self._act(Z2)
        Z3 = A2 @ self.W3 + self.b3
        P = softmax(Z3)
        return {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "P": P}

    def loss(self, P: np.ndarray, y: np.ndarray) -> float:
        n = y.shape[0]
        eps = 1e-12
        return float(-np.mean(np.log(P[np.arange(n), y] + eps)))

    def backward(self, cache: Dict[str, np.ndarray], y: np.ndarray) -> Dict[str, np.ndarray]:
        X, Z1, A1, Z2, A2, P = cache["X"], cache["Z1"], cache["A1"], cache["Z2"], cache["A2"], cache["P"]
        n = X.shape[0]

        Y = one_hot(y, self.out_dim)

        # softmax + cross entropy gradient
        dZ3 = (P - Y) / n
        dW3 = A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * self._d_act(Z2, A2)
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self._d_act(Z1, A1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}

    def step(self, grads: Dict[str, np.ndarray], lr: float):
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]
        self.W3 -= lr * grads["dW3"]
        self.b3 -= lr * grads["db3"]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)["P"]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        P = self.predict_proba(X)
        return self.loss(P, y), accuracy(y, P)

    def grad_magnitudes(self, grads: Dict[str, np.ndarray]) -> Tuple[float, float]:
        return float(np.mean(np.abs(grads["dW1"]))), float(np.mean(np.abs(grads["dW2"])))
