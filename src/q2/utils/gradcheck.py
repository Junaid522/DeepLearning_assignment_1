import numpy as np


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(a) + np.linalg.norm(b) + 1e-12
    return float(num / den)


def gradcheck_shared_bias(model, X, y_onehot, eps=1e-5):
    """
    Finite differences check for shared bias b.
    Returns numeric_grad, analytic_grad, rel_error
    """
    probs, cache = model.forward(X)
    grads = model.backward(cache, y_onehot)
    analytic = grads["db"].copy()

    numeric = np.zeros_like(model.b)

    for i in range(len(model.b)):
        old = model.b[i]

        model.b[i] = old + eps
        p1, _ = model.forward(X)
        L1 = model.loss(p1, y_onehot)

        model.b[i] = old - eps
        p2, _ = model.forward(X)
        L2 = model.loss(p2, y_onehot)

        numeric[i] = (L1 - L2) / (2 * eps)
        model.b[i] = old

    return numeric, analytic, relative_error(numeric, analytic)


def gradcheck_independent_bias(model, X, y_onehot, eps=1e-5):
    """
    Finite differences check for b1 and b2.
    Returns (num_b1, ana_b1, rel_b1), (num_b2, ana_b2, rel_b2)
    """
    probs, cache = model.forward(X)
    grads = model.backward(cache, y_onehot)
    ana_b1 = grads["db1"].copy()
    ana_b2 = grads["db2"].copy()

    # b1
    num_b1 = np.zeros_like(model.b1)
    for i in range(len(model.b1)):
        old = model.b1[i]

        model.b1[i] = old + eps
        p1, _ = model.forward(X)
        L1 = model.loss(p1, y_onehot)

        model.b1[i] = old - eps
        p2, _ = model.forward(X)
        L2 = model.loss(p2, y_onehot)

        num_b1[i] = (L1 - L2) / (2 * eps)
        model.b1[i] = old

    # b2
    num_b2 = np.zeros_like(model.b2)
    for i in range(len(model.b2)):
        old = model.b2[i]

        model.b2[i] = old + eps
        p1, _ = model.forward(X)
        L1 = model.loss(p1, y_onehot)

        model.b2[i] = old - eps
        p2, _ = model.forward(X)
        L2 = model.loss(p2, y_onehot)

        num_b2[i] = (L1 - L2) / (2 * eps)
        model.b2[i] = old

    return (num_b1, ana_b1, relative_error(num_b1, ana_b1)), (num_b2, ana_b2, relative_error(num_b2, ana_b2))
