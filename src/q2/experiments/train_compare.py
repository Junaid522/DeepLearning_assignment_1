import numpy as np
from ..utils.data import one_hot
from ..utils.losses import cross_entropy_onehot
from ..utils.metrics import accuracy_from_probs


def train_model(model, Xtr, ytr, Xva, yva, lr=0.1, steps=400, log_every=20):
    ytr_oh = one_hot(ytr, num_classes=len(np.unique(ytr)))
    yva_oh = one_hot(yva, num_classes=ytr_oh.shape[1])

    history = {"steps": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for step in range(1, steps + 1):
        probs_tr, cache = model.forward(Xtr)
        grads = model.backward(cache, ytr_oh)
        model.step(grads, lr=lr)

        if step % log_every == 0 or step == 1 or step == steps:
            probs_tr, _ = model.forward(Xtr)
            probs_va, _ = model.forward(Xva)

            tr_loss = cross_entropy_onehot(probs_tr, ytr_oh)
            va_loss = cross_entropy_onehot(probs_va, yva_oh)

            tr_acc = accuracy_from_probs(probs_tr, ytr)
            va_acc = accuracy_from_probs(probs_va, yva)

            history["steps"].append(step)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)

    return history
