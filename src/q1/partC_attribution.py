import numpy as np
import pandas as pd
import torch

def gradient_feature_attribution(model, X: np.ndarray, y: np.ndarray, feature_names):
    """
    Importance(i) = mean over samples of | dL/dx_i |
    """
    model.eval()
    x_t = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    y_t = torch.tensor(y, dtype=torch.long)

    logits = model(x_t)
    loss = torch.nn.functional.cross_entropy(logits, y_t)
    loss.backward()

    grads = x_t.grad.detach().cpu().numpy()
    imp = np.mean(np.abs(grads), axis=0)

    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df
