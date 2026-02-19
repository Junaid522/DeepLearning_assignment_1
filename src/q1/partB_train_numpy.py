import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.q1.utils import stratified_split, fit_preprocess, transform
from src.q1.partB_numpy_mlp import NumpyMLP

def train_one(Xtr, ytr, Xva, yva, activation, iters, lr, h1, h2, seed):
    model = NumpyMLP(in_dim=Xtr.shape[1], h1=h1, h2=h2, out_dim=len(np.unique(ytr)), activation=activation, seed=seed)
    hist = {"iter": [], "train_acc": [], "val_acc": [], "train_loss": [], "val_loss": [], "gW1": [], "gW2": []}

    for t in range(1, iters + 1):
        cache = model.forward(Xtr)
        tr_loss = model.loss(cache["P"], ytr)
        grads = model.backward(cache, ytr)
        model.step(grads, lr=lr)

        va_loss, va_acc = model.evaluate(Xva, yva)
        tr_acc = float(np.mean(np.argmax(cache["P"], axis=1) == ytr))
        g1, g2 = model.grad_magnitudes(grads)

        hist["iter"].append(t)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(va_acc)
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["gW1"].append(g1)
        hist["gW2"].append(g2)

    return hist

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plots", required=True)
    ap.add_argument("--iters", type=int, default=250)   # >= 200 required
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--h1", type=int, default=64)
    ap.add_argument("--h2", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.plots, exist_ok=True)

    df = pd.read_csv(args.train)
    train_df, val_df = stratified_split(df, val_frac=0.2, seed=args.seed)

    art = fit_preprocess(train_df)
    Xtr, ytr, _ = transform(train_df, art)
    Xva, yva, _ = transform(val_df, art)

    all_hist = {}
    metrics = {}

    for act in ["sigmoid", "relu"]:
        hist = train_one(Xtr, ytr, Xva, yva, activation=act, iters=args.iters, lr=args.lr, h1=args.h1, h2=args.h2, seed=args.seed)
        all_hist[act] = hist
        metrics[act] = {
            "final_train_acc": float(hist["train_acc"][-1]),
            "final_val_acc": float(hist["val_acc"][-1]),
            "final_train_loss": float(hist["train_loss"][-1]),
            "final_val_loss": float(hist["val_loss"][-1]),
        }

    with open(os.path.join(args.out, "numpy_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save gradient magnitudes per iteration
    rows = []
    for act, hist in all_hist.items():
        for i in range(len(hist["iter"])):
            rows.append({
                "activation": act,
                "iter": hist["iter"][i],
                "gW1_mean_abs": hist["gW1"][i],
                "gW2_mean_abs": hist["gW2"][i],
                "train_acc": hist["train_acc"][i],
                "val_acc": hist["val_acc"][i],
            })
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "numpy_gradients.csv"), index=False)

    # required: single plot for train/val accuracy vs iterations
    plt.figure(figsize=(7,4))
    for act in ["sigmoid", "relu"]:
        plt.plot(all_hist[act]["iter"], all_hist[act]["train_acc"], label=f"train ({act})")
        plt.plot(all_hist[act]["iter"], all_hist[act]["val_acc"], linestyle="--", label=f"val ({act})")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Training & validation accuracy (NumPy MLP)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.plots, "B1_accuracy_curves_numpy.png"), dpi=200)
    plt.close()

    # Part B(b): gradient magnitudes plot
    plt.figure(figsize=(7,4))
    for act in ["sigmoid", "relu"]:
        plt.plot(all_hist[act]["iter"], all_hist[act]["gW1"], label=f"|dW1| mean ({act})")
        plt.plot(all_hist[act]["iter"], all_hist[act]["gW2"], linestyle="--", label=f"|dW2| mean ({act})")
    plt.xlabel("Iteration")
    plt.ylabel("Mean(|gradient|)")
    plt.title("Gradient magnitudes across layers (NumPy MLP)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.plots, "B2_gradient_magnitudes_numpy.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
