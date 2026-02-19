import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from src.q1.utils import stratified_split, fit_preprocess, transform
from src.q1.partC_pytorch_mlp import TorchMLP
from src.q1.partC_attribution import gradient_feature_attribution

def acc_from_logits(logits, y):
    pred = torch.argmax(logits, dim=1)
    return float((pred == y).float().mean().item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plots", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--h1", type=int, default=64)
    ap.add_argument("--h2", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.plots, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.train)
    train_df, val_df = stratified_split(df, val_frac=0.2, seed=args.seed)

    art = fit_preprocess(train_df)
    Xtr, ytr, feat_names = transform(train_df, art)
    Xva, yva, _ = transform(val_df, art)

    test_df = pd.read_csv(args.test)
    Xte, yte, _ = transform(test_df, art)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TorchMLP(in_dim=Xtr.shape[1], h1=args.h1, h2=args.h2, out_dim=len(np.unique(ytr))).to(device)

    # opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)),
        batch_size=args.batch,
        shuffle=True
    )

    history = {"epoch": [], "train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * xb.size(0)
            total_correct += int((torch.argmax(logits, dim=1) == yb).sum().item())
            total_n += xb.size(0)

        train_loss = total_loss / total_n
        train_acc = total_correct / total_n

        model.eval()
        with torch.no_grad():
            va_logits = model(torch.tensor(Xva, dtype=torch.float32).to(device))
            va_loss = float(loss_fn(va_logits, torch.tensor(yva, dtype=torch.long).to(device)).item())
            va_acc = acc_from_logits(va_logits, torch.tensor(yva, dtype=torch.long).to(device))

        history["epoch"].append(ep)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(va_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(va_loss)

    # Test accuracy (required Part D)
    model.eval()
    with torch.no_grad():
        te_logits = model(torch.tensor(Xte, dtype=torch.float32).to(device))
        te_acc = acc_from_logits(te_logits, torch.tensor(yte, dtype=torch.long).to(device))

    metrics = {
        "final_train_acc": float(history["train_acc"][-1]),
        "final_val_acc": float(history["val_acc"][-1]),
        "test_acc": float(te_acc),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch": args.batch,
        "h1": args.h1,
        "h2": args.h2,
        "input_dim": int(Xtr.shape[1]),
    }
    with open(os.path.join(args.out, "torch_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # accuracy curve plot
    plt.figure(figsize=(7,4))
    plt.plot(history["epoch"], history["train_acc"], label="train")
    plt.plot(history["epoch"], history["val_acc"], linestyle="--", label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & validation accuracy (PyTorch MLP)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.plots, "C1_accuracy_curves_torch.png"), dpi=200)
    plt.close()

    # gradient attribution (Part C(b))
    imp_df = gradient_feature_attribution(model, Xva, yva, feat_names)
    imp_df.to_csv(os.path.join(args.out, "feature_importance.csv"), index=False)

    # top features plot
    top = imp_df.head(20).iloc[::-1]
    plt.figure(figsize=(7,5))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Mean(|dL/dx|)")
    plt.title("Top 20 features (gradient attribution, validation)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.plots, "C2_top_features.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
