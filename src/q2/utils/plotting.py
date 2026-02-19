import os
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_curves(history, out_path: str, title: str):
    """
    history: dict with keys:
      steps, train_loss, val_loss, train_acc, val_acc
    """
    ensure_dir(os.path.dirname(out_path))

    steps = history["steps"]

    plt.figure()
    plt.plot(steps, history["train_loss"], label="train loss")
    plt.plot(steps, history["val_loss"], label="val loss")
    plt.title(title + " - Loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(steps, history["train_acc"], label="train acc")
    plt.plot(steps, history["val_acc"], label="val acc")
    plt.title(title + " - Accuracy")
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_acc.png"))
    plt.close()
