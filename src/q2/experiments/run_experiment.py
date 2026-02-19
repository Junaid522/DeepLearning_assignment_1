import numpy as np
from ..models.shared_bias_mlp import SharedBiasMLP
from ..models.independent_bias_mlp import IndependentBiasMLP
from ..utils.data import make_synthetic, train_val_split, one_hot
from ..utils.gradcheck import gradcheck_shared_bias, gradcheck_independent_bias
from .train_compare import train_model
from ..utils.plotting import plot_curves


def main():
    np.random.seed(0)

    # small dataset for gradient checking
    Xg = np.random.randn(10, 5).astype(np.float64)
    yg = np.random.randint(0, 4, size=10).astype(int)
    yg_oh = one_hot(yg, 4)

    print("\n=== Gradient Check ===")
    shared = SharedBiasMLP(input_dim=5, hidden_dim=4, output_dim=4, seed=0)
    num, ana, rel = gradcheck_shared_bias(shared, Xg, yg_oh)
    print("Shared bias relative error:", rel)

    indep = IndependentBiasMLP(input_dim=5, hidden_dim=4, output_dim=4, seed=0)
    (n1, a1, r1), (n2, a2, r2) = gradcheck_independent_bias(indep, Xg, yg_oh)
    print("Independent bias b1 relative error:", r1)
    print("Independent bias b2 relative error:", r2)

    # training experiment
    print("\n=== Training Compare (Shared vs Independent) ===")
    X, y = make_synthetic(N=800, d=5, k=4, seed=1)
    Xtr, ytr, Xva, yva = train_val_split(X, y, val_ratio=0.2, seed=1)

    shared2 = SharedBiasMLP(input_dim=5, hidden_dim=4, output_dim=4, seed=1)
    indep2 = IndependentBiasMLP(input_dim=5, hidden_dim=4, output_dim=4, seed=1)

    hist_shared = train_model(shared2, Xtr, ytr, Xva, yva, lr=0.1, steps=400, log_every=20)
    hist_indep = train_model(indep2, Xtr, ytr, Xva, yva, lr=0.1, steps=400, log_every=20)

    print("Shared final:", "loss=", hist_shared["val_loss"][-1], "acc=", hist_shared["val_acc"][-1])
    print("Indep  final:", "loss=", hist_indep["val_loss"][-1], "acc=", hist_indep["val_acc"][-1])

    # plots
    plot_curves(hist_shared, "results/q2/shared.png", "Shared Bias Model")
    plot_curves(hist_indep, "results/q2/independent.png", "Independent Bias Model")

    print("\nSaved plots to: results/q2/ (shared_loss.png, shared_acc.png, independent_loss.png, independent_acc.png)")


if __name__ == "__main__":
    main()
