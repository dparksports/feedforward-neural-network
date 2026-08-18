"""
Supervised Learning Experiment: Hinton's Forward-Forward Algorithm vs Standard Backpropagation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.forward_forward import FFLayer, FFNetwork
from src.network import NeuralNetwork
from src.layers import Dense
from src.activations import ReLU, Softmax
from src.losses import CategoricalCrossEntropyLoss
from src.optimizers import Adam


def generate_multiclass_patterns(n_samples_per_class: int = 150, num_classes: int = 4, dim: int = 32):
    """
    Generates structured pattern vectors for multi-class classification.
    Each class exhibits unique harmonic / modal signatures.
    """
    np.random.seed(42)
    X_list = []
    y_list = []

    t = np.linspace(0, 2 * np.pi, dim - num_classes)

    for c in range(num_classes):
        # Base signature
        signal = np.sin((c + 1) * t) + 0.5 * np.cos(2 * (c + 1) * t)
        noise = np.random.randn(n_samples_per_class, dim - num_classes) * 0.3
        class_data = signal + noise

        # Prefix with zeros for label overlay region
        zeros_prefix = np.zeros((n_samples_per_class, num_classes))
        full_features = np.hstack([zeros_prefix, class_data])

        X_list.append(full_features)
        y_list.append(np.full(n_samples_per_class, c))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    return X[indices[:split]], y[indices[:split]], X[indices[split:]], y[indices[split:]], num_classes, dim


def main():
    print("=" * 75)
    print("EXPERIMENT: Geoffrey Hinton's Forward-Forward (FF) vs. Standard Backprop FFNN")
    print("=" * 75)

    X_train, y_train, X_val, y_val, num_classes, dim = generate_multiclass_patterns()
    print(f"Dataset: {len(X_train)} Train samples, {len(X_val)} Val samples | {num_classes} Classes | {dim} Features")

    # -------------------------------------------------------------
    # 1. Train Hinton's Forward-Forward Network
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("1. Training Hinton's Forward-Forward (FF) Network")
    print("=" * 50)

    ff_net = FFNetwork([
        FFLayer(in_features=dim, out_features=64, threshold=2.0, seed=42),
        FFLayer(in_features=64, out_features=64, threshold=2.0, seed=43),
        FFLayer(in_features=64, out_features=64, threshold=2.0, seed=44),
    ])

    ff_history = ff_net.train_layer_by_layer(
        x_train=X_train,
        y_train=y_train,
        num_classes=num_classes,
        epochs_per_layer=100,
        batch_size=32,
        lr=0.03,
        threshold=2.0,
        verbose=True,
    )

    ff_val_acc = ff_net.evaluate(X_val, y_val, num_classes=num_classes)
    print(f"\n[FF Network] Final Validation Accuracy (via Goodness Accumulation): {ff_val_acc * 100:.2f}%")

    # -------------------------------------------------------------
    # 2. Train Standard Backpropagation MLP Baseline
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("2. Training Standard Backpropagation MLP Baseline")
    print("=" * 50)

    # 1-hot encode targets for standard cross entropy
    y_train_1hot = np.zeros((len(y_train), num_classes))
    y_train_1hot[np.arange(len(y_train)), y_train] = 1.0

    y_val_1hot = np.zeros((len(y_val), num_classes))
    y_val_1hot[np.arange(len(y_val)), y_val] = 1.0

    bp_net = NeuralNetwork([
        Dense(in_features=dim, out_features=64, init_method="he", seed=42),
        ReLU(),
        Dense(in_features=64, out_features=64, init_method="he", seed=43),
        ReLU(),
        Dense(in_features=64, out_features=num_classes, init_method="xavier", seed=44),
        Softmax(),
    ])

    optimizer = Adam(lr=0.01)
    loss_fn = CategoricalCrossEntropyLoss()

    bp_history = bp_net.fit(
        x_train=X_train,
        y_train=y_train_1hot,
        epochs=100,
        batch_size=32,
        optimizer=optimizer,
        loss_fn=loss_fn,
        val_data=(X_val, y_val_1hot),
        verbose=True,
        print_every=25,
    )

    bp_loss, bp_val_acc = bp_net.evaluate(X_val, y_val_1hot, loss_fn)
    print(f"\n[Backpropagation MLP] Final Validation Accuracy: {bp_val_acc * 100:.2f}% | Loss: {bp_loss:.4f}")

    # -------------------------------------------------------------
    # 3. Generate Comparative Plots
    # -------------------------------------------------------------
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#0f172a")

    # Left: FF Layer Training Loss
    ax1.set_facecolor("#1e293b")
    colors = ["#38bdf8", "#ec4899", "#8b5cf6"]
    for l_idx, losses in enumerate(ff_history["layer_losses"]):
        ax1.plot(losses, label=f"FF Layer {l_idx + 1} Local Loss", color=colors[l_idx], linewidth=2.0)
    ax1.set_xlabel("Epoch", fontsize=11, color="#cbd5e1")
    ax1.set_ylabel("Contrastive Loss", fontsize=11, color="#cbd5e1")
    ax1.set_title("Hinton's FF: Local Layer-by-Layer Convergence", fontsize=13, color="#f8fafc", weight="bold")
    ax1.grid(True, linestyle=":", alpha=0.3, color="#94a3b8")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155")
    ax1.tick_params(colors="#cbd5e1")

    # Right: Accuracy Comparison Bar Chart
    ax2.set_facecolor("#1e293b")
    models = ["Hinton's Forward-Forward\n(Forward-Only Goodness)", "Standard Backpropagation\n(Global Chain-Rule MLP)"]
    accuracies = [ff_val_acc * 100, bp_val_acc * 100]
    bar_colors = ["#06b6d4", "#a855f7"]

    bars = ax2.bar(models, accuracies, color=bar_colors, width=0.45)
    ax2.set_ylabel("Validation Accuracy (%)", fontsize=11, color="#cbd5e1")
    ax2.set_title("Classification Accuracy Comparison", fontsize=13, color="#f8fafc", weight="bold")
    ax2.set_ylim(0, 115)
    ax2.grid(True, linestyle=":", alpha=0.3, color="#94a3b8")
    ax2.tick_params(colors="#cbd5e1")

    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2.0, f"{yval:.2f}%", ha="center", va="bottom", color="#f8fafc", weight="bold", fontsize=12)

    os.makedirs("assets", exist_ok=True)
    out_plot = "assets/ff_vs_backprop_experiment.png"
    plt.suptitle("Hinton's Forward-Forward Algorithm vs. Standard Backprop", fontsize=16, color="#f8fafc", weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"\nSaved empirical comparison figure to: {out_plot}")


if __name__ == "__main__":
    main()
