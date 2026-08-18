"""
Example: Training a Feedforward Neural Network on the non-linear 3-arm spiral dataset.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from src.network import NeuralNetwork
from src.layers import Dense
from src.activations import ReLU, Softmax
from src.losses import CategoricalCrossEntropyLoss
from src.optimizers import Adam


def generate_spiral_dataset(points_per_arm: int = 200, num_classes: int = 3, noise: float = 0.2):
    """Generates the 2D multi-class spiral dataset."""
    N = points_per_arm
    K = num_classes
    X = np.zeros((N * K, 2))
    y = np.zeros(N * K, dtype=int)

    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1.0, N)  # Radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * noise  # Theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    # One-hot encode y
    y_one_hot = np.zeros((N * K, K))
    y_one_hot[np.arange(N * K), y] = 1.0
    return X, y_one_hot, y


def main():
    print("=" * 60)
    print("Training Feedforward Neural Network on 3-Class Spiral Dataset")
    print("=" * 60)

    # 1. Dataset Generation
    np.random.seed(42)
    X, y_1hot, y_labels = generate_spiral_dataset(points_per_arm=300, num_classes=3, noise=0.15)

    # 80/20 train/test split
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, y_train = X[train_idx], y_1hot[train_idx]
    X_val, y_val = X[val_idx], y_1hot[val_idx]

    print(f"Dataset: {len(X)} samples (Train: {len(X_train)}, Val: {len(X_val)})")

    # 2. Model Architecture
    model = NeuralNetwork([
        Dense(in_features=2, out_features=64, init_method="he"),
        ReLU(),
        Dense(in_features=64, out_features=32, init_method="he"),
        ReLU(),
        Dense(in_features=32, out_features=3, init_method="xavier"),
        Softmax(),
    ])

    optimizer = Adam(lr=0.01)
    loss_fn = CategoricalCrossEntropyLoss()

    # 3. Training Loop
    print("\nStarting Training...")
    history = model.fit(
        x_train=X_train,
        y_train=y_train,
        epochs=300,
        batch_size=32,
        optimizer=optimizer,
        loss_fn=loss_fn,
        val_data=(X_val, y_val),
        verbose=True,
        print_every=50,
    )

    # 4. Final Evaluation
    val_loss, val_acc = model.evaluate(X_val, y_val, loss_fn)
    print("\n" + "=" * 60)
    print(f"Final Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc * 100:.2f}%")
    print("=" * 60)

    # 5. Save Decision Boundary and Loss Curves Plot
    plt.figure(figsize=(12, 5))

    # Loss & Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="royalblue")
    plt.plot(history["val_loss"], label="Val Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cross-Entropy Loss vs Epochs")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Decision Boundary Plot
    plt.subplot(1, 2, 2)
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_classes(grid_points)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y_labels, s=25, cmap=plt.cm.Spectral, edgecolors="k", linewidth=0.5)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"Spiral Decision Boundary (Val Acc: {val_acc*100:.1f}%)")

    os.makedirs("artifacts", exist_ok=True)
    plot_path = "artifacts/spiral_decision_boundary.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print(f"Saved visual benchmark plot to: {plot_path}")


if __name__ == "__main__":
    main()
