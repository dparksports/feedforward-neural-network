"""
Unit and Integration Tests for Hinton's Forward-Forward (FF) Algorithm.
"""

import numpy as np
import pytest
from src.forward_forward import FFLayer, FFNetwork


def test_ff_layer_normalization_and_goodness():
    """Verify that L2 normalization produces unit norm vectors and goodness is sum of squares."""
    layer = FFLayer(in_features=8, out_features=16, threshold=2.0)
    x = np.random.randn(10, 8)

    h = layer.forward(x)
    assert h.shape == (10, 16)
    assert np.all(h >= 0.0), "ReLU output should be non-negative"

    h_norm = layer.normalize(h)
    norms = np.linalg.norm(h_norm, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"Normalized activations should have unit norm, got {norms}"

    g = layer.goodness(h)
    assert g.shape == (10, 1)
    expected_g = np.sum(h**2, axis=1, keepdims=True)
    assert np.allclose(g, expected_g)


def test_ff_layer_local_train_step():
    """Verify that a single local train_step increases pos goodness and decreases neg goodness."""
    np.random.seed(42)
    layer = FFLayer(in_features=10, out_features=20, threshold=4.0)

    x_pos = np.random.randn(30, 10) + 1.0
    x_neg = np.random.randn(30, 10) - 1.0

    _, initial_g_pos, initial_g_neg = layer.compute_loss(layer.forward(x_pos), layer.forward(x_neg))

    # Train local layer for 50 steps
    for _ in range(50):
        layer.train_step(x_pos, x_neg, lr=0.05)

    _, final_g_pos, final_g_neg = layer.compute_loss(layer.forward(x_pos), layer.forward(x_neg))

    assert final_g_pos > initial_g_pos, f"Positive goodness should increase: {initial_g_pos:.2f} -> {final_g_pos:.2f}"
    assert final_g_neg < initial_g_neg or final_g_neg < layer.threshold


def test_ff_network_supervised_classification():
    """Verify that FFNetwork learns a non-trivial 3-class pattern classification problem without backprop."""
    np.random.seed(123)
    n_samples_per_class = 60
    num_classes = 3
    dim = 16

    # Create 3 distinct class clusters
    X_list = []
    y_list = []
    for c in range(num_classes):
        cluster = np.random.randn(n_samples_per_class, dim) * 0.5
        cluster[:, num_classes + c] += 3.0  # distinctive feature in data region
        X_list.append(cluster)
        y_list.append(np.full(n_samples_per_class, c))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Train/Val split
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, y_train = X[indices[:split]], y[indices[:split]]
    X_val, y_val = X[indices[split:]], y[indices[split:]]

    net = FFNetwork([
        FFLayer(in_features=dim, out_features=32, threshold=2.0),
        FFLayer(in_features=32, out_features=32, threshold=2.0),
    ])

    net.train_layer_by_layer(
        x_train=X_train,
        y_train=y_train,
        num_classes=num_classes,
        epochs_per_layer=80,
        batch_size=32,
        lr=0.03,
        threshold=2.0,
        verbose=False,
    )

    acc = net.evaluate(X_val, y_val, num_classes=num_classes)
    assert acc >= 0.90, f"Expected Forward-Forward network accuracy >= 90%, got {acc * 100:.2f}%"
