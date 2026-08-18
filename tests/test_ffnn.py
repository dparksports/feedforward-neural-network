"""
Unit and Integration Tests for Feedforward Neural Network.
"""

import numpy as np
import pytest
from src.activations import ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
from src.losses import MSELoss, BinaryCrossEntropyLoss, CategoricalCrossEntropyLoss
from src.layers import Dense
from src.optimizers import SGD, Adam, RMSprop
from src.network import NeuralNetwork
from src.grad_check import gradient_check


def test_dense_layer_shapes():
    """Verify matrix shapes during forward and backward passes."""
    batch_size = 8
    in_dim = 4
    out_dim = 16

    dense = Dense(in_dim, out_dim, init_method="he")
    x = np.random.randn(batch_size, in_dim)
    z = dense.forward(x)

    assert z.shape == (batch_size, out_dim)

    d_out = np.random.randn(batch_size, out_dim)
    dx = dense.backward(d_out)

    assert dx.shape == (batch_size, in_dim)
    assert dense.dW.shape == (in_dim, out_dim)
    assert dense.db.shape == (1, out_dim)


def test_activations():
    """Verify activation output bounds and shapes."""
    x = np.array([[-2.0, 0.0, 3.0]])

    relu = ReLU()
    assert np.allclose(relu.forward(x), [[0.0, 0.0, 3.0]])
    assert np.allclose(relu.backward(np.ones_like(x)), [[0.0, 0.0, 1.0]])

    leaky = LeakyReLU(alpha=0.1)
    assert np.allclose(leaky.forward(x), [[-0.2, 0.0, 3.0]])
    assert np.allclose(leaky.backward(np.ones_like(x)), [[0.1, 0.1, 1.0]])

    sig = Sigmoid()
    s_out = sig.forward(x)
    assert np.all((s_out >= 0.0) & (s_out <= 1.0))

    tanh = Tanh()
    t_out = tanh.forward(x)
    assert np.all((t_out >= -1.0) & (t_out <= 1.0))

    sm = Softmax()
    sm_out = sm.forward(x)
    assert np.isclose(np.sum(sm_out), 1.0)


def test_gradient_checking_mse():
    """Verify analytical backpropagation gradients match finite-difference gradients."""
    np.random.seed(42)
    x = np.random.randn(5, 3)
    y = np.random.randn(5, 2)

    net = NeuralNetwork([
        Dense(3, 8, init_method="xavier"),
        Tanh(),
        Dense(8, 2, init_method="xavier")
    ])

    loss_fn = MSELoss()
    passed, errors = gradient_check(net, x, y, loss_fn, epsilon=1e-5, tolerance=1e-5, verbose=False)
    assert passed, f"Gradient check failed with errors: {errors}"


def test_gradient_checking_classification():
    """Verify analytical gradients for multi-layer ReLU network on classification."""
    np.random.seed(123)
    x = np.random.randn(6, 4)
    # Binary classification labels
    y = np.random.randint(0, 2, size=(6, 1)).astype(float)

    net = NeuralNetwork([
        Dense(4, 6, init_method="he"),
        ReLU(),
        Dense(6, 1, init_method="xavier"),
        Sigmoid()
    ])

    loss_fn = BinaryCrossEntropyLoss()
    passed, errors = gradient_check(net, x, y, loss_fn, epsilon=1e-5, tolerance=1e-4, verbose=False)
    assert passed, f"Gradient check failed with errors: {errors}"


def test_xor_convergence():
    """Test that a 2-layer network solves the non-linear XOR problem to 100% accuracy."""
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])

    np.random.seed(42)
    net = NeuralNetwork([
        Dense(2, 8, init_method="he"),
        ReLU(),
        Dense(8, 1, init_method="xavier"),
        Sigmoid()
    ])

    opt = Adam(lr=0.05)
    loss_fn = BinaryCrossEntropyLoss()

    history = net.fit(X, y, epochs=250, batch_size=4, optimizer=opt, loss_fn=loss_fn, verbose=False)

    preds = net.predict_classes(X)
    assert np.array_equal(preds, y.astype(int)), f"XOR failed. Predicted {preds.ravel()}, expected {y.ravel()}"
    assert history["train_loss"][-1] < 0.1, "Loss did not converge sufficiently on XOR"


def test_multiclass_convergence():
    """Test 3-class classification convergence with Softmax and Cross-Entropy."""
    np.random.seed(42)
    # Generate 3 clusters in 2D
    c1 = np.random.randn(50, 2) + np.array([-3, -3])
    c2 = np.random.randn(50, 2) + np.array([3, -3])
    c3 = np.random.randn(50, 2) + np.array([0, 4])
    X = np.vstack([c1, c2, c3])
    y = np.array([0] * 50 + [1] * 50 + [2] * 50)

    # 1-hot encode y
    y_1hot = np.zeros((150, 3))
    y_1hot[np.arange(150), y] = 1.0

    net = NeuralNetwork([
        Dense(2, 16, init_method="he"),
        ReLU(),
        Dense(16, 3, init_method="xavier"),
        Softmax()
    ])

    opt = Adam(lr=0.02)
    loss_fn = CategoricalCrossEntropyLoss()

    history = net.fit(X, y_1hot, epochs=150, batch_size=32, optimizer=opt, loss_fn=loss_fn, verbose=False)
    loss, acc = net.evaluate(X, y_1hot, loss_fn)

    assert acc >= 0.95, f"Expected >95% accuracy on 3-cluster dataset, got {acc*100:.2f}%"
    assert history["train_loss"][-1] < 0.3


def test_optimizers():
    """Verify SGD with momentum and RMSprop parameter updates reduce loss."""
    np.random.seed(42)
    x = np.random.randn(20, 4)
    y = np.random.randn(20, 2)

    for opt_cls in [lambda: SGD(lr=0.01, momentum=0.9), lambda: RMSprop(lr=0.01)]:
        net = NeuralNetwork([
            Dense(4, 8, init_method="xavier"),
            Tanh(),
            Dense(8, 2, init_method="xavier")
        ])
        loss_fn = MSELoss()
        opt = opt_cls()
        init_loss = loss_fn.forward(net.forward(x), y)
        history = net.fit(x, y, epochs=20, batch_size=10, optimizer=opt, loss_fn=loss_fn, verbose=False)
        final_loss = history["train_loss"][-1]
        assert final_loss < init_loss, f"Optimizer {opt_cls} did not reduce loss ({init_loss:.4f} -> {final_loss:.4f})"

