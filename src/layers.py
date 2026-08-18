"""
Layer Abstractions for Feedforward Neural Network.
Includes Dense (Fully Connected) Layer with gradient tracking and weight initializations.
"""

from typing import Dict, Optional
import numpy as np


class Layer:
    """Base class for all trainable network layers."""

    def __init__(self):
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Dense(Layer):
    r"""
    Fully Connected (Affine) Layer:
    Forward:
        Z = X \cdot W + b
        where X \in \mathbb{R}^{B \times D_{in}},
              W \in \mathbb{R}^{D_{in} \times D_{out}},
              b \in \mathbb{R}^{1 \times D_{out}},
              Z \in \mathbb{R}^{B \times D_{out}}

    Backward:
        \nabla_W = X^T \cdot dZ
        \nabla_b = \sum_{batch} dZ
        \nabla_X = dZ \cdot W^T
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_method: str = "he",
        weight_decay: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_decay = weight_decay

        rng = np.random.default_rng(seed)

        # Weight Initialization
        if init_method.lower() in ("he", "kaiming"):
            # Recommended for ReLU activations: std = sqrt(2 / in_features)
            std = np.sqrt(2.0 / in_features)
            self.W = rng.normal(0.0, std, (in_features, out_features))
        elif init_method.lower() in ("xavier", "glorot"):
            # Recommended for Sigmoid/Tanh/Softmax: std = sqrt(2 / (in_features + out_features))
            std = np.sqrt(2.0 / (in_features + out_features))
            self.W = rng.normal(0.0, std, (in_features, out_features))
        elif init_method.lower() == "normal":
            self.W = rng.normal(0.0, 0.01, (in_features, out_features))
        elif init_method.lower() == "zeros":
            self.W = np.zeros((in_features, out_features))
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")

        # Bias initialized to zeros
        self.b = np.zeros((1, out_features))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.params = {"W": self.W, "b": self.b}
        self.grads = {"W": self.dW, "b": self.db}

        self.input_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x
        return np.dot(x, self.W) + self.b

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        assert self.input_cache is not None, "Must call forward before backward."

        # Compute parameter gradients
        self.dW = np.dot(self.input_cache.T, d_out)
        self.db = np.sum(d_out, axis=0, keepdims=True)

        # Optional L2 regularization / weight decay gradient
        if self.weight_decay > 0.0:
            self.dW += self.weight_decay * self.W

        # Update dictionary references
        self.grads["W"] = self.dW
        self.grads["b"] = self.db

        # Compute input gradient for downstream backpropagation
        dx = np.dot(d_out, self.W.T)
        return dx
