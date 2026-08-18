"""
Activation Functions for Feedforward Neural Network.
Provides forward computation and analytical backward derivatives.
"""

import numpy as np


class Activation:
    """Base class for all activation functions."""

    def __init__(self):
        self.input_cache = None
        self.output_cache = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.forward(z)


class ReLU(Activation):
    r"""
    Rectified Linear Unit:
    f(z) = max(0, z)
    f'(z) = 1 if z > 0 else 0
    """

    def forward(self, z: np.ndarray) -> np.ndarray:
        self.input_cache = z
        return np.maximum(0.0, z)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        # Gradient is 1 for z > 0, 0 otherwise
        dz = np.where(self.input_cache > 0.0, 1.0, 0.0)
        return d_out * dz


class LeakyReLU(Activation):
    r"""
    Leaky Rectified Linear Unit:
    f(z) = max(alpha * z, z)
    f'(z) = 1 if z > 0 else alpha
    """

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, z: np.ndarray) -> np.ndarray:
        self.input_cache = z
        return np.where(z > 0.0, z, self.alpha * z)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        dz = np.where(self.input_cache > 0.0, 1.0, self.alpha)
        return d_out * dz


class Sigmoid(Activation):
    r"""
    Logistic Sigmoid:
    \sigma(z) = \frac{1}{1 + e^{-z}}
    \sigma'(z) = \sigma(z) * (1 - \sigma(z))
    """

    def forward(self, z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid computation avoiding overflow
        z_clipped = np.clip(z, -500.0, 500.0)
        out = 1.0 / (1.0 + np.exp(-z_clipped))
        self.output_cache = out
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        s = self.output_cache
        return d_out * (s * (1.0 - s))


class Tanh(Activation):
    r"""
    Hyperbolic Tangent:
    \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
    \tanh'(z) = 1 - \tanh^2(z)
    """

    def forward(self, z: np.ndarray) -> np.ndarray:
        out = np.tanh(z)
        self.output_cache = out
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        t = self.output_cache
        return d_out * (1.0 - t**2)


class Softmax(Activation):
    r"""
    Softmax activation for multi-class probability distribution:
    S(z)_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}

    Jacobian: \frac{\partial S_i}{\partial z_j} = S_i (\delta_{ij} - S_j)
    Vectorized backward: dZ = S * (d_out - \sum (d_out * S, axis=-1, keepdims=True))
    """

    def forward(self, z: np.ndarray) -> np.ndarray:
        # Shift inputs for numerical stability (prevent exponential overflow)
        shift_z = z - np.max(z, axis=-1, keepdims=True)
        exps = np.exp(shift_z)
        out = exps / np.sum(exps, axis=-1, keepdims=True)
        self.output_cache = out
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        s = self.output_cache
        # dZ_i = \sum_j d_out_j * \partial S_j / \partial z_i
        # Vectorized formula for batch:
        sum_d_out_s = np.sum(d_out * s, axis=-1, keepdims=True)
        return s * (d_out - sum_d_out_s)
