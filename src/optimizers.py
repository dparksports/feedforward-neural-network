"""
Optimization Algorithms for Neural Network Parameter Updates.
Implements SGD (with Momentum), RMSprop, and Adam.
"""

from typing import Dict, List
import numpy as np


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, layers: List) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    r"""
    Stochastic Gradient Descent optimizer with optional momentum.
    Velocity update:
        v = \beta v + \nabla \theta
    Parameter update:
        \theta \leftarrow \theta - \eta v
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        super().__init__(lr=lr)
        self.momentum = momentum
        self.velocities: List[Dict[str, np.ndarray]] = []

    def step(self, layers: List) -> None:
        if not self.velocities:
            for layer in layers:
                if hasattr(layer, "params") and layer.params:
                    vel = {k: np.zeros_like(v) for k, v in layer.params.items()}
                    self.velocities.append(vel)
                else:
                    self.velocities.append({})

        for layer, vel in zip(layers, self.velocities):
            if not hasattr(layer, "params") or not layer.params:
                continue

            for param_name in layer.params:
                param = layer.params[param_name]
                grad = layer.grads[param_name]

                if self.momentum > 0.0:
                    vel[param_name] = self.momentum * vel[param_name] + grad
                    param -= self.lr * vel[param_name]
                else:
                    param -= self.lr * grad


class RMSprop(Optimizer):
    r"""
    RMSprop Optimizer:
    v = \beta v + (1 - \beta) (\nabla \theta)^2
    \theta \leftarrow \theta - \frac{\eta}{\sqrt{v} + \epsilon} \nabla \theta
    """

    def __init__(self, lr: float = 0.001, beta: float = 0.9, eps: float = 1e-8):
        super().__init__(lr=lr)
        self.beta = beta
        self.eps = eps
        self.cache: List[Dict[str, np.ndarray]] = []

    def step(self, layers: List) -> None:
        if not self.cache:
            for layer in layers:
                if hasattr(layer, "params") and layer.params:
                    c = {k: np.zeros_like(v) for k, v in layer.params.items()}
                    self.cache.append(c)
                else:
                    self.cache.append({})

        for layer, c in zip(layers, self.cache):
            if not hasattr(layer, "params") or not layer.params:
                continue

            for param_name in layer.params:
                param = layer.params[param_name]
                grad = layer.grads[param_name]

                c[param_name] = self.beta * c[param_name] + (1.0 - self.beta) * (grad**2)
                param -= (self.lr / (np.sqrt(c[param_name]) + self.eps)) * grad


class Adam(Optimizer):
    r"""
    Adam (Adaptive Moment Estimation) Optimizer:
    m = \beta_1 m + (1 - \beta_1) \nabla \theta
    v = \beta_2 v + (1 - \beta_2) (\nabla \theta)^2
    \hat{m} = \frac{m}{1 - \beta_1^t}
    \hat{v} = \frac{v}{1 - \beta_2^t}
    \theta \leftarrow \theta - \frac{\eta}{\sqrt{\hat{v}} + \epsilon} \hat{m}
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__(lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m: List[Dict[str, np.ndarray]] = []
        self.v: List[Dict[str, np.ndarray]] = []

    def step(self, layers: List) -> None:
        self.t += 1

        if not self.m:
            for layer in layers:
                if hasattr(layer, "params") and layer.params:
                    self.m.append({k: np.zeros_like(val) for k, val in layer.params.items()})
                    self.v.append({k: np.zeros_like(val) for k, val in layer.params.items()})
                else:
                    self.m.append({})
                    self.v.append({})

        for i, layer in enumerate(layers):
            if not hasattr(layer, "params") or not layer.params:
                continue

            for param_name in layer.params:
                param = layer.params[param_name]
                grad = layer.grads[param_name]

                # Update biased first moment estimate
                self.m[i][param_name] = self.beta1 * self.m[i][param_name] + (1.0 - self.beta1) * grad
                # Update biased second raw moment estimate
                self.v[i][param_name] = self.beta2 * self.v[i][param_name] + (1.0 - self.beta2) * (grad**2)

                # Compute bias-corrected estimates
                m_hat = self.m[i][param_name] / (1.0 - self.beta1**self.t)
                v_hat = self.v[i][param_name] / (1.0 - self.beta2**self.t)

                # Update parameters
                param -= (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat
