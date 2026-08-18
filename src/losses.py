"""
Loss Functions for Feedforward Neural Network.
Provides forward loss calculation and analytical gradients w.r.t predictions.
"""

import numpy as np


class Loss:
    """Base class for loss functions."""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return self.forward(y_pred, y_true)


class MSELoss(Loss):
    r"""
    Mean Squared Error Loss:
    L(y_pred, y_true) = \frac{1}{2N} \sum_{i=1}^N \|y_pred_i - y_true_i\|^2
    \frac{\partial L}{\partial y_pred} = \frac{1}{N} (y_pred - y_true)
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        N = y_pred.shape[0]
        return float(0.5 * np.sum((y_pred - y_true) ** 2) / N)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        N = y_pred.shape[0]
        return (y_pred - y_true) / N


class BinaryCrossEntropyLoss(Loss):
    r"""
    Binary Cross-Entropy Loss:
    L = - \frac{1}{N} \sum [ y \log(\hat{y} + \epsilon) + (1-y)\log(1-\hat{y} + \epsilon) ]
    \frac{\partial L}{\partial \hat{y}} = \frac{1}{N} \frac{\hat{y} - y}{\hat{y}(1 - \hat{y}) + \epsilon}
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        N = y_pred.shape[0]
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        return float(loss)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        N = y_pred.shape[0]
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        return ((y_pred - y_true) / (y_pred * (1.0 - y_pred))) / N


class CategoricalCrossEntropyLoss(Loss):
    r"""
    Categorical Cross-Entropy Loss (for Multi-Class Classification):
    L = - \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{i,k} \log(\hat{y}_{i,k} + \epsilon)

    Gradient w.r.t probabilities:
    \frac{\partial L}{\partial \hat{y}} = - \frac{1}{N} \frac{y}{\hat{y} + \epsilon}

    Note: When paired directly with Softmax output z, the combined gradient is:
    \frac{\partial L}{\partial z} = \frac{1}{N} (\hat{y} - y)
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def _format_labels(self, y_true: np.ndarray, num_classes: int) -> np.ndarray:
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            y_indices = y_true.astype(int).ravel()
            y_one_hot = np.zeros((len(y_indices), num_classes))
            y_one_hot[np.arange(len(y_indices)), y_indices] = 1.0
            return y_one_hot
        return y_true

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        N = y_pred.shape[0]
        K = y_pred.shape[1]
        y_true_one_hot = self._format_labels(y_true, K)
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        loss = -np.sum(y_true_one_hot * np.log(y_pred)) / N
        return float(loss)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        N = y_pred.shape[0]
        K = y_pred.shape[1]
        y_true_one_hot = self._format_labels(y_true, K)
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        return (-y_true_one_hot / y_pred) / N
