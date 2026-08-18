"""
Neural Network Sequential Model Container.
Manages forward propagation, backpropagation, and training loops.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from .losses import Loss
from .optimizers import Optimizer


class NeuralNetwork:
    """Sequential Feedforward Neural Network container."""

    def __init__(self, layers: Optional[List] = None):
        self.layers: List = layers if layers is not None else []
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def add(self, layer) -> "NeuralNetwork":
        """Appends a layer or activation function to the architecture."""
        self.layers.append(layer)
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Sequential forward pass across all layers."""
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_grad: np.ndarray) -> np.ndarray:
        """Sequential backward pass across all layers in reverse order."""
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass inference."""
        return self.forward(x)

    def predict_classes(self, x: np.ndarray) -> np.ndarray:
        """Returns the predicted class indices."""
        probs = self.predict(x)
        if probs.ndim == 1 or probs.shape[1] == 1:
            return (probs > 0.5).astype(int)
        return np.argmax(probs, axis=1)

    def evaluate(
        self, x: np.ndarray, y: np.ndarray, loss_fn: Loss
    ) -> Tuple[float, float]:
        """Evaluates the model on test/validation data."""
        y_pred = self.forward(x)
        loss = loss_fn.forward(y_pred, y)

        if y_pred.shape[1] > 1:
            y_pred_cls = np.argmax(y_pred, axis=1)
            y_true_cls = y.ravel() if (y.ndim == 1 or y.shape[1] == 1) else np.argmax(y, axis=1)
            acc = float(np.mean(y_pred_cls == y_true_cls))
        else:
            y_pred_cls = (y_pred > 0.5).astype(int).ravel()
            y_true_cls = y.ravel()
            acc = float(np.mean(y_pred_cls == y_true_cls))

        return loss, acc

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        batch_size: int,
        optimizer: Optimizer,
        loss_fn: Loss,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = True,
        print_every: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Trains the network using mini-batch gradient descent with backpropagation.
        """
        n_samples = x_train.shape[0]
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        for epoch in range(1, epochs + 1):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            epoch_losses = []

            # Mini-batch loop
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                xb = x_shuffled[start_idx:end_idx]
                yb = y_shuffled[start_idx:end_idx]

                # 1. Forward Pass
                y_pred = self.forward(xb)

                # 2. Loss computation
                batch_loss = loss_fn.forward(y_pred, yb)
                epoch_losses.append(batch_loss)

                # 3. Backward Pass (Backpropagation)
                loss_grad = loss_fn.backward(y_pred, yb)
                self.backward(loss_grad)

                # 4. Optimization Step
                optimizer.step(self.layers)

            train_loss = float(np.mean(epoch_losses))
            _, train_acc = self.evaluate(x_train, y_train, loss_fn)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            val_str = ""
            if val_data is not None:
                x_val, y_val = val_data
                val_loss, val_acc = self.evaluate(x_val, y_val, loss_fn)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                val_str = f" | Val Loss: {val_loss:.4f} - Val Acc: {val_acc * 100:.2f}%"

            if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == epochs):
                print(
                    f"Epoch {epoch:4d}/{epochs:4d} | Train Loss: {train_loss:.4f} - Train Acc: {train_acc * 100:.2f}%{val_str}"
                )

        return self.history
