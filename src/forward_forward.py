"""
Geoffrey Hinton's Forward-Forward (FF) Algorithm Implementation from Scratch.
Reference: "The Forward-Forward Algorithm: Some Preliminary Investigations" (Geoffrey Hinton, Dec 2022).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class FFLayer:
    r"""
    A single layer trained via Hinton's Forward-Forward Algorithm.

    Key Concepts:
    - Forward pass produces non-linear activations: h = ReLU(X * W + b)
    - Goodness metric: G(h) = \sum_j h_{i,j}^2 (sum of squared neural activities)
    - Contrastive Loss (Local):
        L = \ln(1 + \exp(\theta - G_{pos})) + \ln(1 + \exp(G_{neg} - \theta))
        where \theta is the goodness threshold.
    - Normalized output passed to next layer:
        h_{norm} = \frac{h}{\|h\|_2 + \epsilon}
        (Strips magnitude so downstream layers must extract new feature correlations)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        threshold: float = 2.0,
        eps: float = 1e-6,
        seed: Optional[int] = None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.eps = eps

        rng = np.random.default_rng(seed)
        # He/Kaiming initialization
        std = np.sqrt(2.0 / in_features)
        self.W = rng.normal(0.0, std, (in_features, out_features))
        self.b = np.zeros((1, out_features))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Computes un-normalized ReLU activations: h = max(0, X*W + b)."""
        z = np.dot(x, self.W) + self.b
        return np.maximum(0.0, z)

    def normalize(self, h: np.ndarray) -> np.ndarray:
        """Computes L2 row-normalized activations for downstream layers."""
        norms = np.linalg.norm(h, axis=1, keepdims=True) + self.eps
        return h / norms

    def forward_normalized(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns both un-normalized activations (for goodness) and normalized activations (for next layer)."""
        h = self.forward(x)
        h_norm = self.normalize(h)
        return h, h_norm

    def goodness(self, h: np.ndarray) -> np.ndarray:
        r"""Calculates goodness as the sum of squared activations for each sample: G_i = \sum_j h_{i,j}^2."""
        return np.sum(h**2, axis=1, keepdims=True)

    def compute_loss(
        self, h_pos: np.ndarray, h_neg: np.ndarray
    ) -> Tuple[float, float, float]:
        r"""
        Computes the layer's local contrastive loss.
        Positive loss: \ln(1 + \exp(\theta - G_{pos}))  -> pushes G_{pos} > \theta
        Negative loss: \ln(1 + \exp(G_{neg} - \theta))  -> pushes G_{neg} < \theta
        """
        g_pos = self.goodness(h_pos)
        g_neg = self.goodness(h_neg)

        # Log-Sum-Exp / Softplus formulation for numerical stability:
        # ln(1 + e^z) = max(0, z) + ln(1 + e^{-|z|})
        diff_pos = self.threshold - g_pos
        loss_pos = np.mean(np.log1p(np.exp(-np.abs(diff_pos))) + np.maximum(0.0, diff_pos))

        diff_neg = g_neg - self.threshold
        loss_neg = np.mean(np.log1p(np.exp(-np.abs(diff_neg))) + np.maximum(0.0, diff_neg))

        total_loss = float(loss_pos + loss_neg)
        return total_loss, float(np.mean(g_pos)), float(np.mean(g_neg))

    def train_step(
        self,
        x_pos: np.ndarray,
        x_neg: np.ndarray,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
    ) -> Tuple[float, float, float]:
        r"""
        Executes a local gradient descent update without backpropagating across layers.

        Derivation:
        Let L = \ln(1 + e^{\theta - G_{pos}}) + \ln(1 + e^{G_{neg} - \theta})
        \frac{\partial L}{\partial G_{pos}} = - \sigma(\theta - G_{pos}) = - (1 - \sigma(G_{pos} - \theta))
        \frac{\partial L}{\partial G_{neg}} = \sigma(G_{neg} - \theta)

        Since G = \sum_j h_j^2, \frac{\partial G}{\partial h_j} = 2 h_j
        \frac{\partial h_j}{\partial z_j} = \mathbb{I}(z_j > 0)
        \frac{\partial z}{\partial W} = X^T

        Thus:
        dL/dz_pos = - \sigma(\theta - G_{pos}) \odot 2 h_{pos} \odot \mathbb{I}(z_{pos} > 0)
        dL/dz_neg =   \sigma(G_{neg} - \theta) \odot 2 h_{neg} \odot \mathbb{I}(z_{neg} > 0)
        """
        N_pos = x_pos.shape[0]
        N_neg = x_neg.shape[0]

        # 1. Forward passes for positive and negative data
        z_pos = np.dot(x_pos, self.W) + self.b
        h_pos = np.maximum(0.0, z_pos)
        g_pos = self.goodness(h_pos)

        z_neg = np.dot(x_neg, self.W) + self.b
        h_neg = np.maximum(0.0, z_neg)
        g_neg = self.goodness(h_neg)

        # 2. Compute probabilities / sigmoid weights
        # p_pos = \sigma(G_{pos} - \theta) \in (0, 1) -> 1 - p_pos is the error factor
        p_pos = 1.0 / (1.0 + np.exp(-np.clip(g_pos - self.threshold, -50.0, 50.0)))
        # p_neg = \sigma(G_{neg} - \theta) \in (0, 1) -> p_neg is the error factor
        p_neg = 1.0 / (1.0 + np.exp(-np.clip(g_neg - self.threshold, -50.0, 50.0)))

        # 3. Compute local derivatives w.r.t pre-activations Z
        # Positive gradients: desire to increase goodness
        dz_pos = -2.0 * (1.0 - p_pos) * h_pos * (z_pos > 0.0) / N_pos
        # Negative gradients: desire to decrease goodness
        dz_neg =  2.0 * p_neg * h_neg * (z_neg > 0.0) / N_neg

        # 4. Accumulate parameter gradients
        dW = np.dot(x_pos.T, dz_pos) + np.dot(x_neg.T, dz_neg)
        db = np.sum(dz_pos, axis=0, keepdims=True) + np.sum(dz_neg, axis=0, keepdims=True)

        if weight_decay > 0.0:
            dW += weight_decay * self.W

        # 5. Local parameter update (No backpropagation through earlier layers!)
        self.W -= lr * dW
        self.b -= lr * db

        loss, mean_g_pos, mean_g_neg = self.compute_loss(h_pos, h_neg)
        return loss, mean_g_pos, mean_g_neg


class FFNetwork:
    """
    Forward-Forward Multi-Layer Neural Network Container.
    Implements supervised classification via label overlaying and goodness accumulation.
    """

    def __init__(self, layers: Optional[List[FFLayer]] = None):
        self.layers: List[FFLayer] = layers if layers is not None else []

    def add(self, layer: FFLayer) -> "FFNetwork":
        self.layers.append(layer)
        return self

    @staticmethod
    def overlay_label(x: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Overlays the class label onto the first `num_classes` features of the input data.
        - The target class dimension is set to max value (e.g., 10.0 or max(x)).
        - Other class indicator dimensions are set to 0.0.
        """
        x_mod = x.copy()
        max_val = max(1.0, float(np.max(x)))
        # Zero out the label indicator region
        x_mod[:, :num_classes] = 0.0

        if y.ndim == 1:
            for i, label in enumerate(y):
                x_mod[i, int(label)] = max_val
        elif y.ndim == 2 and y.shape[1] == 1:
            for i, label in enumerate(y.ravel()):
                x_mod[i, int(label)] = max_val
        else:
            # One-hot encoded input
            x_mod[:, :num_classes] = y * max_val

        return x_mod

    def generate_positive_and_negative_pairs(
        self, x: np.ndarray, y: np.ndarray, num_classes: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates:
        - Positive samples: (x, correct_label)
        - Negative samples: (x, incorrect_random_label)
        """
        n_samples = x.shape[0]
        y_int = y.ravel().astype(int) if (y.ndim == 1 or y.shape[1] == 1) else np.argmax(y, axis=1)

        # Generate false labels (randomly chosen from wrong classes)
        false_labels = (y_int + np.random.randint(1, num_classes, size=n_samples)) % num_classes

        x_pos = self.overlay_label(x, y_int, num_classes)
        x_neg = self.overlay_label(x, false_labels, num_classes)
        return x_pos, x_neg

    def train_layer_by_layer(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        num_classes: int,
        epochs_per_layer: int = 100,
        batch_size: int = 64,
        lr: float = 0.03,
        threshold: float = 2.0,
        verbose: bool = True,
    ) -> Dict[str, List]:
        """
        Trains each layer greedily and independently using positive and negative passes.
        Once layer l is trained, its normalized activations h_norm become the inputs for layer l+1.
        """
        history: Dict[str, List] = {"layer_losses": [], "pos_goodness": [], "neg_goodness": []}
        n_samples = x_train.shape[0]

        # Generate initial positive and negative datasets
        x_pos, x_neg = self.generate_positive_and_negative_pairs(x_train, y_train, num_classes)

        h_pos_curr = x_pos
        h_neg_curr = x_neg

        for l_idx, layer in enumerate(self.layers):
            layer.threshold = threshold
            layer_loss_hist = []

            if verbose:
                print(f"\n--- Training Layer {l_idx + 1}/{len(self.layers)} ({layer.in_features} -> {layer.out_features}) ---")

            for epoch in range(1, epochs_per_layer + 1):
                indices = np.random.permutation(n_samples)
                h_pos_shuffled = h_pos_curr[indices]
                h_neg_shuffled = h_neg_curr[indices]

                batch_losses = []
                batch_g_pos = []
                batch_g_neg = []

                for start_idx in range(0, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)
                    xb_pos = h_pos_shuffled[start_idx:end_idx]
                    xb_neg = h_neg_shuffled[start_idx:end_idx]

                    loss, g_pos, g_neg = layer.train_step(xb_pos, xb_neg, lr=lr)
                    batch_losses.append(loss)
                    batch_g_pos.append(g_pos)
                    batch_g_neg.append(g_neg)

                avg_loss = float(np.mean(batch_losses))
                avg_g_pos = float(np.mean(batch_g_pos))
                avg_g_neg = float(np.mean(batch_g_neg))
                layer_loss_hist.append(avg_loss)

                if verbose and (epoch % max(1, epochs_per_layer // 4) == 0 or epoch == 1 or epoch == epochs_per_layer):
                    print(
                        f"Layer {l_idx + 1} | Epoch {epoch:3d}/{epochs_per_layer:3d} | "
                        f"Loss: {avg_loss:.4f} | Pos Goodness: {avg_g_pos:.2f} | Neg Goodness: {avg_g_neg:.2f}"
                    )

            history["layer_losses"].append(layer_loss_hist)

            # Compute normalized activations for the NEXT layer
            _, h_pos_curr = layer.forward_normalized(h_pos_curr)
            _, h_neg_curr = layer.forward_normalized(h_neg_curr)

        return history

    def predict(self, x: np.ndarray, num_classes: int) -> np.ndarray:
        r"""
        Inference via Goodness Accumulation:
        For each candidate class label c \in [0, K-1]:
        1. Overlays label c on the input x.
        2. Passes x_c through the network.
        3. Accumulates total goodness G_total(c) across all hidden layers.
        4. Predicts class with highest total goodness: \hat{y} = \arg\max_c G_{total}(c).
        """
        n_samples = x.shape[0]
        accumulated_goodness = np.zeros((n_samples, num_classes))

        for c in range(num_classes):
            # Overlay candidate class label c
            x_candidate = self.overlay_label(x, np.full(n_samples, c), num_classes)

            h_curr = x_candidate
            for layer in self.layers:
                h_raw, h_norm = layer.forward_normalized(h_curr)
                # Accumulate goodness for candidate c
                accumulated_goodness[:, c] += layer.goodness(h_raw).ravel()
                h_curr = h_norm

        return np.argmax(accumulated_goodness, axis=1)

    def evaluate(self, x: np.ndarray, y: np.ndarray, num_classes: int) -> float:
        """Computes classification accuracy."""
        preds = self.predict(x, num_classes)
        y_int = y.ravel().astype(int) if (y.ndim == 1 or y.shape[1] == 1) else np.argmax(y, axis=1)
        acc = float(np.mean(preds == y_int))
        return acc
