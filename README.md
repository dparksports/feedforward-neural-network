<div align="center">

# Feedforward Neural Network (FFNN) from Scratch in Python

[![Python 3.12](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pytest](https://img.shields.io/badge/Tests-7%20Passed-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br/>

<img src="assets/ffnn_architecture_infographic.jpg" alt="FFNN Architecture & Backpropagation Infographic" width="100%" style="border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.5);"/>

<p align="center">
  <b>A modular, high-performance, from-scratch implementation of Feedforward Neural Networks (Multi-Layer Perceptrons) in Python.</b><br/>
  Featuring complete analytical backpropagation, numerical gradient checking, modern optimizers (Adam, RMSprop, Momentum), and in-depth computational complexity analysis.
</p>

[📚 Full Theoretical & Mathematical Guide](FEEDFORWARD_NEURAL_NETWORK.md) • [🚀 Quickstart](#-quickstart) • [📐 Architecture & Math](#-architecture--mathematical-mechanics) • [⚡ Computational Analysis](#-computational-analysis--the-12-compute-law) • [🧪 Benchmarks](#-empirical-benchmarks)

</div>

---

## 🌟 Highlights

- **Pure NumPy Core**: Transparent matrix calculus and tensor operations without heavy black-box autograd frameworks.
- **Analytical Backpropagation Engine**: Exact multivariate chain-rule gradient derivations verified via two-sided finite-difference gradient checking ($\text{Rel Error} < 10^{-5}$).
- **Production-Ready Modularity**: Decoupled `Dense` layers, activations (`ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax`), and stabilized losses (`MSE`, `BCE`, `Categorical Cross-Entropy`).
- **First-Class Optimizers**: Built-in implementations of `SGD` (with Nesterov/Momentum), `RMSprop`, and `Adam` with bias corrections.
- **Isolated Virtual Environment**: One-command reproducible setup via `setup_env.sh` and Python `venv`.

---

## 📐 Architecture & Mathematical Mechanics

### 1. Forward Propagation
Information flows unidirectionally across $L$ parameterized layers without recurrent cycles:

$$\mathbf{Z}^{[l]} = \mathbf{A}^{[l-1]} \mathbf{W}^{[l]} + \mathbf{b}^{[l]}$$
$$\mathbf{A}^{[l]} = g^{[l]}(\mathbf{Z}^{[l]})$$

Where $\mathbf{X} = \mathbf{A}^{[0]} \in \mathbb{R}^{B \times D_{in}}$, $\mathbf{W}^{[l]} \in \mathbb{R}^{D_{in} \times D_{out}}$, and $\mathbf{b}^{[l]} \in \mathbb{R}^{1 \times D_{out}}$.

<div align="center">
  <img src="assets/activations_infographic.png" alt="Activation Functions & Analytical Derivatives" width="95%" style="border-radius: 8px; margin: 15px 0;"/>
</div>

---

### 2. Analytical Backpropagation (Chain Rule)

Backpropagation computes the gradient of scalar loss $\mathcal{L}$ with respect to all parameters in reverse topological order:

1. **Output Sensitivity Adjoint ($\boldsymbol{\delta}^{[L]}$)**:
   $$\boldsymbol{\delta}^{[L]} \triangleq \frac{\partial \mathcal{L}}{\partial \mathbf{Z}^{[L]}} = \frac{1}{B} (\hat{\mathbf{Y}} - \mathbf{Y}) \quad \text{(for Softmax + Categorical Cross-Entropy)}$$

2. **Hidden Error Propagation ($\boldsymbol{\delta}^{[l-1]}$)**:
   $$\boldsymbol{\delta}^{[l-1]} = \left( \boldsymbol{\delta}^{[l]} (\mathbf{W}^{[l]})^T \right) \odot g'^{[l-1]}(\mathbf{Z}^{[l-1]})$$

3. **Parameter Gradient Accumulation**:
   $$\nabla_{\mathbf{W}^{[l]}} \mathcal{L} = (\mathbf{A}^{[l-1]})^T \boldsymbol{\delta}^{[l]} \quad \in \mathbb{R}^{D_{in} \times D_{out}}$$
   $$\nabla_{\mathbf{b}^{[l]}} \mathcal{L} = \sum_{i=1}^B \boldsymbol{\delta}_{i, :}^{[l]} = \mathbf{1}_{1 \times B} \boldsymbol{\delta}^{[l]} \quad \in \mathbb{R}^{1 \times D_{out}}$$

---

## ⚡ Computational Analysis & The 1:2 Compute Law

<div align="center">
  <img src="assets/computational_profile.png" alt="Computational Profile: Forward vs. Backward Latency" width="95%" style="border-radius: 8px; margin: 15px 0;"/>
</div>

### GEMM FLOPs Breakdown per Layer
Every matrix multiplication $(M \times K) \times (K \times N)$ requires $2 \cdot M \cdot N \cdot K$ FLOPs:

| Stage | Operation | Matrix Dimensions | FLOPs ($B \times D_{in} \times D_{out}$) |
| :--- | :--- | :--- | :--- |
| **Forward Pass** | $Z = A W + b$ | $(B \times D_{in}) \times (D_{in} \times D_{out})$ | **$2 B D_{in} D_{out}$** (1 GEMM) |
| **Backward ($\nabla_W$)** | $\nabla_W = A^T \delta$ | $(D_{in} \times B) \times (B \times D_{out})$ | **$2 B D_{in} D_{out}$** (1 GEMM) |
| **Backward ($\nabla_A$)** | $\nabla_A = \delta W^T$ | $(B \times D_{out}) \times (D_{out} \times D_{in})$ | **$2 B D_{in} D_{out}$** (1 GEMM) |
| **Total Backward** | $\nabla_W + \nabla_A$ | Two Matrix Multiplications | **$4 B D_{in} D_{out}$** (2 GEMMs) |
| **Full Training Step** | Forward + Backward | Three Matrix Multiplications | **$6 B D_{in} D_{out}$** (3 GEMMs) |

> 💡 **The 1:2 Computational Law**: The backward pass inherently requires **$2\times$ the FLOPs** of the forward pass because computing parameter gradients ($\nabla_W$) and downstream error flow ($\nabla_A$) requires two distinct matrix multiplications per layer.

### Memory Complexity: Forward-Only vs. Training
- **Inference (Forward-Only)**: Memory complexity is $\mathcal{O}(\max_l d_l)$ (intermediate layer buffers can be reused in-place).
- **Training (Backpropagation)**: Memory complexity is $\mathcal{O}\left(\sum_{l=1}^L B \cdot d_l\right)$ because **all intermediate activations $\mathbf{A}^{[l-1]}$ and pre-activations $\mathbf{Z}^{[l]}$ must be retained in memory** until the backward step for that layer executes.

---

## 🧪 Empirical Benchmarks

### 1. Non-Linear 3-Arm Spiral Classification
Trained a 3-layer network (`2 -> 64 -> 32 -> 3`) using `ReLU`, `Softmax`, and `Adam(lr=0.01)` on a non-linearly separable spiral dataset:
- **Validation Accuracy**: **`98.89%`**
- **Validation Loss**: **`0.0263`**

<div align="center">
  <img src="assets/spiral_decision_boundary.png" alt="Spiral Dataset Decision Boundary and Convergence" width="90%" style="border-radius: 8px; margin: 15px 0;"/>
</div>

---

## 🚀 Quickstart

### 1. Automated Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd feedforward

# Initialize virtual environment and install dependencies
bash setup_env.sh

# Activate virtual environment
source .venv/bin/activate
```

### 2. Run Automated Pytest Suite
```bash
source .venv/bin/activate
python -m pytest -v tests/
```

### 3. Run Demos & Benchmarks
```bash
# Train on the 3-arm spiral dataset
python examples/train_spiral.py

# Run computational FLOPs and latency profiler
python examples/benchmark.py
```

---

## 💻 Minimal Working Example

```python
import numpy as np
from src.network import NeuralNetwork
from src.layers import Dense
from src.activations import ReLU, Sigmoid
from src.losses import BinaryCrossEntropyLoss
from src.optimizers import Adam

# 1. Non-linear XOR Problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 2. Build Model Architecture
model = NeuralNetwork([
    Dense(in_features=2, out_features=8, init_method="he"),
    ReLU(),
    Dense(in_features=8, out_features=1, init_method="xavier"),
    Sigmoid()
])

# 3. Train with Adam Optimizer
optimizer = Adam(lr=0.05)
loss_fn = BinaryCrossEntropyLoss()

model.fit(X, y, epochs=300, batch_size=4, optimizer=optimizer, loss_fn=loss_fn, verbose=True)

# 4. Predict
predictions = model.predict_classes(X)
print("Predictions:\n", predictions)
```

---

## 📁 Repository Structure

```
feedforward/
├── .gitignore                      # Git ignore file
├── requirements.txt                # Python dependencies (NumPy, Matplotlib, Pytest)
├── setup_env.sh                    # Automated venv bootstrap script
├── FEEDFORWARD_NEURAL_NETWORK.md   # Comprehensive theoretical treatise
├── README.md                       # Main visual overview & quickstart
├── assets/                         # Infographics and architecture diagrams
│   ├── ffnn_architecture_infographic.jpg
│   ├── activations_infographic.png
│   ├── computational_profile.png
│   └── spiral_decision_boundary.png
├── src/                            # Modular FFNN Core Package
│   ├── __init__.py
│   ├── activations.py              # ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
│   ├── losses.py                   # MSE, Binary Cross-Entropy, Categorical Cross-Entropy
│   ├── layers.py                   # Dense Layer (He/Xavier init, gradient cache)
│   ├── optimizers.py               # SGD, Momentum, RMSprop, Adam
│   ├── network.py                  # NeuralNetwork sequential container
│   └── grad_check.py               # Finite-difference gradient checker
├── examples/
│   ├── train_spiral.py             # Spiral classification training script
│   └── benchmark.py                # FLOPs and latency profiler
└── tests/
    └── test_ffnn.py                # 7 Pytest unit & integration test cases
```

---

## 📖 Extended Reading
For a rigorous mathematical derivation of multivariate chain rule Jacobians, the Universal Approximation Theorem, and memory-bandwidth scaling laws, read:
👉 **[`FEEDFORWARD_NEURAL_NETWORK.md`](FEEDFORWARD_NEURAL_NETWORK.md)**

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
