"""
Computational Profiler and Benchmark for Feedforward Neural Network.
Measures Forward Pass vs. Backward Pass latency, FLOPs, and activation memory scaling.
"""

import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.network import NeuralNetwork
from src.layers import Dense
from src.activations import ReLU
from src.losses import MSELoss


def run_benchmark():
    print("=" * 70)
    print("COMPUTATIONAL BENCHMARK: Forward vs Backward Pass Analysis")
    print("=" * 70)

    batch_sizes = [32, 128, 512]
    hidden_dims = [256, 512, 1024]
    depths = [2, 5, 10]

    loss_fn = MSELoss()

    print(f"{'Batch':<8}{'Depth':<8}{'Hidden':<10}{'Fwd (ms)':<12}{'Bwd (ms)':<12}{'Bwd/Fwd Ratio':<15}{'Est. GFLOPs':<12}")
    print("-" * 75)

    for depth in depths:
        for hidden in hidden_dims:
            for bsz in batch_sizes:
                layers = [Dense(hidden, hidden, init_method="he")]
                for _ in range(depth - 1):
                    layers.append(ReLU())
                    layers.append(Dense(hidden, hidden, init_method="he"))

                net = NeuralNetwork(layers)
                x = np.random.randn(bsz, hidden)
                y = np.random.randn(bsz, hidden)

                # Warmup
                for _ in range(5):
                    pred = net.forward(x)
                    grad = loss_fn.backward(pred, y)
                    net.backward(grad)

                # Measure Forward Pass
                n_iters = 50
                t0 = time.perf_counter()
                for _ in range(n_iters):
                    pred = net.forward(x)
                fwd_time = (time.perf_counter() - t0) / n_iters * 1000.0

                # Measure Backward Pass
                grad = loss_fn.backward(pred, y)
                t0 = time.perf_counter()
                for _ in range(n_iters):
                    net.backward(grad)
                bwd_time = (time.perf_counter() - t0) / n_iters * 1000.0

                ratio = bwd_time / fwd_time if fwd_time > 0 else 0.0

                # FLOPs per pass:
                # Forward: depth * (2 * bsz * hidden * hidden)
                # Backward: depth * (4 * bsz * hidden * hidden)
                total_flops = depth * (6 * bsz * hidden * hidden)
                total_gflops = (total_flops / ((fwd_time + bwd_time) / 1000.0)) / 1e9

                print(f"{bsz:<8}{depth:<8}{hidden:<10}{fwd_time:<12.3f}{bwd_time:<12.3f}{ratio:<15.2f}{total_gflops:<12.2f}")

    print("=" * 75)
    print("Observation: The theoretical compute ratio of Backward to Forward pass is ~2.0x")
    print("because backward propagation requires two matrix multiplications per layer:")
    print("1) dW = X^T @ dZ (gradient w.r.t weights)")
    print("2) dX = dZ @ W^T (gradient w.r.t inputs for previous layer)")
    print("whereas forward propagation requires only one (Z = X @ W).")
    print("=" * 75)


if __name__ == "__main__":
    run_benchmark()
