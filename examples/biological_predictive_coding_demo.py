"""
Simulation of Biological Recursive Predictive Coding (Rao & Ballard, Friston)
Demonstrates continuous-time top-down recursive settling vs feedforward passes.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class BiologicalPredictiveCodingLayer:
    r"""
    A 2-Compartment / Predictive Coding Cortical Module.

    Top-Down Prediction: \mu^{[l-1]} = W_{fb} \cdot r^{[l]}
    Bottom-Up Residual Error: \epsilon^{[l-1]} = r^{[l-1]} - \mu^{[l-1]}
    Dynamical State Settling: \frac{dr^{[l]}}{dt} = -\alpha \epsilon^{[l]} + \beta W_{ff}^T \epsilon^{[l-1]}
    Hebbian Plasticity: \Delta W \propto \epsilon^{[l-1]} (r^{[l]})^T
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Forward and feedback synaptic matrices (initially symmetric or random)
        self.W_ff = rng.normal(0.0, 1.0 / np.sqrt(in_dim), (in_dim, out_dim))
        self.W_fb = self.W_ff.T.copy()  # Top-down generative weights

    def settle_dynamics(
        self,
        sensory_input: np.ndarray,
        num_timesteps: int = 40,
        dt: float = 0.05,
        decay: float = 0.1,
    ):
        """
        Simulates continuous-time dynamical relaxation across cortical recursive loops.
        """
        N = sensory_input.shape[0]

        # Initial representation state (r)
        r = np.zeros((N, self.out_dim))

        error_history = []
        state_trajectory = []

        for t in range(num_timesteps):
            # 1. Top-down feedback prediction (r: N x out_dim, W_fb: out_dim x in_dim)
            mu = np.dot(r, self.W_fb)

            # 2. Bottom-up prediction error residual
            error = sensory_input - mu

            # 3. Dynamic continuous-time differential equation update
            # dr/dt = -decay * r + W_ff^T @ error
            dr = -decay * r + np.dot(error, self.W_ff)
            r += dt * dr

            # Apply biological rectification (firing rate cannot be negative)
            r = np.maximum(0.0, r)

            total_error_energy = float(np.mean(error**2))
            error_history.append(total_error_energy)
            state_trajectory.append(r[0].copy())

        return r, error_history, np.array(state_trajectory)


def main():
    print("=" * 70)
    print("BIOLOGICAL NEURAL SIMULATION: Recursive Predictive Coding Settling")
    print("=" * 70)

    # 1. Generate sensory signal
    np.random.seed(42)
    in_dim = 16
    out_dim = 8
    sensory_data = np.random.randn(10, in_dim)

    pc_module = BiologicalPredictiveCodingLayer(in_dim=in_dim, out_dim=out_dim)

    print(f"Running continuous-time attractor dynamics across {in_dim} -> {out_dim} cortical columns...")
    r_settled, error_hist, traj = pc_module.settle_dynamics(sensory_data, num_timesteps=60, dt=0.08)

    print(f"Initial Prediction Error Energy: {error_hist[0]:.4f}")
    print(f"Settled Attractor Error Energy:  {error_hist[-1]:.4f} (Reduced by {(1 - error_hist[-1]/error_hist[0])*100:.1f}%)")

    # 2. Plot Dynamic Convergence
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f172a")

    ax1.set_facecolor("#1e293b")
    ax1.plot(error_hist, color="#f43f5e", linewidth=2.5, label="Prediction Error Energy: ||x - W_fb · r||^2")
    ax1.set_xlabel("Continuous Time Steps (dt)", fontsize=11, color="#cbd5e1")
    ax1.set_ylabel("Mean Squared Error Energy", fontsize=11, color="#cbd5e1")
    ax1.set_title("Cortical Error Minimization via Top-Down Feedback", fontsize=13, color="#f8fafc", weight="bold")
    ax1.grid(True, linestyle=":", alpha=0.3, color="#94a3b8")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155")
    ax1.tick_params(colors="#cbd5e1")

    ax2.set_facecolor("#1e293b")
    for dim_idx in range(min(5, out_dim)):
        ax2.plot(traj[:, dim_idx], label=f"Neuron Activity r_{dim_idx}(t)", linewidth=2.0)
    ax2.set_xlabel("Continuous Time Steps (dt)", fontsize=11, color="#cbd5e1")
    ax2.set_ylabel("Firing Rate r(t)", fontsize=11, color="#cbd5e1")
    ax2.set_title("Neural Firing Trajectory Settling into Attractor", fontsize=13, color="#f8fafc", weight="bold")
    ax2.grid(True, linestyle=":", alpha=0.3, color="#94a3b8")
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", loc="upper left", fontsize=9)
    ax2.tick_params(colors="#cbd5e1")

    plt.suptitle("Biological Cortical Recursion: Dynamic Attractor Settling", fontsize=16, color="#f8fafc", weight="bold", y=0.98)
    plt.tight_layout()
    out_path = "assets/biological_pc_simulation.png"
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Saved biological predictive coding simulation to: {out_path}")


if __name__ == "__main__":
    main()
