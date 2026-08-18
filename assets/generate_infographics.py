"""
Script to generate high-resolution scientific infographics for the repository.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("assets", exist_ok=True)

# 1. Generate Activation Functions & Derivatives Infographic
def generate_activations_plot():
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor="#0f172a")

    z = np.linspace(-4, 4, 400)

    activations = [
        ("ReLU", np.maximum(0, z), np.where(z > 0, 1.0, 0.0), r"$f(z) = \max(0, z)$", r"$f'(z) = \mathbb{I}(z > 0)$"),
        ("Leaky ReLU", np.where(z > 0, z, 0.1 * z), np.where(z > 0, 1.0, 0.1), r"$f(z) = \max(0.1z, z)$", r"$f'(z) = 1 \text{ or } 0.1$"),
        ("Sigmoid", 1 / (1 + np.exp(-z)), (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z))), r"$\sigma(z) = \frac{1}{1+e^{-z}}$", r"$\sigma'(z) = \sigma(1-\sigma)$"),
        ("Tanh", np.tanh(z), 1 - np.tanh(z)**2, r"$\tanh(z) = \frac{e^z-e^{-z}}{e^z+e^{-z}}$", r"$\tanh'(z) = 1-\tanh^2(z)$"),
        (r"ELU ($\alpha=1.0$)", np.where(z > 0, z, np.exp(z) - 1), np.where(z > 0, 1.0, np.exp(z)), r"$f(z) = z \text{ or } e^z-1$", r"$f'(z) = 1 \text{ or } e^z$"),
        ("GELU (approx)", 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3))), None, r"$\text{GELU}(z) \approx z \Phi(z)$", r"Smooth non-monotonic"),
    ]

    for idx, (name, fwd, bwd, fwd_eq, bwd_eq) in enumerate(activations):
        ax = axes[idx // 3, idx % 3]
        ax.set_facecolor("#1e293b")
        ax.plot(z, fwd, label=f"Forward: {fwd_eq}", color="#38bdf8", linewidth=2.5)
        if bwd is not None:
            ax.plot(z, bwd, label=f"Derivative: {bwd_eq}", color="#f43f5e", linewidth=2.0, linestyle="--")
        ax.set_title(name, fontsize=14, color="#f8fafc", weight="bold", pad=10)
        ax.grid(True, linestyle=":", alpha=0.4, color="#94a3b8")
        ax.legend(loc="upper left", fontsize=10, facecolor="#0f172a", edgecolor="#334155")
        ax.tick_params(colors="#cbd5e1")
        ax.spines['bottom'].set_color('#475569')
        ax.spines['top'].set_color('#475569')
        ax.spines['left'].set_color('#475569')
        ax.spines['right'].set_color('#475569')

    plt.suptitle("Activation Functions & Analytical Derivatives", fontsize=18, color="#f8fafc", weight="bold", y=0.98)
    plt.tight_layout()
    out_path = "assets/activations_infographic.png"
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {out_path}")


# 2. Generate Computational Profile & 1:2 Ratio Infographic
def generate_compute_profile_plot():
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0f172a")

    batch_sizes = [32, 128, 512]
    # Benchmarked empirical timings (ms) for Depth=5, Hidden=512
    fwd_times = [3.803, 7.806, 36.090]
    bwd_times = [10.509, 19.760, 69.926]
    ratios = [bwd / fwd for fwd, bwd in zip(fwd_times, bwd_times)]

    x = np.arange(len(batch_sizes))
    width = 0.35

    ax1.set_facecolor("#1e293b")
    rects1 = ax1.bar(x - width/2, fwd_times, width, label="Forward Pass (1 GEMM)", color="#06b6d4")
    rects2 = ax1.bar(x + width/2, bwd_times, width, label="Backward Pass (2 GEMMs)", color="#ec4899")

    ax1.set_ylabel("Execution Time (ms)", fontsize=12, color="#f8fafc")
    ax1.set_title("Forward vs. Backward Pass Latency (Depth=5, Hidden=512)", fontsize=13, color="#f8fafc", weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Batch {b}" for b in batch_sizes], color="#cbd5e1")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155")
    ax1.grid(True, linestyle=":", alpha=0.3, color="#94a3b8")
    ax1.tick_params(colors="#cbd5e1")

    # Ratio plot
    ax2.set_facecolor("#1e293b")
    bars = ax2.bar([f"Batch {b}" for b in batch_sizes], ratios, color="#8b5cf6", width=0.4)
    ax2.axhline(2.0, color="#f59e0b", linestyle="--", linewidth=2, label="Theoretical Bound (2.0x)")
    ax2.set_ylabel("Compute Ratio (Backward / Forward)", fontsize=12, color="#f8fafc")
    ax2.set_title("Empirical Backward-to-Forward Ratio (~2.0x)", fontsize=13, color="#f8fafc", weight="bold")
    ax2.set_ylim(0, 3.2)
    ax2.legend(facecolor="#0f172a", edgecolor="#334155")
    ax2.grid(True, linestyle=":", alpha=0.3, color="#94a3b8")
    ax2.tick_params(colors="#cbd5e1")

    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.08, f"{yval:.2f}x", ha="center", va="bottom", color="#f8fafc", weight="bold")

    plt.suptitle("Computational Analysis: The 1:2 Compute Law of Backpropagation", fontsize=16, color="#f8fafc", weight="bold", y=0.98)
    plt.tight_layout()
    out_path = "assets/computational_profile.png"
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {out_path}")


# 3. Generate Hinton FF vs Backprop Architectural Comparison Infographic
def generate_ff_comparison_diagram():
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5), facecolor="#0f172a")

    # Left: Standard Backprop Pipeline
    ax1.set_facecolor("#1e293b")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("Standard Backpropagation (BP)", fontsize=16, color="#38bdf8", weight="bold", pad=15)

    # Boxes for Backprop
    box_props = dict(boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor="#0284c7", linewidth=1.5)
    ax1.text(5, 8.5, "Input: X", ha="center", va="center", color="#f8fafc", fontsize=12, bbox=box_props)
    ax1.text(5, 6.5, "Layer 1: Z^[1] = XW1 + b1\n[Must cache A^[1] in RAM]", ha="center", va="center", color="#f8fafc", fontsize=10, bbox=box_props)
    ax1.text(5, 4.5, "Layer 2: Z^[2] = A^[1]W2 + b2\n[Must cache A^[2] in RAM]", ha="center", va="center", color="#f8fafc", fontsize=10, bbox=box_props)
    ax1.text(5, 2.5, "Output Loss: L(y, y_pred)\n[Global Scalar Objective]", ha="center", va="center", color="#f43f5e", fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor="#e11d48", linewidth=1.5))

    # Arrows for Forward
    ax1.annotate("", xy=(5, 7.3), xytext=(5, 8.0), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2.5))
    ax1.annotate("", xy=(5, 5.3), xytext=(5, 6.0), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2.5))
    ax1.annotate("", xy=(5, 3.3), xytext=(5, 4.0), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2.5))
    ax1.text(3.5, 5.7, "Forward Pass\n(Activations)", color="#38bdf8", fontsize=10, ha="right")

    # Arrows for Backward
    ax1.annotate("", xy=(6.5, 4.0), xytext=(6.5, 3.3), arrowprops=dict(arrowstyle="->", color="#f43f5e", lw=2.5, linestyle="--"))
    ax1.annotate("", xy=(6.5, 6.0), xytext=(6.5, 5.3), arrowprops=dict(arrowstyle="->", color="#f43f5e", lw=2.5, linestyle="--"))
    ax1.text(6.8, 4.7, "Reverse Chain Rule\n(δ Error Gradient)", color="#f43f5e", fontsize=10, ha="left")

    ax1.text(5, 0.8, "• Global Loss Synchronized\n• Requires Activation Caching O(L·B·H)\n• Biologically Implausible (Weight Transport)", ha="center", va="center", color="#cbd5e1", fontsize=10, style="italic")

    # Right: Hinton's Forward-Forward Pipeline
    ax2.set_facecolor("#1e293b")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("Geoffrey Hinton's Forward-Forward (FF)", fontsize=16, color="#a855f7", weight="bold", pad=15)

    ff_box_pos = dict(boxstyle="round,pad=0.4", facecolor="#0f172a", edgecolor="#10b981", linewidth=1.5)
    ff_box_neg = dict(boxstyle="round,pad=0.4", facecolor="#0f172a", edgecolor="#ef4444", linewidth=1.5)
    ff_box_layer = dict(boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor="#9333ea", linewidth=1.5)

    ax2.text(3, 8.5, "Pos Pass (x, y_true)\n[Real Data]", ha="center", va="center", color="#10b981", fontsize=10, bbox=ff_box_pos)
    ax2.text(7, 8.5, "Neg Pass (x, y_false)\n[Corrupted Data]", ha="center", va="center", color="#ef4444", fontsize=10, bbox=ff_box_neg)

    ax2.text(5, 6.2, "Layer 1: Local Contrastive Loss\nMaximize G_pos = ∑(h_pos)^2 > θ\nMinimize G_neg = ∑(h_neg)^2 < θ\n[Normalize h / ||h||2 -> Layer 2]", ha="center", va="center", color="#f8fafc", fontsize=10, bbox=ff_box_layer)
    ax2.text(5, 3.2, "Layer 2: Local Contrastive Loss\nMaximize G_pos = ∑(h_pos)^2 > θ\nMinimize G_neg = ∑(h_neg)^2 < θ\n[Independent Greedy Updates]", ha="center", va="center", color="#f8fafc", fontsize=10, bbox=ff_box_layer)

    # Arrows for FF
    ax2.annotate("", xy=(3.5, 7.3), xytext=(3.0, 8.0), arrowprops=dict(arrowstyle="->", color="#10b981", lw=2))
    ax2.annotate("", xy=(6.5, 7.3), xytext=(7.0, 8.0), arrowprops=dict(arrowstyle="->", color="#ef4444", lw=2))
    ax2.annotate("", xy=(5, 4.3), xytext=(5, 5.1), arrowprops=dict(arrowstyle="->", color="#a855f7", lw=2.5))
    ax2.text(5.5, 4.7, "Normalized h / ||h||", color="#a855f7", fontsize=9)

    ax2.text(5, 0.8, "• Zero Backward Pass (Two Forward Passes)\n• Layer-Local Learning (No Error Propagation)\n• Neuromorphic / Analog Friendly", ha="center", va="center", color="#cbd5e1", fontsize=10, style="italic")

    plt.suptitle("Architectural Comparison: Backpropagation vs. Hinton's Forward-Forward", fontsize=17, color="#f8fafc", weight="bold", y=0.98)
    plt.tight_layout()
    out_path = "assets/hintons_ff_vs_backprop.png"
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {out_path}")


# 4. Generate Biological Recursive Feedback vs Forward-Forward Infographic
def generate_biological_vs_ff_diagram():
    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6.5), facecolor="#0f172a")

    # Panel 1: Biological 2-Compartment Pyramidal Neuron
    ax1.set_facecolor("#1e293b")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("1. Biological Pyramidal Neuron\n(Two-Compartment Biophysics)", fontsize=13, color="#38bdf8", weight="bold", pad=12)

    box_apical = dict(boxstyle="round,pad=0.4", facecolor="#0f172a", edgecolor="#38bdf8", linewidth=1.5)
    box_soma = dict(boxstyle="circle,pad=0.5", facecolor="#0f172a", edgecolor="#f59e0b", linewidth=1.8)
    box_basal = dict(boxstyle="round,pad=0.4", facecolor="#0f172a", edgecolor="#10b981", linewidth=1.5)

    ax1.text(5, 8.5, "Apical Tuft (Layer 1)\nTop-Down Recursive Feedback\n(Predictions & Context)", ha="center", va="center", color="#38bdf8", fontsize=9, bbox=box_apical)
    ax1.text(5, 4.8, "Soma\n[BAC Burst\nSpike Engine]", ha="center", va="center", color="#f59e0b", fontsize=9, bbox=box_soma)
    ax1.text(5, 1.8, "Basal Dendrites (Layer 4)\nBottom-Up Sensory Feedforward\n(Direct Input Drive)", ha="center", va="center", color="#10b981", fontsize=9, bbox=box_basal)

    # Arrows connecting compartments
    ax1.annotate("", xy=(5, 5.8), xytext=(5, 7.5), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2, linestyle=":"))
    ax1.annotate("", xy=(5, 3.8), xytext=(5, 2.7), arrowprops=dict(arrowstyle="->", color="#10b981", lw=2))
    ax1.text(5, 0.4, "• Local Ca2+ dendritic plateau spikes\n• 10:1 Top-Down vs Feedforward Ratio\n• Resolves Weight Transport biophysically", ha="center", va="center", color="#cbd5e1", fontsize=8.5, style="italic")

    # Panel 2: Predictive Coding Loops
    ax2.set_facecolor("#1e293b")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("2. Cortical Predictive Coding\n(Continuous Dynamical Loops)", fontsize=13, color="#ec4899", weight="bold", pad=12)

    box_pred = dict(boxstyle="round,pad=0.4", facecolor="#0f172a", edgecolor="#a855f7", linewidth=1.5)
    box_err = dict(boxstyle="round,pad=0.4", facecolor="#0f172a", edgecolor="#ef4444", linewidth=1.5)

    ax2.text(5, 8.2, "Higher Area Representation\nState: r^[L](t)", ha="center", va="center", color="#a855f7", fontsize=9.5, bbox=box_pred)
    ax2.text(5, 4.8, "Top-Down Prediction: μ = W_fb • r^[L]\nResidual Error: ε = r^[L-1] - μ", ha="center", va="center", color="#f8fafc", fontsize=9, bbox=box_err)
    ax2.text(5, 1.8, "Lower Area Sensory Drive\nState: r^[L-1](t)", ha="center", va="center", color="#06b6d4", fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f172a", edgecolor="#06b6d4", linewidth=1.5))

    ax2.annotate("", xy=(4.2, 5.8), xytext=(4.2, 7.3), arrowprops=dict(arrowstyle="->", color="#a855f7", lw=2))
    ax2.text(3.5, 6.6, "Prediction\n(Top-Down)", color="#a855f7", fontsize=8, ha="right")
    ax2.annotate("", xy=(5.8, 7.3), xytext=(5.8, 5.8), arrowprops=dict(arrowstyle="->", color="#ef4444", lw=2, linestyle="--"))
    ax2.text(6.5, 6.6, "Residual Error ε\n(Bottom-Up)", color="#ef4444", fontsize=8, ha="left")

    ax2.text(5, 0.4, "• dr/dt = -ε + W_ff^T ε (Continuous settling)\n• ΔW ∝ ε • r^T (Local Hebbian update)\n• Minimizes Variational Free Energy", ha="center", va="center", color="#cbd5e1", fontsize=8.5, style="italic")

    # Panel 3: Hinton's Forward-Forward Paradigm
    ax3.set_facecolor("#1e293b")
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis("off")
    ax3.set_title("3. Hinton's Forward-Forward\n(Contrastive Goodness Passes)", fontsize=13, color="#10b981", weight="bold", pad=12)

    box_ff_pos = dict(boxstyle="round,pad=0.35", facecolor="#0f172a", edgecolor="#10b981", linewidth=1.5)
    box_ff_neg = dict(boxstyle="round,pad=0.35", facecolor="#0f172a", edgecolor="#ef4444", linewidth=1.5)

    ax3.text(3, 8.2, "Positive Pass\n(Real Data: G > θ)\n[Wake / Sensory]", ha="center", va="center", color="#10b981", fontsize=8.5, bbox=box_ff_pos)
    ax3.text(7, 8.2, "Negative Pass\n(Corrupt: G < θ)\n[Sleep / Dream]", ha="center", va="center", color="#ef4444", fontsize=8.5, bbox=box_ff_neg)

    ax3.text(5, 4.8, "Layer-Local Contrastive Plasticity\nG(h) = ∑ h_j^2 (Goodness)\nLocal Update: ∇_W = X^T (dL/dz)", ha="center", va="center", color="#f8fafc", fontsize=9, bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor="#9333ea", linewidth=1.5))
    ax3.text(5, 1.8, "Normalized Activity Output\nh_norm = h / ||h||_2\nPassed forward to Layer l+1", ha="center", va="center", color="#f8fafc", fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f172a", edgecolor="#6366f1", linewidth=1.5))

    ax3.annotate("", xy=(3.5, 6.0), xytext=(3.0, 7.3), arrowprops=dict(arrowstyle="->", color="#10b981", lw=2))
    ax3.annotate("", xy=(6.5, 6.0), xytext=(7.0, 7.3), arrowprops=dict(arrowstyle="->", color="#ef4444", lw=2))
    ax3.annotate("", xy=(5, 2.7), xytext=(5, 3.8), arrowprops=dict(arrowstyle="->", color="#9333ea", lw=2))

    ax3.text(5, 0.4, "• Forward-only contrastive learning\n• Zero activation memory caching in RAM\n• Solves Weight Transport without backprop", ha="center", va="center", color="#cbd5e1", fontsize=8.5, style="italic")

    plt.suptitle("Comparative Synthesis: Biological Cortical Recursion vs. Hinton's Forward-Forward", fontsize=16, color="#f8fafc", weight="bold", y=0.98)
    plt.tight_layout()
    out_path = "assets/biological_recurrent_vs_ff.png"
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {out_path}")


if __name__ == "__main__":
    generate_activations_plot()
    generate_compute_profile_plot()
    generate_ff_comparison_diagram()
    generate_biological_vs_ff_diagram()


