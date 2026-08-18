"""
Gradient Checking Utility.
Verifies analytical backpropagation gradients against two-sided finite difference approximations.
"""

from typing import Dict, Tuple
import numpy as np
from .losses import Loss


def gradient_check(
    model,
    x: np.ndarray,
    y: np.ndarray,
    loss_fn: Loss,
    epsilon: float = 1e-6,
    tolerance: float = 1e-6,
    verbose: bool = True,
) -> Tuple[bool, Dict[str, float]]:
    """
    Performs finite difference gradient checking on all trainable parameters in the model.

    Computes:
        g_numerical = (J(theta + eps) - J(theta - eps)) / (2 * eps)
        relative_error = ||g_analytical - g_numerical||_2 / (||g_analytical||_2 + ||g_numerical||_2 + 1e-15)

    Returns:
        (passed, error_dict)
    """
    # 1. Compute analytical gradients via forward + backward pass
    y_pred = model.forward(x)
    loss_grad = loss_fn.backward(y_pred, y)
    model.backward(loss_grad)

    errors = {}
    passed = True

    for l_idx, layer in enumerate(model.layers):
        if not hasattr(layer, "params") or not layer.params:
            continue

        for param_name in layer.params:
            param = layer.params[param_name]
            analytical_grad = layer.grads[param_name].copy()
            numerical_grad = np.zeros_like(param)

            it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
            while not it.finished:
                idx = it.multi_index
                orig_val = param[idx]

                # theta + epsilon
                param[idx] = orig_val + epsilon
                loss_plus = loss_fn.forward(model.forward(x), y)

                # theta - epsilon
                param[idx] = orig_val - epsilon
                loss_minus = loss_fn.forward(model.forward(x), y)

                # Restore original parameter
                param[idx] = orig_val

                # Central difference formula
                numerical_grad[idx] = (loss_plus - loss_minus) / (2.0 * epsilon)
                it.iternext()

            # Compute relative error
            numerator = np.linalg.norm(analytical_grad - numerical_grad)
            denominator = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad) + 1e-15
            rel_error = float(numerator / denominator)

            key = f"Layer_{l_idx}_{layer.__class__.__name__}_{param_name}"
            errors[key] = rel_error

            if rel_error > tolerance:
                passed = False
                if verbose:
                    print(
                        f"[FAILED] {key}: Rel Error = {rel_error:.2e} > Tolerance = {tolerance:.2e}"
                    )
            else:
                if verbose:
                    print(
                        f"[PASSED] {key}: Rel Error = {rel_error:.2e} <= Tolerance = {tolerance:.2e}"
                    )

    return passed, errors
