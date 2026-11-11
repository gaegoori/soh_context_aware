"""OOF-error regression selector. Inference never accepts SOH targets.

Three independent ridge objectives: mean((estimated_error - OOF_error)**2)
+ l2 * ||w||**2, with an unpenalized intercept. All errors use fractional SOH.
"""
import numpy as np
import torch

from scripts.models.adaptive_models import (LOOKBACKS, context_at, fit_context_ridge,
                             normalize_window, transform_context)


def fit_error_selector(x, errors, l2=0.1):
    x = np.asarray(x, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] != 15 or not len(x)
            or errors.shape != (len(x), 3) or not np.isfinite(x).all()
            or not np.isfinite(errors).all() or (errors < 0).any()
            or not np.isfinite(l2) or l2 <= 0):
        raise ValueError('Expected N x 15 context, N x 3 nonnegative OOF errors, positive L2')
    fits = [fit_context_ridge(x, errors[:, j], l2) for j in range(3)]
    return np.column_stack([w for w, _ in fits]), np.array([b for _, b in fits])


def select_error_index(context, weight, bias):
    estimates = np.asarray(context, dtype=np.float64) @ weight + bias
    if estimates.shape != (3,) or not np.isfinite(estimates).all():
        raise ValueError('Expected three finite error estimates')
    # Linear regression can extrapolate below zero. Do not clip: clipping would
    # collapse distinct scores to ties. Record negative estimates as diagnostics.
    # Candidate order 10, 20, 50 makes an exact tie choose the shortest length.
    return int(np.argmin(estimates)), estimates


def infer_error_one(history, n, predictors, parameters):
    raw = context_at(history, n)
    context = transform_context(raw, parameters['context_mean'], parameters['context_std'])
    index, _ = select_error_index(context, parameters['error_weight'], parameters['error_bias'])
    length = int(LOOKBACKS[index])
    x = normalize_window(history, n, length,
                         parameters['channel_mean'], parameters['channel_std'])
    with torch.inference_mode():
        prediction = float(predictors[length](x).item())
    return prediction, length
