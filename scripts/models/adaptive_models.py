"""Causal context features, fixed regularized models, and one-sample inference.

No function on the inference path accepts SOH targets or optimal-lookback labels.
Only the selected predictor is executed by the Adaptive method.
"""
import csv
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
import torch

from scripts.preprocessing.preprocess_main import CONTEXT_NAMES

LOOKBACKS = np.array([10, 20, 50], dtype=np.int64)
DEPLOYABLE_METHODS = ('Fixed10', 'Fixed20', 'Fixed50', 'GlobalBestOOF',
                      'LabelMajority', 'Adaptive', 'ContextOnlyRidge')
ALL_METHODS = DEPLOYABLE_METHODS + ('Oracle',)


@dataclass
class CycleHistory:
    """Canonical completed-cycle signals and incrementally cacheable metadata."""
    x: np.ndarray
    duration: np.ndarray
    discharge_ah: np.ndarray
    cumulative_ah: np.ndarray
    end_time: np.ndarray
    first_start: float
    temperature_missing: np.ndarray

    @classmethod
    def load(cls, root, battery):
        root = Path(root)
        with np.load(root / f'{battery}.npz', allow_pickle=False) as data:
            # Explicit allowlist: neither reference SOH nor interpolated SOH is loaded.
            x = data['x'].copy()
        with (root / f'{battery}_cycles.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != len(x):
            raise ValueError('Cycle metadata and signal arrays do not align')
        col = lambda key: np.array([float(r[key]) for r in rows], dtype=np.float64)
        ah = col('discharge_Ah')
        return cls(x, col('duration_s'), ah, np.cumsum(ah), col('end_time_s'),
                   float(rows[0]['start_time_s']), col('temperature_missing_fraction'))


def context_at(history, history_end_exclusive):
    """Reproduce the 15 canonical features using only completed past cycles."""
    n = int(history_end_exclusive)
    if not 1 <= n <= len(history.x):
        raise ValueError('Invalid past-history endpoint')
    i, lo = n - 1, max(0, n - 10)
    window = history.x[lo:n]
    duration = history.duration[lo:n]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        moments = [float(fn(window[:, :, channel])) for channel in (0, 1, 3)
                   for fn in (np.nanmean, np.nanstd)]
    values = [n, (history.end_time[i] - history.first_start) / 86400,
              history.cumulative_ah[i], history.duration[i], history.discharge_ah[i],
              history.temperature_missing[i], float(duration.mean()), float(duration.std()),
              float(np.polyfit(np.arange(len(duration)), duration, 1)[0])
              if len(duration) > 1 else 0., *moments]
    return np.asarray(values, dtype=np.float32)


def fit_context_scaler(training_context):
    x = np.asarray(training_context, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(CONTEXT_NAMES) or np.isinf(x).any():
        raise ValueError('Expected N x 15 finite-or-NaN context values')
    if np.any(np.all(np.isnan(x), axis=0)):
        raise ValueError('A training feature is entirely missing; change the protocol explicitly')
    mean, std = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
    return mean, np.where(std > 1e-6, std, 1.)


def transform_context(raw, mean, std):
    x = np.asarray(raw, dtype=np.float64)
    return (np.where(np.isfinite(x), x, mean) - mean) / std


def selector_objective(parameters, x, classes, l2):
    """Mean multinomial log loss + l2/2 * ||W||^2; bias is unpenalized."""
    dimensions = x.shape[1]
    weight = parameters[:dimensions * 3].reshape(dimensions, 3)
    bias = parameters[dimensions * 3:]
    logits = x @ weight + bias
    probabilities = softmax(logits, axis=1)
    loss = np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(x)), classes])
    loss += 0.5 * l2 * np.square(weight).sum()
    residual = probabilities.copy()
    residual[np.arange(len(x)), classes] -= 1
    residual /= len(x)
    gradient = np.concatenate([(x.T @ residual + l2 * weight).ravel(), residual.sum(axis=0)])
    return float(loss), gradient


def fit_selector(x, classes, l2=0.1):
    x, classes = np.asarray(x, dtype=np.float64), np.asarray(classes, dtype=np.int64)
    if x.ndim != 2 or classes.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError('Invalid selector training arrays')
    if set(classes) != {0, 1, 2}:
        raise ValueError('This predeclared three-class experiment requires all training classes')
    result = minimize(selector_objective, np.zeros(x.shape[1] * 3 + 3),
                      args=(x, classes, l2), method='L-BFGS-B', jac=True,
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'Selector optimizer failed: {result.message}')
    weight, bias = result.x[:-3].reshape(x.shape[1], 3), result.x[-3:]
    stats = dict(objective=float(result.fun), iterations=int(result.nit),
                 gradient_max_abs=float(np.abs(result.jac).max()),
                 optimizer_message=str(result.message),
                 training_accuracy_diagnostic=float(np.mean(np.argmax(x @ weight + bias, axis=1) == classes)))
    return weight, bias, stats


def fit_context_ridge(x, soh, l2=0.1):
    """Minimize mean squared error + l2*||w||^2; unpenalized intercept."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(soh, dtype=np.float64)
    if y.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid context-only training arrays')
    design = np.column_stack([x, np.ones(len(x))])
    penalty = np.eye(design.shape[1]) * (l2 * len(x))
    penalty[-1, -1] = 0.
    solution = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return solution[:-1], float(solution[-1])


def select_index(context, weight, bias):
    probability = softmax(context @ weight + bias)
    return int(np.argmax(probability)), probability


def normalize_window(history, n, lookback, mean, std):
    n, lookback = int(n), int(lookback)
    if lookback not in LOOKBACKS or not lookback <= n <= len(history.x):
        raise ValueError('Invalid predictor window')
    x = history.x[n-lookback:n]
    mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
    normalized = ((np.where(np.isfinite(x), x, mean) - mean) / std).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(normalized[None]))


def infer_one(history, n, predictors, parameters, method):
    """Start from resampled signals + cached cycle metadata; return prediction and L."""
    if method not in DEPLOYABLE_METHODS:
        raise ValueError('Oracle is not a deployable inference method')
    if method in ('Adaptive', 'ContextOnlyRidge'):
        raw = context_at(history, n)
        context = transform_context(raw, parameters['context_mean'], parameters['context_std'])
        if method == 'ContextOnlyRidge':
            return float(context @ parameters['ridge_weight'] + parameters['ridge_bias']), None
        index, _ = select_index(context, parameters['selector_weight'], parameters['selector_bias'])
        lookback = int(LOOKBACKS[index])
    elif method.startswith('Fixed'):
        lookback = int(method[5:])
    else:
        lookback = int(parameters['global_best_lb' if method == 'GlobalBestOOF' else 'majority_lb'])
    x = normalize_window(history, n, lookback, parameters['channel_mean'], parameters['channel_std'])
    with torch.inference_mode():
        prediction = float(predictors[lookback](x).item())
    return prediction, lookback
