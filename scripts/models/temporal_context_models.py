"""Causal variable-window statistics, ridge predictors and two selectors.

L changes nine recent-window statistics, not the six shared cumulative/last-cycle
features. All prediction paths are target-free and execute one chosen predictor.
"""
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from scripts.models.adaptive_models import LOOKBACKS, fit_context_ridge
from scripts.models.error_selector_models import fit_error_selector, select_error_index

L2 = 0.1
METHODS = ('Usage3', 'LastState6', 'FixedContext10', 'FixedContext20', 'FixedContext50',
           'GlobalBestOOF', 'LabelMajority', 'Adaptive', 'ErrorAware', 'Oracle')
DEPLOYABLE = METHODS[:-1]


def fixed_features(history, n):
    n = int(n)
    if not 1 <= n <= len(history.x):
        raise ValueError('Invalid past-history endpoint')
    i = n - 1
    return np.array([n, (history.end_time[i]-history.first_start)/86400,
        history.cumulative_ah[i], history.duration[i], history.discharge_ah[i],
        history.temperature_missing[i]], dtype=np.float32)


def window_features(history, n, length):
    n, length = int(n), int(length)
    if length not in LOOKBACKS or not length <= n <= len(history.x):
        raise ValueError('A complete candidate history is required')
    window = history.x[n-length:n]
    duration = history.duration[n-length:n]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        moments = [float(fn(window[:, :, ch])) for ch in (0, 1, 3)
                   for fn in (np.nanmean, np.nanstd)]
    return np.array([float(duration.mean()), float(duration.std()),
        float(np.polyfit(np.arange(length), duration, 1)[0]), *moments], dtype=np.float32)


def context_at_L(history, n, length):
    return np.r_[fixed_features(history, n), window_features(history, n, length)]


def fit_scaler(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or not len(raw) or np.isinf(raw).any():
        raise ValueError('Expected a nonempty matrix of finite-or-NaN values')
    active = np.isfinite(raw).any(axis=0)
    mean = np.zeros(raw.shape[1]); std = np.ones(raw.shape[1])
    if active.any():
        mean[active] = np.nanmean(raw[:, active], axis=0)
        std[active] = np.nanstd(raw[:, active], axis=0)
    std = np.where(std > 1e-6, std, 1.)
    return dict(mean=mean, std=std, active=active)


def transform(raw, scaler):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape[-1] != len(scaler['mean']) or np.isinf(raw).any():
        raise ValueError('Invalid context input')
    out = (np.where(np.isfinite(raw), raw, scaler['mean'])-scaler['mean'])/scaler['std']
    return np.where(scaler['active'], out, 0.)


def fit_ridge(raw, targets):
    scaler = fit_scaler(raw)
    weight, bias = fit_context_ridge(transform(raw, scaler), targets, L2)
    return dict(**scaler, weight=weight, bias=np.asarray(bias))


def predict_ridge(raw, model):
    return transform(raw, model) @ model['weight'] + model['bias']


def fit_classifier(x, labels):
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or labels.shape != (len(x),) or not len(x) or not np.isfinite(x).all():
        raise ValueError('Invalid classifier data')
    observed, target = np.unique(labels, return_inverse=True)
    if not set(observed).issubset({0, 1, 2}):
        raise ValueError('Invalid candidate classes')
    d, k = x.shape[1], len(observed)
    if k == 1:
        return dict(class_labels=observed, class_weight=np.zeros((d, 1)),
                    class_bias=np.zeros(1)), dict(kind='constant', iterations=0, gradient_max_abs=0.)
    def objective(theta):
        w, b = theta[:-k].reshape(d, k), theta[-k:]
        logits = x @ w + b
        residual = softmax(logits, axis=1)-np.eye(k)[target]
        loss = np.mean(logsumexp(logits, axis=1)-logits[np.arange(len(x)), target])
        loss += L2/2*np.square(w).sum()
        grad = np.r_[(x.T @ residual/len(x)+L2*w).ravel(), residual.mean(axis=0)]
        return float(loss), grad
    result = minimize(objective, np.zeros(d*k+k), method='L-BFGS-B', jac=True,
                      options=dict(maxiter=500, ftol=1e-12, gtol=1e-8))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'Selector optimizer failed: {result.message}')
    return dict(class_labels=observed, class_weight=result.x[:-k].reshape(d,k),
                class_bias=result.x[-k:]), dict(kind=f'{k}-class', iterations=int(result.nit),
                objective=float(result.fun), gradient_max_abs=float(np.abs(result.jac).max()))


def fit_selectors(raw_c10, errors, batteries):
    errors = np.asarray(errors, dtype=np.float64)
    batteries = np.asarray(batteries)
    scaler = fit_scaler(raw_c10)
    x = transform(raw_c10, scaler)
    labels = np.argmin(errors, axis=1)
    classifier, stats = fit_classifier(x, labels)
    w, b = fit_error_selector(x, errors, L2)
    macro = np.mean([errors[batteries == cell].mean(axis=0) for cell in np.unique(batteries)], axis=0)
    counts = np.bincount(labels, minlength=3)
    return dict(**scaler, **classifier, error_weight=w, error_bias=b,
        global_best_lb=np.asarray(LOOKBACKS[np.argmin(macro)]),
        majority_lb=np.asarray(LOOKBACKS[np.argmax(counts)])), dict(
        classifier=stats, class_counts=counts.tolist(), oof_cell_macro_mae=macro.tolist(),
        inactive_context_features=np.flatnonzero(~scaler['active']).tolist())


def choose(raw_c10, selector, method):
    x = transform(raw_c10, selector)
    if method == 'Adaptive':
        observed = selector['class_labels']
        probs = np.zeros(3)
        probs[observed] = softmax(x @ selector['class_weight'] + selector['class_bias'])
        return int(LOOKBACKS[np.argmax(probs)]), probs
    if method == 'ErrorAware':
        index, estimates = select_error_index(x, selector['error_weight'], selector['error_bias'])
        return int(LOOKBACKS[index]), estimates
    raise ValueError('Expected a learned selector method')


def infer_one(history, n, bank, selector, method):
    if method not in DEPLOYABLE:
        raise ValueError('Oracle is posthoc only')
    if method in ('Usage3', 'LastState6'):
        raw = fixed_features(history, n)[:3 if method == 'Usage3' else 6]
        return float(predict_ridge(raw, bank[method])), None
    raw = None
    if method in ('Adaptive', 'ErrorAware'):
        raw = context_at_L(history, n, 10)
        length, _ = choose(raw, selector, method)
        # Reuse C10 if selected, and the same six shared features otherwise.
        if length != 10:
            raw = np.r_[raw[:6], window_features(history, n, length)]
    elif method.startswith('FixedContext'):
        length = int(method[len('FixedContext'):])
    else:
        length = int(selector['global_best_lb' if method == 'GlobalBestOOF' else 'majority_lb'])
    if raw is None:
        raw = context_at_L(history, n, length)
    return float(predict_ridge(raw, bank[length])), length
