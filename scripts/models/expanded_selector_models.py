"""Target-free Selector rules; threshold units are fractional SOH absolute error."""
import numpy as np
from scripts.models.temporal_context_models import fit_scaler, transform, fit_classifier
from scripts.models.error_selector_models import fit_error_selector

LENGTHS = (10, 20, 50)
# Fixed before training. None means never switch, not an infinite JSON number.
THRESHOLDS = (0., .0005, .001, .002, .005, .01, .02, None)


def fit_selector(raw, errors, batteries, classifier=False):
    errors = np.asarray(errors, dtype=float)
    batteries = np.asarray(batteries)
    scaler = fit_scaler(raw)
    x = transform(raw, scaler)
    w, b = fit_error_selector(x, errors, .1)
    means = np.mean([errors[batteries == c].mean(axis=0)
                     for c in np.unique(batteries)], axis=0)
    model = dict(**scaler, error_weight=w, error_bias=b,
                 default_index=np.asarray(np.argmin(means)))
    stats = dict(oof_macro_errors=means.tolist())
    if classifier:
        params, optimizer = fit_classifier(x, np.argmin(errors, axis=1))
        model.update(params)
        stats['classifier_optimizer'] = optimizer
    return model, stats


def error_scores(raw, model):
    return transform(raw, model) @ model['error_weight'] + model['error_bias']


def classify(raw, model):
    logits = transform(raw, model) @ model['class_weight'] + model['class_bias']
    return model['class_labels'][np.argmax(logits, axis=-1)]


def gate(scores, default_index, threshold):
    """Switch only for predicted improvement > threshold. No true SOH input."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 3 or not np.isfinite(scores).all():
        raise ValueError('Expected finite N x 3 predicted error scores')
    if default_index not in (0, 1, 2):
        raise ValueError('Invalid default candidate index')
    if threshold is not None and (not np.isfinite(threshold) or threshold < 0):
        raise ValueError('Threshold must be nonnegative or None for never switch')
    best = np.argmin(scores, axis=1)
    gain = scores[:, default_index] - scores[np.arange(len(scores)), best]
    switch = np.zeros(len(scores), dtype=bool) if threshold is None else gain > threshold
    return np.where(switch, best, default_index), gain


def select_threshold(macro_maes, switches):
    """Lowest inner cell-macro MAE; ties within 1e-12 prefer fewer switches."""
    maes = np.asarray(macro_maes)
    if maes.shape != (len(THRESHOLDS),) or not np.isfinite(maes).all():
        raise ValueError('Invalid threshold validation scores')
    eligible = np.flatnonzero(maes <= maes.min() + 1e-12)
    return int(min(eligible, key=lambda j: (switches[j], -j)))
