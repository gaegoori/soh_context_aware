"""Tests for gradient correctness, causal features and selected-expert routing."""
import copy
import unittest

import numpy as np
from scipy.optimize import approx_fprime
import torch

from scripts.models.adaptive_models import (CycleHistory, context_at, fit_context_scaler,
    fit_selector, infer_one, selector_objective, transform_context)
from scripts.preprocessing.preprocess_main import CONTEXT_NAMES
from scripts.experiments.train_adaptive_selector import align_selector_data


class AdaptiveTests(unittest.TestCase):
    def history(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(70, 128, 4)).astype(np.float32)
        duration = np.linspace(10, 30, len(x))
        ah = np.linspace(.1, .4, len(x))
        return CycleHistory(x, duration, ah, np.cumsum(ah), np.cumsum(duration), 0.,
                            np.zeros(len(x)))

    def test_multinomial_gradient_and_optimum(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(30, 15))
        y = np.tile([0, 1, 2], 10)
        initial = rng.normal(size=48) * .1
        _, analytic = selector_objective(initial, x, y, .1)
        numeric = approx_fprime(initial, lambda p: selector_objective(p, x, y, .1)[0], 1e-6)
        np.testing.assert_allclose(analytic, numeric, atol=1e-6, rtol=2e-4)
        w, b, stats = fit_selector(x, y)
        loss, gradient = selector_objective(np.r_[w.ravel(), b], x, y, .1)
        self.assertLess(np.max(np.abs(gradient)), 1e-5)
        self.assertLess(loss, np.log(3))
        self.assertTrue(np.isfinite(stats['objective']))

    def test_scaler_is_not_refit_at_inference(self):
        x = np.arange(45, dtype=float).reshape(3, 15)
        x[0, 0] = np.nan
        mean, std = fit_context_scaler(x)
        saved_mean, saved_std = mean.copy(), std.copy()
        test = np.full(15, 1e6)
        test[0] = np.nan
        result = transform_context(test, mean, std)
        self.assertEqual(result[0], 0.)
        self.assertGreater(result[1], 1000.)
        np.testing.assert_array_equal(mean, saved_mean)
        np.testing.assert_array_equal(std, saved_std)

    def test_future_signals_and_metadata_cannot_change_context(self):
        history = self.history()
        expected = context_at(history, 50)
        poisoned = copy.deepcopy(history)
        for name in ('x', 'duration', 'discharge_ah', 'cumulative_ah', 'end_time', 'temperature_missing'):
            getattr(poisoned, name)[50:] = 1e10
        np.testing.assert_array_equal(context_at(poisoned, 50), expected)
        history.x[:50, :, 3] = np.nan
        self.assertTrue(np.isnan(context_at(history, 50)[-2:]).all())

    def test_adaptive_runs_exactly_one_selected_expert(self):
        calls = []
        class Expert:
            def __init__(self, length): self.length = length
            def __call__(self, x):
                calls.append((self.length, tuple(x.shape)))
                return torch.tensor([.8])
        params = dict(context_mean=np.zeros(15), context_std=np.ones(15),
            selector_weight=np.zeros((15, 3)), selector_bias=np.array([0., 4., 0.]),
            channel_mean=np.zeros(4), channel_std=np.ones(4))
        history = self.history()
        bank = {lb: Expert(lb) for lb in (10, 20, 50)}
        prediction, lb = infer_one(history, 50, bank, params, 'Adaptive')
        self.assertEqual(lb, 20)
        self.assertEqual(calls, [(20, (1, 20, 128, 4))])
        self.assertAlmostEqual(prediction, .8)
        with self.assertRaises(ValueError):
            infer_one(history, 50, bank, params, 'Oracle')

    def test_label_alignment_uses_ids_and_rejects_test_cell(self):
        data = dict(sample_id=np.array(['b', 'a']), battery=np.array(['RW4', 'RW3']),
                    context_names=CONTEXT_NAMES, context_raw=np.ones((2, 15)),
                    mean_errors=np.array([[.3, .2, .1], [.1, .2, .3]]),
                    class_label=np.array([2, 0]), best_lookback=np.array([50, 10]))
        rows = [dict(sample_id='a', battery='RW3'), dict(sample_id='b', battery='RW4')]
        _, classes, _, _ = align_selector_data(data, rows, 'RW6')
        np.testing.assert_array_equal(classes, [0, 2])
        with self.assertRaises(ValueError):
            align_selector_data(data, rows, 'RW3')


if __name__ == '__main__':
    unittest.main()
