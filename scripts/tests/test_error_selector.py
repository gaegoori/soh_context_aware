"""Behavioral checks for the error selector, using synthetic data only."""
import unittest
import numpy as np
import torch
from scripts.models.adaptive_models import CycleHistory, context_at
from scripts.models.error_selector_models import fit_error_selector, infer_error_one, select_error_index


class ErrorSelectorTest(unittest.TestCase):
    def test_independent_augmented_least_squares(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(31, 15))
        y = np.abs(rng.normal(size=(31, 3)))
        w, b = fit_error_selector(x, y, .1)
        design = np.column_stack((x, np.ones(len(x))))
        penalty = np.diag(np.r_[np.full(15, np.sqrt(len(x)*.1)), 0.])
        want = np.linalg.lstsq(np.vstack((design, penalty)),
                              np.vstack((y, np.zeros((16, 3)))), rcond=None)[0]
        np.testing.assert_allclose(np.vstack((w, b)), want, atol=1e-12)
        # Duplicating every sample leaves a mean-loss regularized fit unchanged.
        w2, b2 = fit_error_selector(np.tile(x, (2, 1)), np.tile(y, (2, 1)), .1)
        np.testing.assert_allclose(w, w2, atol=1e-12)
        np.testing.assert_allclose(b, b2, atol=1e-12)

    def test_ties_and_unclipped_negative_scores(self):
        x, w = np.zeros(15), np.zeros((15, 3))
        self.assertEqual(select_error_index(x, w, np.ones(3))[0], 0)
        self.assertEqual(select_error_index(x, w, np.array([.1, -.1, -.2]))[0], 2)

    def test_reject_invalid_training_targets(self):
        for y in (np.zeros((3, 2)), -np.ones((3, 3)), np.full((3, 3), np.nan)):
            with self.assertRaises(ValueError):
                fit_error_selector(np.ones((3, 15)), y)

    def test_only_chosen_predictor_and_past_history(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(70, 128, 4)).astype(np.float32)
        history = CycleHistory(x, np.arange(70.)+1, np.ones(70),
                               np.arange(70.)+1, np.arange(70.)*100, 0., np.zeros(70))
        params = dict(context_mean=np.zeros(15), context_std=np.ones(15),
                      channel_mean=np.zeros(4), channel_std=np.ones(4),
                      error_weight=np.zeros((15, 3)), error_bias=np.array([2., 1., 3.]))
        calls = []
        class Recorder:
            def __init__(self, length): self.length = length
            def __call__(self, data):
                calls.append(self.length)
                self_shape = (1, self.length, 128, 4)
                assert tuple(data.shape) == self_shape
                np.testing.assert_array_equal(data.numpy()[0], x[60-self.length:60])
                return data.mean().reshape(1)
        bank = {lb:Recorder(lb) for lb in (10,20,50)}
        before_context = context_at(history, 60)
        before = infer_error_one(history, 60, bank, params)
        self.assertEqual(calls, [20])
        history.x[60:] = 9999
        history.duration[60:] = 8888
        history.cumulative_ah[60:] = 7777
        history.end_time[60:] = 1e12
        np.testing.assert_array_equal(context_at(history, 60), before_context)
        self.assertEqual(infer_error_one(history, 60, bank, params), before)
        self.assertEqual(calls, [20,20])


if __name__ == '__main__':
    unittest.main()
