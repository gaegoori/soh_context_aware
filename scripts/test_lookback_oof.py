"""Small tests for label arithmetic and predictor isolation/reproducibility."""
import unittest
from pathlib import Path

import numpy as np
import torch

from train_lookback_oof import (LOOKBACKS, CyclePredictor, build_labels,
                               load_batch, state_hash, train_predictor)


def rows_for(predictions):
    return [dict(outer_test_battery='RW6', heldout_battery='RW3', sample_id='RW3_ref004',
                 stage='late', soh_true=0.5, seed=seed, lookback=lb, soh_pred=value)
            for seed, values in enumerate(predictions) for lb, value in zip(LOOKBACKS, values)]


class LabelTests(unittest.TestCase):
    def test_mean_error_not_error_of_mean_prediction(self):
        # LB10's errors cancel if predictions are averaged first; LB20 truly wins.
        labels, per_seed = build_labels(rows_for([[0.25, 0.625, 0.875], [0.75, 0.625, 0.875]]), [0, 1])
        self.assertEqual(labels[0]['best_lookback'], 20)
        self.assertEqual(labels[0]['mean_error_lb10'], 0.25)
        self.assertEqual(labels[0]['error_gap_to_second_best'], 0.125)
        self.assertEqual(labels[0]['seed_agreement'], 1.0)
        self.assertEqual(len(per_seed), 2)

    def test_exact_tie_selects_shortest(self):
        labels, _ = build_labels(rows_for([[0.625, 0.625, 0.625]]), [0])
        self.assertEqual(labels[0]['best_lookback'], 10)
        self.assertEqual(labels[0]['error_gap_to_second_best'], 0)

    def test_mean_winner_can_differ_from_seed_majority(self):
        labels, _ = build_labels(rows_for([[0.5, 0.625, 0.875], [0.5, 0.625, 0.875], [1.0, 0.625, 0.875]]), [0, 1, 2])
        self.assertEqual(labels[0]['best_lookback'], 20)
        self.assertAlmostEqual(labels[0]['seed_agreement'], 1/3)
        self.assertEqual(labels[0]['seed_wins_lb10'], 2)

    def test_incomplete_duplicate_or_misaligned_predictions_rejected(self):
        rows = rows_for([[0.25, 0.625, 0.875]])
        for bad in (rows[:-1], rows + [rows[0]], rows + rows_for([[0.5, 0.5, 0.5]])):
            with self.assertRaises(ValueError):
                build_labels(bad, [0])
        rows[0]['soh_true'] = 0.75
        with self.assertRaises(ValueError):
            build_labels(rows, [0])

    def test_outer_fold_labels_remain_separate(self):
        rows = rows_for([[0.5, 0.625, 0.875]])
        others = rows_for([[0.875, 0.625, 0.5]])
        for r in others:
            r['outer_test_battery'] = 'RW5'
        labels, _ = build_labels(rows + others, [0])
        self.assertEqual({r['outer_test_battery']: r['best_lookback'] for r in labels}, {'RW5': 50, 'RW6': 10})


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_initialization_matching_and_training_repeatability(self):
        x = torch.from_numpy(np.random.default_rng(7).normal(size=(3, 50, 128, 4)).astype(np.float32))
        y = torch.tensor([0.9, 0.8, 0.7])
        a = train_predictor(x[:, -10:], y, seed=42, epochs=2, learning_rate=0.001)
        b = train_predictor(x[:, -10:], y, seed=42, epochs=2, learning_rate=0.001)
        c = train_predictor(x, y, seed=42, epochs=2, learning_rate=0.001)
        self.assertEqual(state_hash(a[0]), state_hash(b[0]))
        self.assertEqual(a[1], b[1])
        self.assertEqual(a[2], c[2])
        self.assertEqual(len(a[1]), 2)

    def test_samples_do_not_mix_during_pooling(self):
        model = CyclePredictor().eval()
        x = torch.ones(2, 20, 128, 4)
        with torch.inference_mode():
            before = model(x).clone()
            x[1] = -100
            after = model(x)
        torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)

    def test_training_workflow_rejects_outer_test_batch(self):
        with self.assertRaises(ValueError):
            load_batch(Path('/unused'), 'unused', 'test')


if __name__ == '__main__':
    unittest.main()
