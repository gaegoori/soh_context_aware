"""Checks for the experiment's actual failure risks, not metric snapshots."""
import copy
import unittest
import numpy as np
import torch
from scripts.preprocessing.battery_data import DEFAULT_ROOT
from scripts.models.context_history_models import (HistoryPredictor, VARIANTS, CONTEXT_GROUPS,
    train_bundle, input_arrays)
from scripts.experiments.train_lookback_oof import set_seed, state_hash


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        self.wave = torch.randn(3, 10, 128, 4)
        self.meta = torch.randn(3, 10, 2)
        self.context = torch.randn(3, 15)

    def test_same_capacity_initialization(self):
        hashes, counts = [], []
        for variant in VARIANTS:
            set_seed(42)
            m = HistoryPredictor(variant)
            hashes.append(state_hash(m))
            counts.append(sum(p.numel() for p in m.parameters()))
        self.assertEqual(len(set(hashes)), 1)
        self.assertEqual(len(set(counts)), 1)

    def test_disabled_inputs_do_not_change_predictions(self):
        for name in VARIANTS:
            m = HistoryPredictor(name).eval()
            base = m(self.wave, self.meta, self.context)
            if not m.metadata_on:
                torch.testing.assert_close(base, m(self.wave, self.meta * 100+25, self.context))
            if not m.context_on:
                torch.testing.assert_close(base, m(self.wave, self.meta, self.context*100+25))

    def test_permutation_and_order_sensitivity(self):
        for name in VARIANTS:
            m = HistoryPredictor(name).eval()
            a = m(self.wave, self.meta, self.context)
            b = m(self.wave.flip(1), self.meta.flip(1), self.context)
            if m.trend_on:
                self.assertGreater(float((a-b).abs().max()), 1e-5)
            else:
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)

    def test_group_masks(self):
        self.assertTrue(set(CONTEXT_GROUPS['ContextNoDurationAh']).isdisjoint([2,3,4,6,7,8]))
        self.assertTrue(set(CONTEXT_GROUPS['ContextNoTemperature']).isdisjoint([5,13,14]))

    def test_future_and_target_poisoning(self):
        split, rows, histories, scalers = train_bundle(DEFAULT_ROOT, 'RW3')
        self.assertNotIn('RW3', histories)
        row = rows[0]
        n, b = int(row['history_end_exclusive']), row['battery']
        before = input_arrays([row], histories, scalers, 50)
        poison = copy.deepcopy(histories)
        for array in (poison[b].x, poison[b].duration, poison[b].discharge_ah,
                      poison[b].cumulative_ah, poison[b].end_time, poison[b].temperature_missing):
            array[n:] = 99999
        bad_row = dict(row, soh='9999', capacity_Ah='9999', stage='POISONED')
        after = input_arrays([bad_row], poison, scalers, 50)
        for a,c in zip(before, after):
            torch.testing.assert_close(a,c,rtol=0,atol=0)
        # The last metadata entry corresponds exactly to the last input cycle.
        restored = before[1][0,-1].numpy()*np.array(scalers['metadata_std'])+scalers['metadata_mean']
        np.testing.assert_allclose(restored, [histories[b].duration[n-1], histories[b].discharge_ah[n-1]], rtol=1e-6)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
