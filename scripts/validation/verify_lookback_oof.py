"""Independently check every OOF artifact, label, split, and saved prediction."""
from scripts.paths import SCRIPTS, hash_matches
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.preprocessing.battery_data import BatteryDataset, DEFAULT_ROOT
from scripts.experiments.train_lookback_oof import (CyclePredictor, DEFAULT_OUTPUT, LOOKBACKS,
                               read_csv, sha256, state_hash, write_json)


def verify(root, output):
    protocol = json.loads((output / 'protocol.json').read_text())
    run = json.loads((output / 'run_summary.json').read_text())
    assert run['status'] == 'TRAINING_AND_LABELS_COMPLETE'
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    assert signature == run['protocol_sha256']
    for name, digest in protocol['files_sha256'].items():
        assert hash_matches(root / name, digest), f'Changed input: {name}'
    for name, digest in protocol['code_sha256'].items():
        assert hash_matches(SCRIPTS / name, digest), f'Changed source: {name}'
    torch.set_num_threads(protocol['threads'])
    torch.use_deterministic_algorithms(True)
    samples = {r['sample_id']: r for r in read_csv(root / 'samples.csv')}
    manifest = json.loads((root / 'manifest.json').read_text())
    batteries = manifest['batteries']
    seeds = run['seeds']
    predictions = read_csv(output / 'predictions.csv')
    labels = read_csv(output / 'labels.csv')
    by_key = {}
    for row in predictions:
        key = (row['outer_test_battery'], row['sample_id'], int(row['seed']), int(row['lookback']))
        assert key not in by_key
        by_key[key] = row
    expected_keys = {(outer, sid, seed, lb) for outer in batteries
                     for sid, sample in samples.items() if sample['battery'] != outer
                     for seed in seeds for lb in LOOKBACKS}
    assert set(by_key) == expected_keys
    assert len(predictions) == run['prediction_rows']
    assert len(samples) == run['unique_reference_samples']
    init_hashes = defaultdict(set)
    checked_models = 0
    for fold in protocol['folds']:
        split = json.loads((root / 'folds' / fold / 'split.json').read_text())
        train, heldout, test = [set(split[k]) for k in ('train', 'validation', 'test')]
        assert not (train & heldout or train & test or heldout & test)
        tr_b, va_b, te_b = [set(split[k + '_batteries']) for k in ('train', 'validation', 'test')]
        assert not (tr_b & va_b or tr_b & te_b or va_b & te_b)
        assert (len(tr_b), len(va_b), len(te_b)) == (2, 1, 1)
        scale = json.loads((root / 'folds' / fold / 'scalers.json').read_text())
        assert set(scale['fit_sample_ids']) == train
        ds = BatteryDataset(fold, 'validation', lookback=50, root=root)
        items = [ds[i] for i in range(len(ds))]
        x50 = torch.from_numpy(np.stack([r['x'] for r in items]))
        outer, heldout_battery = next(iter(te_b)), next(iter(va_b))
        for seed in seeds:
            for lb in LOOKBACKS:
                directory = output / 'runs' / fold / f'seed_{seed}' / f'lb_{lb}'
                complete = json.loads((directory / 'complete.json').read_text())
                assert complete['protocol_sha256'] == signature
                for name, digest in complete['files_sha256'].items():
                    assert hash_matches(directory / name, digest)
                checkpoint = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True)
                assert checkpoint['protocol_sha256'] == signature
                assert checkpoint['epochs'] == protocol['epochs']
                assert checkpoint['lookback'] == lb and checkpoint['seed'] == seed
                assert checkpoint['fold'] == fold
                assert set(checkpoint['train_sample_ids']) == train
                assert set(checkpoint['train_batteries']) == tr_b
                assert checkpoint['heldout_battery'] == heldout_battery
                assert checkpoint['outer_test_battery'] == outer
                assert checkpoint['scalers'] == scale
                init_hashes[seed].add(checkpoint['initial_state_sha256'])
                history = read_csv(directory / 'training_history.csv')
                assert [int(r['epoch']) for r in history] == list(range(1, protocol['epochs'] + 1))
                assert set(history[0]) == {'epoch', 'train_mse_before_update'}
                assert np.isfinite([float(r['train_mse_before_update']) for r in history]).all()
                model = CyclePredictor()
                model.load_state_dict(checkpoint['state_dict'])
                model.eval()
                with torch.inference_mode():
                    actual_predictions = model(x50[:, -lb:].contiguous()).numpy()
                local = read_csv(directory / 'predictions.csv')
                assert len(local) == len(items)
                assert [r['sample_id'] for r in local] == [r['sample_id'] for r in items]
                stats = json.loads((directory / 'metrics.json').read_text())
                assert stats['final_state_sha256'] == state_hash(model)
                errors = []
                for item, saved, actual in zip(items, local, actual_predictions):
                    sid = item['sample_id']
                    row = by_key[outer, sid, seed, lb]
                    assert saved == row
                    assert row['heldout_battery'] == heldout_battery != outer
                    assert row['stage'] == samples[sid]['stage']
                    assert float(row['soh_true']) == float(samples[sid]['soh'])
                    np.testing.assert_allclose(float(row['soh_pred']), actual, rtol=1e-6, atol=1e-7)
                    error = abs(float(row['soh_pred'])-float(row['soh_true']))
                    np.testing.assert_allclose(float(row['absolute_error']), error, rtol=1e-12)
                    errors.append(error)
                np.testing.assert_allclose(stats['mae'], np.mean(errors), rtol=1e-12)
                np.testing.assert_allclose(stats['rmse'], np.sqrt(np.mean(np.square(errors))), rtol=1e-12)
                checked_models += 1
    assert all(len(values) == 1 for values in init_hashes.values())
    for seed, hashes in init_hashes.items():
        torch.manual_seed(seed)
        assert state_hash(CyclePredictor()) == next(iter(hashes))
    assert checked_models == 12 * len(seeds) * 3 == run['models']
    label_map = {}
    for row in labels:
        outer, sid = row['outer_test_battery'], row['sample_id']
        assert (outer, sid) not in label_map
        label_map[outer, sid] = row
        errors = np.array([[abs(float(by_key[outer, sid, seed, lb]['soh_pred']) - float(samples[sid]['soh']))
                            for lb in LOOKBACKS] for seed in seeds])
        means = errors.mean(axis=0)
        index = int(np.argmin(means))
        assert int(row['best_lookback']) == LOOKBACKS[index]
        assert int(row['class_label']) == index
        assert int(row['seed_count']) == len(seeds)
        for j, lb in enumerate(LOOKBACKS):
            np.testing.assert_allclose(float(row[f'mean_error_lb{lb}']), means[j], rtol=1e-12)
            assert int(row[f'seed_wins_lb{lb}']) == int(np.sum(np.argmin(errors, axis=1) == j))
        ordered = sorted(means)
        np.testing.assert_allclose(float(row['error_gap_to_second_best']), ordered[1]-ordered[0], atol=1e-14)
        np.testing.assert_allclose(float(row['seed_agreement']), np.mean(np.argmin(errors, axis=1) == index))
    expected_label_keys = {(outer, sid) for outer in batteries for sid, r in samples.items() if r['battery'] != outer}
    assert set(label_map) == expected_label_keys
    assert len(labels) == run['label_rows']
    seed_label_map = {}
    for row in read_csv(output / 'labels_by_seed.csv'):
        key = row['outer_test_battery'], row['sample_id'], int(row['seed'])
        assert key not in seed_label_map
        seed_label_map[key] = row
        errors = [float(by_key[key + (lb,)]['absolute_error']) for lb in LOOKBACKS]
        assert int(row['best_lookback']) == LOOKBACKS[int(np.argmin(errors))]
        np.testing.assert_allclose([float(row[f'error_lb{lb}']) for lb in LOOKBACKS], errors)
    assert set(seed_label_map) == {(outer, sid, seed) for outer, sid in expected_label_keys for seed in seeds}
    contexts = {}
    for b in batteries:
        with np.load(root / f'{b}.npz', allow_pickle=False) as data:
            contexts[b] = data['context']
    counts = {}
    for outer in batteries:
        folder = output / 'selector_data' / f'test_{outer}'
        schema = json.loads((folder / 'schema.json').read_text())
        assert schema['input_keys'] == ['context_raw']
        assert schema['feature_names'] == manifest['context_names']
        with np.load(folder / 'dataset.npz', allow_pickle=False) as ds:
            assert list(ds['context_names']) == manifest['context_names']
            assert len(set(ds['sample_id'])) == len(ds['sample_id'])
            expected = {sid for o, sid in expected_label_keys if o == outer}
            assert set(ds['sample_id']) == expected
            assert outer not in ds['battery']
            assert ds['context_raw'].shape == (len(expected), 15)
            assert np.array_equal(ds['context_observed_mask'], np.isfinite(ds['context_raw']))
            for i, sid in enumerate(ds['sample_id']):
                sample = samples[sid]
                original = contexts[sample['battery']][int(sample['history_end_exclusive'])-1]
                np.testing.assert_allclose(ds['context_raw'][i], original, rtol=0, atol=0, equal_nan=True)
                row = label_map[outer, sid]
                assert ds['best_lookback'][i] == int(row['best_lookback'])
                assert ds['class_label'][i] == int(row['class_label'])
                np.testing.assert_allclose(ds['mean_errors'][i], [float(row[f'mean_error_lb{lb}']) for lb in LOOKBACKS])
                np.testing.assert_allclose(ds['seed_agreement'][i], float(row['seed_agreement']))
            counts[outer] = len(expected)
    result = dict(status='PASS', models=checked_models, seeds=seeds, epochs=protocol['epochs'],
                  predictions=len(predictions), label_rows=len(labels), labels_by_outer_test=counts,
                  unique_reference_samples=len(samples),
                  checks=['input/source/artifact SHA256', 'battery-disjoint train/label/test sets',
                          'complete common sample/seed/lookback grids', 'fixed final checkpoint epoch',
                          'same seed gives identical initial weights across lookbacks and folds',
                          'all saved checkpoints reproduce held-out predictions', 'independent label/error recomputation',
                          'mean error rather than ensemble prediction error', 'raw context and missing-mask alignment',
                          'separate selector datasets for each outer test battery'],
                  scope='Inner OOF label generation only; not an outer-test evaluation.')
    write_json(output / 'verification.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    verify(args.data_root.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
