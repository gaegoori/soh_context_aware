"""Independently reproduce outer predictions, train-only fits, metrics and timing grids."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import torch

from battery_data import BatteryDataset, DEFAULT_ROOT
from train_lookback_oof import CyclePredictor, read_csv, sha256, state_hash, write_json

LENGTHS = (10, 20, 50)
METHODS = ('Fixed10', 'Fixed20', 'Fixed50', 'GlobalBestOOF', 'LabelMajority',
           'Adaptive', 'ContextOnlyRidge', 'Oracle')
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'outputs/adaptive_selector_v1'


def near(actual, expected, atol=1e-10):
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=atol, equal_nan=True)


def verify(root, output):
    protocol = json.loads((output / 'protocol.json').read_text())
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    run = json.loads((output / 'run_summary.json').read_text())
    assert run['status'] == 'OUTER_EVALUATION_COMPLETE'
    assert run['protocol_sha256'] == signature and not run['test_used_for_tuning']
    labels_root = Path(protocol['source_labels_root'])
    for base, hashes in ((root, protocol['data_sha256']),
                         (labels_root, protocol['labels_sha256']),
                         (Path(__file__).parent, protocol['code_sha256'])):
        for name, digest in hashes.items():
            assert sha256(base / name) == digest, f'Changed input/source: {base / name}'
    manifest = json.loads((output / 'artifact_manifest.json').read_text())
    assert manifest['protocol_sha256'] == signature
    for name, digest in manifest['files_sha256'].items():
        assert sha256(output / name) == digest, f'Changed output: {name}'
    frozen = json.loads((output / 'protocol_frozen.json').read_text())
    fitted = json.loads((output / 'fitting_complete.json').read_text())
    assert frozen['protocol_sha256'] == fitted['protocol_sha256'] == signature
    assert (datetime.fromisoformat(frozen['frozen_utc']) <=
            datetime.fromisoformat(fitted['completed_utc']) <=
            datetime.fromisoformat(run['completed_utc']))

    samples = {r['sample_id']: r for r in read_csv(root / 'samples.csv')}
    batteries, seeds = protocol['batteries'], protocol['seeds']
    canonical = {}
    for b in batteries:
        with np.load(root / f'{b}.npz', allow_pickle=False) as data:
            canonical[b] = {name: data[name].copy() for name in ('x', 'context', 'cycle_valid')}
    predictions = read_csv(output / 'predictions.csv')
    pred = {(r['outer_test_battery'], r['sample_id'], int(r['seed']), r['method']): r
            for r in predictions}
    expected_keys = {(r['battery'], sid, s, m) for sid, r in samples.items() for s in seeds for m in METHODS}
    assert len(pred) == len(predictions) == run['prediction_rows']
    assert set(pred) == expected_keys
    selected_rows = read_csv(output / 'selections.csv')
    selection = {r['sample_id']: r for r in selected_rows}
    assert len(selection) == len(selected_rows) == len(samples) == run['unique_test_samples']
    initial_states = defaultdict(set)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checked = 0
    for outer in batteries:
        folder = root / 'folds' / f'test_{outer}__final'
        split = json.loads((folder / 'split.json').read_text())
        scale = json.loads((folder / 'scalers.json').read_text())
        train = [samples[sid] for sid in split['train']]
        test = [samples[sid] for sid in split['test']]
        assert set(split['train_batteries']) == set(batteries) - {outer}
        assert split['test_batteries'] == [outer] and not split['validation']
        assert not set(split['train']) & set(split['test'])
        assert all(r['battery'] != outer for r in train) and all(r['battery'] == outer for r in test)
        assert set(scale['fit_sample_ids']) == set(split['train'])
        # Reconstruct exactly the unique training cycles used for signal statistics.
        signal_fit = []
        for b in split['train_batteries']:
            indices = sorted({i for r in train if r['battery'] == b
                              for i in range(int(r['history_end_exclusive'])-50, int(r['history_end_exclusive']))})
            assert indices == scale['fit_cycle_indices'][b]
            assert canonical[b]['cycle_valid'][indices].all()
            signal_fit.append(canonical[b]['x'][indices].reshape(-1, 4))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            signals = np.concatenate(signal_fit)
            mu, sigma = np.nanmean(signals, axis=0), np.nanstd(signals, axis=0)
        sigma = np.where(np.isfinite(sigma) & (sigma > 1e-6), sigma, 1.)
        near(mu, scale['channel_mean']); near(sigma, scale['channel_std'])
        model_root = output / 'models' / f'test_{outer}'
        context_folder = model_root / 'context_models'
        fit = json.loads((context_folder / 'fit.json').read_text())
        assert fit['train_sample_ids'] == split['train'] and fit['train_batteries'] == split['train_batteries']
        assert not fit['tuning_performed'] and fit['feature_names'] == protocol['features']
        with np.load(context_folder / 'parameters.npz', allow_pickle=False) as saved:
            p = {name: saved[name].copy() for name in saved.files}
        raw = np.array([canonical[r['battery']]['context'][int(r['history_end_exclusive'])-1]
                        for r in train], dtype=np.float64)
        mean, std = np.nanmean(raw, axis=0), np.nanstd(raw, axis=0)
        std = np.where(std > 1e-6, std, 1.)
        near(p['context_mean'], mean); near(p['context_std'], std)
        near(p['channel_mean'], mu); near(p['channel_std'], sigma)
        x = (np.where(np.isfinite(raw), raw, mean) - mean) / std
        with np.load(labels_root / 'selector_data' / f'test_{outer}' / 'dataset.npz', allow_pickle=False) as data:
            lookup = {sid: i for i, sid in enumerate(data['sample_id'])}
            order = [lookup[r['sample_id']] for r in train]
            classes, errors = data['class_label'][order], data['mean_errors'][order]
        counts = np.bincount(classes, minlength=3)
        near(counts, fit['class_counts'])
        assert int(p['majority_lb']) == LENGTHS[int(np.argmax(counts))]
        macro_errors = np.mean([errors[np.array([r['battery'] == b for r in train])].mean(axis=0)
                                for b in split['train_batteries']], axis=0)
        near(macro_errors, fit['oof_macro_mae_by_lookback'])
        assert int(p['global_best_lb']) == LENGTHS[int(np.argmin(macro_errors))]
        # Independent first-order optimality checks for both convex fitted models.
        logits = x @ p['selector_weight'] + p['selector_bias']
        probability = np.exp(logits - logits.max(axis=1, keepdims=True))
        probability /= probability.sum(axis=1, keepdims=True)
        residual = probability - np.eye(3)[classes]
        gradient = x.T @ residual / len(x) + protocol['selector']['l2'] * p['selector_weight']
        assert np.max(np.abs(gradient)) < 1e-5 and np.max(np.abs(residual.mean(axis=0))) < 1e-5
        ridge_residual = x @ p['ridge_weight'] + p['ridge_bias'] - np.array([float(r['soh']) for r in train])
        near(x.T @ ridge_residual / len(x) + protocol['context_only']['l2'] * p['ridge_weight'], 0.)
        near(ridge_residual.mean(), 0.)
        for row in test:
            sid, n = row['sample_id'], int(row['history_end_exclusive'])
            raw_test = canonical[outer]['context'][n-1].astype(np.float64)
            context = (np.where(np.isfinite(raw_test), raw_test, mean)-mean)/std
            logits = context @ p['selector_weight'] + p['selector_bias']
            probabilities = np.exp(logits-logits.max()); probabilities /= probabilities.sum()
            sel = selection[sid]
            assert sel['outer_test_battery'] == outer
            near([float(sel[f'p_lb{lb}']) for lb in LENGTHS], probabilities, atol=1e-6)
            assert int(sel['selected_lookback']) == LENGTHS[int(np.argmax(probabilities))]
            for seed in seeds:
                near(float(pred[outer, sid, seed, 'ContextOnlyRidge']['soh_pred']),
                     context @ p['ridge_weight'] + p['ridge_bias'], atol=1e-6)
        ds = BatteryDataset(f'test_{outer}__final', 'test', lookback=50, root=root)
        items = [ds[i] for i in range(len(ds))]
        for seed in seeds:
            for lb in LENGTHS:
                directory = model_root / f'seed_{seed}' / f'lb_{lb}'
                checkpoint = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True)
                assert checkpoint['protocol_sha256'] == signature
                assert checkpoint['epochs'] == 200 and checkpoint['seed'] == seed and checkpoint['lookback'] == lb
                assert checkpoint['train_sample_ids'] == split['train'] and checkpoint['train_batteries'] == split['train_batteries']
                assert checkpoint['outer_test_battery'] == outer and checkpoint['scalers'] == scale
                initial_states[seed].add(checkpoint['initial_state_sha256'])
                model = CyclePredictor(); model.load_state_dict(checkpoint['state_dict']); model.eval()
                fit_stats = json.loads((directory / 'fit.json').read_text())
                assert state_hash(model) == fit_stats['final_state_sha256']
                history = read_csv(directory / 'training_history.csv')
                assert [int(r['epoch']) for r in history] == list(range(1, 201))
                assert np.isfinite([float(r['train_mse_before_update']) for r in history]).all()
                with torch.inference_mode():
                    for item in items:
                        tensor = torch.from_numpy(item['x'][None, -lb:].copy())
                        actual = model(tensor).item()
                        near(actual, float(pred[outer, item['sample_id'], seed, f'Fixed{lb}']['soh_pred']), atol=1e-7)
                checked += 1
            for row in test:
                sid, truth = row['sample_id'], float(row['soh'])
                fixed = {lb: float(pred[outer, sid, seed, f'Fixed{lb}']['soh_pred']) for lb in LENGTHS}
                expected_lb = dict(Adaptive=int(selection[sid]['selected_lookback']),
                    GlobalBestOOF=int(p['global_best_lb']), LabelMajority=int(p['majority_lb']),
                    Oracle=min(LENGTHS, key=lambda lb: abs(fixed[lb]-truth)))
                for method in METHODS:
                    saved = pred[outer, sid, seed, method]
                    assert float(saved['soh_true']) == truth and saved['stage'] == row['stage']
                    near(float(saved['absolute_error']), abs(float(saved['soh_pred'])-truth))
                    if method in expected_lb:
                        assert int(saved['lookback']) == expected_lb[method]
                        near(float(saved['soh_pred']), fixed[expected_lb[method]])
                    elif method.startswith('Fixed'):
                        assert int(saved['lookback']) == int(method[5:])
                    else:
                        assert saved['lookback'] == ''
        for directory in [context_folder] + [model_root / f'seed_{s}' / f'lb_{lb}' for s in seeds for lb in LENGTHS]:
            complete = json.loads((directory / 'complete.json').read_text())
            assert complete['protocol_sha256'] == signature
            for name, digest in complete['files_sha256'].items():
                assert sha256(directory / name) == digest
    assert checked == run['predictor_models'] == len(batteries)*len(seeds)*3
    for seed, hashes in initial_states.items():
        assert len(hashes) == 1
        torch.manual_seed(seed)
        assert state_hash(CyclePredictor()) == next(iter(hashes))

    # Independently aggregate prediction errors and inspect complete timing grids.
    stats = {}
    for outer in batteries:
        for seed in seeds:
            for method in METHODS:
                rows = [r for (o, _, s, m), r in pred.items() if (o, s, m) == (outer, seed, method)]
                errors = np.array([float(r['soh_pred'])-float(r['soh_true']) for r in rows])
                stats[outer, seed, method] = (float(np.abs(errors).mean()), float(np.sqrt(np.mean(errors**2))))
    for r in read_csv(output / 'metrics_per_seed.csv'):
        near([float(r['mae']), float(r['rmse'])], stats[r['outer_test_battery'], int(r['seed']), r['method']])
    for r in read_csv(output / 'metrics_by_battery.csv'):
        values = np.array([stats[r['outer_test_battery'], seed, r['method']] for seed in seeds])
        near([float(r['mae_mean']), float(r['rmse_mean'])], values.mean(axis=0))
        near(float(r['mae_seed_sd']), values[:, 0].std(ddof=1) if len(seeds)>1 else 0.)
    for r in read_csv(output / 'metrics_by_stage.csv'):
        values = [float(p['absolute_error']) for p in predictions
                  if (p['outer_test_battery'], p['stage'], p['method']) == (r['outer_test_battery'], r['stage'], r['method'])]
        assert len(values) == int(r['samples_per_seed'])*len(seeds)
        near(float(r['mae_mean']), np.mean(values))
    for r in read_csv(output / 'paired_comparisons.csv'):
        differences = [stats[r['outer_test_battery'], s, 'Adaptive'][0]-stats[r['outer_test_battery'], s, r['baseline']][0] for s in seeds]
        near(float(r['adaptive_minus_baseline_mae']), np.mean(differences))
        near(float(r['paired_delta_seed_sd']), np.std(differences, ddof=1) if len(seeds)>1 else 0.)
    for r in read_csv(output / 'selection_diagnostics.csv'):
        outer, sid = r['outer_test_battery'], r['sample_id']
        errors = [np.mean([float(pred[outer, sid, s, f'Fixed{lb}']['absolute_error']) for s in seeds]) for lb in LENGTHS]
        best = LENGTHS[int(np.argmin(errors))]
        chosen = int(selection[sid]['selected_lookback'])
        assert int(r['posthoc_seed_mean_best_lb']) == best
        assert int(r['matches_posthoc_best']) == (best == chosen)
        near(float(r['excess_mae_vs_seed_mean_oracle']), errors[LENGTHS.index(chosen)]-min(errors))
    timings = read_csv(output / 'latency_samples.csv')
    timing_keys = set()
    for row in timings:
        key = (row['outer_test_battery'], row['sample_id'], int(row['seed']), row['method'])
        assert row['method'] != 'Oracle' and int(row['latency_ns']) > 0
        assert row['lookback'] == pred[key]['lookback']
        assert (*key, int(row['repetition'])) not in timing_keys
        timing_keys.add((*key, int(row['repetition'])))
    assert timing_keys == {(*key, i) for key in expected_keys if key[-1] != 'Oracle'
                           for i in range(protocol['benchmark']['rounds'])}
    assert len(timings) == run['timing_rows']
    for row in read_csv(output / 'summary.csv'):
        m = row['method']
        macro = np.array([[stats[o, s, m][0] for o in batteries] for s in seeds]).mean(axis=1)
        near(float(row['battery_macro_mae']), macro.mean())
        near(float(row['macro_mae_seed_sd']), macro.std(ddof=1) if len(seeds)>1 else 0.)
        near(float(row['battery_macro_rmse']), np.mean([stats[o, s, m][1] for o in batteries for s in seeds]))
        subset = [r for r in predictions if r['method'] == m]
        near(float(row['pooled_mae']), np.mean([float(r['absolute_error']) for r in subset]))
        pooled_rmse = [np.sqrt(np.mean([float(r['absolute_error'])**2 for r in subset if int(r['seed']) == s])) for s in seeds]
        near(float(row['pooled_rmse']), np.mean(pooled_rmse))
        if m != 'ContextOnlyRidge':
            near(float(row['mean_lookback']), np.mean([int(r['lookback']) for r in subset]))
        if m != 'Oracle':
            cost = [int(r['latency_ns'])/1e6 for r in timings if r['method'] == m]
            near([float(row['latency_median_ms']), float(row['latency_mean_ms']), float(row['latency_p95_ms'])],
                 [np.median(cost), np.mean(cost), np.percentile(cost, 95)])
    result = dict(status='PASS', protocol_sha256=signature, predictor_models=checked,
        selectors=len(batteries), unique_test_samples=len(samples), predictions=len(predictions), timings=len(timings),
        checks=['input/source/artifact hashes', 'battery-disjoint fits and training-only scalers',
                'convex selector and ridge first-order optimality', 'saved checkpoints reproduce all fixed predictions',
                'canonical context reproduces selector probabilities and ridge predictions',
                'adaptive routes to its selected fixed predictor', 'OOF-only global/majority choice',
                'posthoc oracle and diagnostics', 'independent metrics and complete latency grid'],
        limitations='Recorded timings cannot prove device isolation; four cells and seed SD do not establish population generalization.')
    write_json(output / 'verification.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    verify(args.data_root.resolve(), args.output.resolve())
