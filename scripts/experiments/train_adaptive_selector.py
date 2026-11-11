"""Frozen-protocol outer LOBO evaluation of adaptive lookback selection.

This first experiment intentionally performs NO selector hyperparameter search.
Ordinary CV of existing OOF labels would not isolate the label-generating models.
All fitted objects are completed before any outer-test predictions are evaluated.
"""
from scripts.paths import PROJECT_ROOT, SCRIPTS, hash_matches, source_names
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import scipy
import torch

from scripts.models.adaptive_models import (ALL_METHODS, DEPLOYABLE_METHODS, LOOKBACKS, CONTEXT_NAMES,
    CycleHistory, context_at, fit_context_ridge, fit_context_scaler, fit_selector,
    infer_one, select_index, transform_context)
from scripts.preprocessing.battery_data import BatteryDataset, DEFAULT_ROOT
from scripts.experiments.train_lookback_oof import (CyclePredictor, DEFAULT_SEEDS, read_csv, sha256,
    state_hash, train_predictor, write_csv, write_json)

PROJECT = PROJECT_ROOT
DEFAULT_LABELS = PROJECT / 'outputs/context_labels_v1'
DEFAULT_OUTPUT = PROJECT / 'outputs/adaptive_selector_v1'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_final_split(root, outer, batteries):
    folder = root / 'folds' / f'test_{outer}__final'
    split = json.loads((folder / 'split.json').read_text())
    if (set(split['train_batteries']) != set(batteries) - {outer}
            or split['test_batteries'] != [outer] or split['validation_batteries']
            or split['validation'] or set(split['train']) & set(split['test'])):
        raise ValueError(f'Invalid outer split: {outer}')
    scalers = json.loads((folder / 'scalers.json').read_text())
    if set(scalers['fit_sample_ids']) != set(split['train']):
        raise ValueError('Signal normalization contains non-training sample IDs')
    if set(scalers['fit_cycle_indices']) != set(split['train_batteries']):
        raise ValueError('Signal normalization contains non-training batteries')
    return split, scalers


def align_selector_data(data, training_rows, outer):
    """Match by stable sample ID rather than assuming CSV/NPZ row ordering."""
    ids = list(map(str, data['sample_id']))
    expected = [r['sample_id'] for r in training_rows]
    if len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError('Selector rows must exactly match the outer-training IDs')
    if list(data['context_names']) != CONTEXT_NAMES:
        raise ValueError('Selector inputs must be the 15 approved context features')
    index = {sid: i for i, sid in enumerate(ids)}
    order = np.array([index[sid] for sid in expected])
    batteries = np.asarray(data['battery'])[order]
    if outer in batteries or list(batteries) != [r['battery'] for r in training_rows]:
        raise ValueError('Selector labels contain an outer-test cell or mismatched IDs')
    x, y = np.asarray(data['context_raw'])[order], np.asarray(data['class_label'])[order]
    errors = np.asarray(data['mean_errors'])[order]
    if x.shape != (len(expected), 15) or errors.shape != (len(expected), 3):
        raise ValueError('Invalid context/error dimensions')
    if not np.isfinite(errors).all() or not np.array_equal(y, np.argmin(errors, axis=1)):
        raise ValueError('Classes differ from the minimum mean-error labels')
    if not np.array_equal(LOOKBACKS[y], np.asarray(data['best_lookback'])[order]):
        raise ValueError('Class-to-lookback mapping differs')
    return x, y, errors, batteries


def make_protocol(args):
    root, labels = args.data_root, args.labels
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['context_names'] != CONTEXT_NAMES or manifest['lookbacks'] != LOOKBACKS.tolist():
        raise ValueError('Canonical feature or lookback definitions changed')
    old_run = json.loads((labels / 'run_summary.json').read_text())
    old_check = json.loads((labels / 'verification.json').read_text())
    if old_run['status'] != 'TRAINING_AND_LABELS_COMPLETE' or old_check['status'] != 'PASS':
        raise ValueError('Verified OOF labels are required')
    if old_run['seeds'] != list(DEFAULT_SEEDS):
        raise ValueError('This protocol uses labels averaged across the original five seeds')
    data_files = [root / name for name in ('manifest.json', 'samples.csv', 'audit.json')]
    label_files = [labels / name for name in ('protocol.json', 'run_summary.json',
                                             'verification.json', 'labels.csv')]
    for outer in manifest['batteries']:
        load_final_split(root, outer, manifest['batteries'])
        data_files += [root / f'{outer}.npz', root / f'{outer}_cycles.csv']
        folder = root / 'folds' / f'test_{outer}__final'
        data_files += [folder / 'split.json', folder / 'scalers.json']
        data_files += [folder / f'{b}.npz' for b in manifest['batteries']]
        label_files += [labels / 'selector_data' / f'test_{outer}' / name
                        for name in ('dataset.npz', 'schema.json')]
    source = SCRIPTS
    return dict(
        experiment='adaptive_selector_v1', version=1, batteries=manifest['batteries'],
        features=CONTEXT_NAMES, lookbacks=LOOKBACKS.tolist(), seeds=args.seeds,
        selector=dict(model='multinomial logistic regression, 15 inputs / 3 outputs',
                      objective='mean cross entropy + l2/2 * squared weights; unpenalized bias',
                      l2=args.selector_l2, initialization='all zeros',
                      optimizer='scipy L-BFGS-B; maxiter=500, ftol=1e-12, gtol=1e-8',
                      class_weights='none; equal sample weights',
                      selection='probability argmax; exact ties prefer shorter lookback'),
        context_only=dict(model='ridge regression', l2=args.ridge_l2,
                          objective='mean squared error + l2 * squared weights; unpenalized intercept'),
        preprocessing='Training-sample context means/std; mean imputation. No test fit. Signal scalers from final training fold.',
        predictor=dict(epochs=200, learning_rate=0.001, training_threads=args.train_threads,
                       architecture='existing CyclePredictor, independently refit per L on 3 training cells',
                       initialization='same seed -> identical starting weights for each L',
                       early_stopping=False),
        validation='No selector tuning or internal holdout score; fixed configuration before all outer tests. Existing OOF labels are not randomly split for validation.',
        outer_test='One never-fitted battery per fold; each of four cells tested once. No model, epoch, feature or threshold selection using these results.',
        global_best='argmin of OOF MAE averaged equally across the three training batteries; mean over label seeds',
        majority='most frequent OOF class in this training partition; exact ties choose shorter L',
        oracle='per-sample, per-predictor-seed test absolute-error minimum; diagnosis only, no latency claim',
        benchmark=dict(rounds=args.benchmark_rounds, warmup_rounds=args.benchmark_warmup,
                       threads=1, batch_size=1, shuffled_order_seed=20261012,
                       scope='in-memory resampled cycles and cached completed-cycle metadata -> context calculation if needed -> context/signal transform -> selection -> one predictor -> Python scalar',
                       excludes='file IO, raw segmentation/resampling, model loading and one-time metadata cache construction',
                       includes='recompute 15 context features at every Adaptive/ContextOnly request; no cached context vector; one selected predictor only'),
        methods=list(ALL_METHODS), primary_metric='battery-macro SOH MAE; paired per-seed results',
        uncertainty='seed SD is optimization variability, not uncertainty over independent batteries',
        source_data_root=str(root), source_labels_root=str(labels),
        data_sha256={str(p.relative_to(root)): sha256(p) for p in sorted(set(data_files))},
        labels_sha256={str(p.relative_to(labels)): sha256(p) for p in sorted(set(label_files))},
        code_sha256={name: sha256(source / name) for name in source_names(('adaptive_models.py',
                     'train_adaptive_selector.py', 'train_lookback_oof.py',
                     'battery_data.py', 'preprocess_main.py'))},
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__, torch=str(torch.__version__),
                      platform=platform.platform(), machine=platform.machine(), cpu_count=os.cpu_count()))


def complete_matches(folder, signature):
    path = folder / 'complete.json'
    if not path.exists():
        return False
    saved = json.loads(path.read_text())
    if saved['protocol_sha256'] != signature:
        raise ValueError(f'Changed protocol at {folder}')
    for name, digest in saved['files_sha256'].items():
        if not hash_matches(folder / name, digest):
            raise ValueError(f'Changed trained artifact: {folder / name}')
    return True


def begin_partial(folder):
    partial = folder.with_name(folder.name + '.partial')
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    return partial


def commit_partial(partial, folder, signature):
    write_json(partial / 'complete.json', dict(protocol_sha256=signature,
               files_sha256={p.name: sha256(p) for p in partial.iterdir() if p.is_file()}))
    if folder.exists():
        raise FileExistsError(f'Incomplete destination must be investigated: {folder}')
    partial.rename(folder)


def fit_all(args, protocol, signature):
    root, output = args.data_root, args.output
    samples = {r['sample_id']: r for r in read_csv(root / 'samples.csv')}
    torch.set_num_threads(args.train_threads)
    torch.use_deterministic_algorithms(True)
    progress, total = 0, len(protocol['batteries']) * len(args.seeds) * 3
    for outer in protocol['batteries']:
        split, scalers = load_final_split(root, outer, protocol['batteries'])
        training_rows = [samples[sid] for sid in split['train']]
        if any(row['battery'] == outer for row in training_rows):
            raise ValueError('Test row entered training')
        context_folder = output / 'models' / f'test_{outer}' / 'context_models'
        if not complete_matches(context_folder, signature):
            with np.load(args.labels / 'selector_data' / f'test_{outer}' / 'dataset.npz',
                         allow_pickle=False) as data:
                xraw, classes, oof_errors, batteries = align_selector_data(data, training_rows, outer)
            # Audit live feature extraction against canonical training contexts.
            histories = {b: CycleHistory.load(root, b) for b in split['train_batteries']}
            for row, raw in zip(training_rows, xraw):
                actual = context_at(histories[row['battery']], int(row['history_end_exclusive']))
                np.testing.assert_allclose(actual, raw, rtol=1e-6, atol=1e-5, equal_nan=True)
            mean, std = fit_context_scaler(xraw)
            x = transform_context(xraw, mean, std)
            weight, bias, optimization = fit_selector(x, classes, args.selector_l2)
            targets = np.array([float(row['soh']) for row in training_rows])
            ridge_weight, ridge_bias = fit_context_ridge(x, targets, args.ridge_l2)
            oof_macro_errors = np.mean([oof_errors[batteries == b].mean(axis=0)
                                       for b in split['train_batteries']], axis=0)
            counts = np.bincount(classes, minlength=3)
            global_best = int(LOOKBACKS[int(np.argmin(oof_macro_errors))])
            majority = int(LOOKBACKS[int(np.argmax(counts))])
            partial = begin_partial(context_folder)
            np.savez_compressed(partial / 'parameters.npz', context_mean=mean, context_std=std,
                selector_weight=weight, selector_bias=bias, ridge_weight=ridge_weight,
                ridge_bias=np.asarray(ridge_bias), global_best_lb=np.asarray(global_best),
                majority_lb=np.asarray(majority), channel_mean=np.array(scalers['channel_mean'], dtype=np.float32),
                channel_std=np.array(scalers['channel_std'], dtype=np.float32))
            write_json(partial / 'fit.json', dict(outer_test_battery=outer,
                train_batteries=split['train_batteries'], train_sample_ids=split['train'],
                feature_names=CONTEXT_NAMES, class_counts=counts.tolist(),
                context_missing_by_feature=np.isnan(xraw).sum(axis=0).tolist(),
                global_best_lb=global_best, majority_lb=majority,
                oof_macro_mae_by_lookback=oof_macro_errors.tolist(),
                selector_optimization=optimization, tuning_performed=False))
            commit_partial(partial, context_folder, signature)
            print(json.dumps(dict(event='selector_fitted', outer=outer, samples=len(training_rows),
                                  global_best_lb=global_best, class_counts=counts.tolist())), flush=True)
        ds = BatteryDataset(f'test_{outer}__final', 'train', lookback=50, root=root)
        items = [ds[i] for i in range(len(ds))]
        x50 = torch.from_numpy(np.stack([r['x'] for r in items]))
        y = torch.from_numpy(np.array([r['soh'] for r in items], dtype=np.float32))
        if [r['sample_id'] for r in items] != split['train']:
            raise ValueError('Training tensor alignment changed')
        for seed in args.seeds:
            for lb in LOOKBACKS.tolist():
                folder = output / 'models' / f'test_{outer}' / f'seed_{seed}' / f'lb_{lb}'
                reused = complete_matches(folder, signature)
                if not reused:
                    model, history, initial_hash, mse, seconds = train_predictor(
                        x50[:, -lb:].contiguous(), y, seed, 200, 0.001)
                    partial = begin_partial(folder)
                    torch.save(dict(state_dict=model.state_dict(), seed=seed, lookback=lb,
                        epochs=200, protocol_sha256=signature, outer_test_battery=outer,
                        train_batteries=split['train_batteries'], train_sample_ids=split['train'],
                        initial_state_sha256=initial_hash, scalers=scalers), partial / 'checkpoint.pt')
                    write_csv(partial / 'training_history.csv', history)
                    write_json(partial / 'fit.json', dict(final_train_mse=mse, training_seconds=seconds,
                        final_state_sha256=state_hash(model), initial_state_sha256=initial_hash))
                    commit_partial(partial, folder, signature)
                progress += 1
                print(json.dumps(dict(event='predictor_fitted', completed=progress, total=total,
                                      outer=outer, seed=seed, lookback=lb, reused=reused)), flush=True)
    write_json(output / 'fitting_complete.json', dict(status='ALL_MODELS_FITTED',
        protocol_sha256=signature, completed_utc=utc_now(), predictor_models=total,
        selectors=len(protocol['batteries']), context_only_models=len(protocol['batteries'])))


def load_bank(output, outer, seed, signature):
    bank = {}
    for lb in LOOKBACKS.tolist():
        folder = output / 'models' / f'test_{outer}' / f'seed_{seed}' / f'lb_{lb}'
        if not complete_matches(folder, signature):
            raise ValueError(f'Missing predictor {folder}')
        checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
        model = CyclePredictor()
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        bank[lb] = model
    return bank


def benchmark_requests(history, rows, bank, parameters, seed, outer,
                       rounds, warmup, expected_predictions):
    requests = [(method, i) for method in DEPLOYABLE_METHODS for i in range(len(rows))]
    rng = np.random.default_rng(20261012 + seed * 100 + int(outer[2:]))
    output = []
    for repetition in range(-warmup, rounds):
        order = rng.permutation(len(requests))
        for k in order:
            method, i = requests[k]
            row = rows[i]
            n = int(row['history_end_exclusive'])
            started = time.perf_counter_ns()
            pred, lb = infer_one(history, n, bank, parameters, method)
            elapsed = time.perf_counter_ns() - started
            # Validation and bookkeeping are outside the timed inference path.
            np.testing.assert_allclose(pred, expected_predictions[method, row['sample_id']], rtol=1e-6, atol=1e-7)
            if repetition >= 0:
                output.append(dict(outer_test_battery=outer, seed=seed, method=method,
                    sample_id=row['sample_id'], repetition=repetition,
                    lookback='' if lb is None else lb, latency_ns=elapsed))
    return output


def evaluate_all(args, protocol, signature):
    root, output = args.data_root, args.output
    if (output / 'artifact_manifest.json').exists():
        saved = json.loads((output / 'artifact_manifest.json').read_text())
        if saved['protocol_sha256'] != signature:
            raise ValueError('Completed evaluation has a different protocol')
        for name, digest in saved['files_sha256'].items():
            if not hash_matches(output / name, digest):
                raise ValueError(f'Completed evaluation artifact changed: {name}')
        print('Evaluation already complete; preserved existing predictions and timings.', flush=True)
        return
    fitted = json.loads((output / 'fitting_complete.json').read_text())
    if fitted['status'] != 'ALL_MODELS_FITTED' or fitted['protocol_sha256'] != signature:
        raise ValueError('All models must be fitted before outer evaluation')
    samples = {r['sample_id']: r for r in read_csv(root / 'samples.csv')}
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    predictions, selections, timings = [], [], []
    for outer in protocol['batteries']:
        split, _ = load_final_split(root, outer, protocol['batteries'])
        rows = [samples[sid] for sid in split['test']]
        if any(r['battery'] != outer for r in rows):
            raise ValueError('Evaluation does not match the excluded cell')
        folder = output / 'models' / f'test_{outer}' / 'context_models'
        if not complete_matches(folder, signature):
            raise ValueError('Missing fitted selector')
        with np.load(folder / 'parameters.npz', allow_pickle=False) as saved:
            parameters = {key: saved[key].copy() for key in saved.files}
        history = CycleHistory.load(root, outer)
        with np.load(root / f'{outer}.npz', allow_pickle=False) as canonical:
            for row in rows:
                n = int(row['history_end_exclusive'])
                raw = context_at(history, n)
                np.testing.assert_allclose(raw, canonical['context'][n-1], rtol=1e-6, atol=1e-5, equal_nan=True)
                context = transform_context(raw, parameters['context_mean'], parameters['context_std'])
                selected, probability = select_index(context, parameters['selector_weight'], parameters['selector_bias'])
                selections.append(dict(outer_test_battery=outer, sample_id=row['sample_id'],
                    selected_lookback=int(LOOKBACKS[selected]), **{f'p_lb{lb}': float(probability[j])
                    for j, lb in enumerate(LOOKBACKS.tolist())}))
        for seed in args.seeds:
            bank = load_bank(output, outer, seed, signature)
            expected, fixed = {}, {}
            for method in DEPLOYABLE_METHODS:
                for row in rows:
                    n = int(row['history_end_exclusive'])
                    pred, lb = infer_one(history, n, bank, parameters, method)
                    if not np.isfinite(pred):
                        raise ValueError('Nonfinite SOH prediction')
                    expected[method, row['sample_id']] = pred
                    if method.startswith('Fixed'):
                        fixed[row['sample_id'], lb] = pred
                    if method in ('Adaptive', 'GlobalBestOOF', 'LabelMajority'):
                        np.testing.assert_allclose(pred, fixed[row['sample_id'], lb], rtol=1e-6, atol=1e-7)
                    truth = float(row['soh'])
                    predictions.append(dict(outer_test_battery=outer, seed=seed, method=method,
                        sample_id=row['sample_id'], stage=row['stage'], soh_true=truth, soh_pred=pred,
                        absolute_error=abs(pred-truth), lookback='' if lb is None else lb))
            # SOH-dependent oracle is strictly post hoc; never used by infer_one or fitting.
            for row in rows:
                truth, sid = float(row['soh']), row['sample_id']
                lb = min(LOOKBACKS.tolist(), key=lambda length: abs(fixed[sid, length]-truth))
                pred = fixed[sid, lb]
                predictions.append(dict(outer_test_battery=outer, seed=seed, method='Oracle',
                    sample_id=sid, stage=row['stage'], soh_true=truth, soh_pred=pred,
                    absolute_error=abs(pred-truth), lookback=lb))
            timings.extend(benchmark_requests(history, rows, bank, parameters, seed, outer,
                args.benchmark_rounds, args.benchmark_warmup, expected))
            print(json.dumps(dict(event='outer_evaluated', outer=outer, seed=seed,
                                  samples=len(rows), timed_methods=len(DEPLOYABLE_METHODS))), flush=True)
    write_csv(output / 'predictions.csv', predictions)
    write_csv(output / 'selections.csv', selections)
    write_csv(output / 'latency_samples.csv', timings)
    summarize(output, predictions, selections, timings, protocol, signature)
    write_json(output / 'artifact_manifest.json', dict(protocol_sha256=signature,
        files_sha256={p.name: sha256(p) for p in output.iterdir()
                      if p.is_file() and p.name != 'verification.json'}))


def error_stats(rows):
    error = np.array([float(r['soh_pred']) - float(r['soh_true']) for r in rows])
    lengths = [float(r['lookback']) for r in rows if r['lookback'] != '']
    return dict(samples=len(rows), mae=float(np.abs(error).mean()),
                rmse=float(np.sqrt(np.square(error).mean())),
                mean_lookback=float(np.mean(lengths)) if lengths else '')


def mean_sd(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.


def summarize(output, predictions, selections, timings, protocol, signature):
    seeds, batteries = protocol['seeds'], protocol['batteries']
    groups, stage_groups, latency = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in predictions:
        groups[row['outer_test_battery'], int(row['seed']), row['method']].append(row)
        stage_groups[row['outer_test_battery'], row['stage'], int(row['seed']), row['method']].append(row)
    for row in timings:
        latency[row['outer_test_battery'], row['method']].append(float(row['latency_ns']) / 1e6)
    per_seed = [dict(outer_test_battery=o, seed=s, method=m, **error_stats(rows))
                for (o, s, m), rows in sorted(groups.items())]
    write_csv(output / 'metrics_per_seed.csv', per_seed)
    stats = {(r['outer_test_battery'], r['seed'], r['method']): r for r in per_seed}
    per_battery, per_stage = [], []
    for outer in batteries:
        for method in ALL_METHODS:
            values = [stats[outer, seed, method] for seed in seeds]
            mae, sd = mean_sd([v['mae'] for v in values])
            rmse, rsd = mean_sd([v['rmse'] for v in values])
            costs = latency.get((outer, method), [])
            per_battery.append(dict(outer_test_battery=outer, method=method,
                samples_per_seed=values[0]['samples'], seed_count=len(seeds), mae_mean=mae,
                mae_seed_sd=sd, rmse_mean=rmse, rmse_seed_sd=rsd,
                mean_lookback='' if method == 'ContextOnlyRidge' else float(np.mean([v['mean_lookback'] for v in values])),
                latency_median_ms=float(np.median(costs)) if costs else '',
                latency_mean_ms=float(np.mean(costs)) if costs else '',
                latency_p95_ms=float(np.percentile(costs, 95)) if costs else ''))
        for stage in ('early', 'middle', 'late'):
            for method in ALL_METHODS:
                rows = [error_stats(stage_groups[outer, stage, seed, method]) for seed in seeds
                        if stage_groups.get((outer, stage, seed, method))]
                if rows:
                    mae, sd = mean_sd([r['mae'] for r in rows])
                    per_stage.append(dict(outer_test_battery=outer, stage=stage, method=method,
                        samples_per_seed=rows[0]['samples'], mae_mean=mae, mae_seed_sd=sd))
    write_csv(output / 'metrics_by_battery.csv', per_battery)
    write_csv(output / 'metrics_by_stage.csv', per_stage)
    summary = []
    for method in ALL_METHODS:
        macro = [np.mean([stats[o, s, method]['mae'] for o in batteries]) for s in seeds]
        macro_rmse = [np.mean([stats[o, s, method]['rmse'] for o in batteries]) for s in seeds]
        pooled = [error_stats([r for r in predictions if r['method'] == method and int(r['seed']) == s])
                  for s in seeds]
        mean, sd = mean_sd(macro)
        lengths = [float(r['lookback']) for r in predictions if r['method'] == method and r['lookback'] != '']
        costs = [float(r['latency_ns']) / 1e6 for r in timings if r['method'] == method]
        summary.append(dict(method=method, battery_macro_mae=mean, macro_mae_seed_sd=sd,
            battery_macro_rmse=float(np.mean(macro_rmse)), pooled_mae=float(np.mean([p['mae'] for p in pooled])),
            pooled_rmse=float(np.mean([p['rmse'] for p in pooled])),
            mean_lookback=float(np.mean(lengths)) if lengths else '',
            latency_median_ms=float(np.median(costs)) if costs else '',
            latency_mean_ms=float(np.mean(costs)) if costs else '',
            latency_p95_ms=float(np.percentile(costs, 95)) if costs else ''))
    write_csv(output / 'summary.csv', summary)
    paired = []
    for outer in batteries:
        for baseline in ('Fixed10', 'Fixed20', 'Fixed50', 'GlobalBestOOF', 'LabelMajority', 'ContextOnlyRidge'):
            differences = [stats[outer, seed, 'Adaptive']['mae'] - stats[outer, seed, baseline]['mae'] for seed in seeds]
            delta, sd = mean_sd(differences)
            paired.append(dict(outer_test_battery=outer, baseline=baseline,
                adaptive_minus_baseline_mae=delta, paired_delta_seed_sd=sd))
    write_csv(output / 'paired_comparisons.csv', paired)
    # Post-hoc labels from the refitted test predictors diagnose OOF-to-refit transfer.
    by_selection, diagnostics = {}, []
    for row in selections:
        by_selection[row['outer_test_battery'], row['sample_id']] = row
    for outer in batteries:
        for (o, sid), row in by_selection.items():
            if o != outer:
                continue
            mean_errors = [np.mean([float(r['absolute_error']) for r in predictions
                if r['outer_test_battery'] == outer and r['sample_id'] == sid and r['method'] == f'Fixed{lb}'])
                for lb in LOOKBACKS]
            oracle_lb = int(LOOKBACKS[int(np.argmin(mean_errors))])
            selected_lb = int(row['selected_lookback'])
            diagnostics.append(dict(outer_test_battery=outer, sample_id=sid,
                selected_lookback=selected_lb, posthoc_seed_mean_best_lb=oracle_lb,
                matches_posthoc_best=int(selected_lb == oracle_lb),
                excess_mae_vs_seed_mean_oracle=float(mean_errors[LOOKBACKS.tolist().index(selected_lb)] - min(mean_errors))))
    write_csv(output / 'selection_diagnostics.csv', diagnostics)
    write_report(output, summary, per_battery, paired, selections, diagnostics, protocol)
    write_json(output / 'run_summary.json', dict(status='OUTER_EVALUATION_COMPLETE',
        protocol_sha256=signature, completed_utc=utc_now(), predictor_models=len(batteries)*len(seeds)*3,
        selectors=len(batteries), context_only_models=len(batteries),
        unique_test_samples=len(selections), prediction_rows=len(predictions), timing_rows=len(timings),
        seed_count=len(seeds), outer_cells=batteries, test_used_for_tuning=False,
        methods=list(ALL_METHODS)))


def write_report(output, summary, per_battery, paired, selections, diagnostics, protocol):
    by_method = {r['method']: r for r in summary}
    adaptive, fixed = by_method['Adaptive'], by_method['GlobalBestOOF']
    delta = adaptive['battery_macro_mae'] - fixed['battery_macro_mae']
    lines = ['# Adaptive Selector — 외부 배터리 평가', '',
        '## 고정한 실험 설계', '',
        '- 배터리 4개를 하나씩 외부 테스트로 제외. 각 셀은 한 번 테스트하며 SOH 정답은 총 62개다.',
        '- 학습 측 3개 셀의 기존 OOF Context–Lookback 자료로 Selector를 학습했다.',
        f"- Selector: 15 → 3 softmax 선형 분류, L2={protocol['selector']['l2']}. Context-only: ridge 회귀, L2={protocol['context_only']['l2']}. 튜닝 없이 고정한 첫 실험이다.",
        f"- Predictor: 외부 테스트를 제외한 3개 셀로 LB10/20/50을 각각 200 epochs 재학습; {len(protocol['seeds'])}개 seed 사용.",
        '- GlobalBestOOF는 학습 측 OOF 오차로 고른 고정 길이이며, 테스트 결과로 길이를 고르지 않았다.',
        '- 정확도 주 지표는 배터리 4개의 MAE를 동일 가중 평균한 값이다. SOH 0.01은 1%p다.', '',
        '## 전체 결과', '',
        '| 방법 | 배터리 평균 MAE | Seed SD | 전체 샘플 MAE | 평균 L | 추론 중앙값(ms) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        length = '해당 없음' if r['mean_lookback'] == '' else f"{r['mean_lookback']:.2f}"
        latency = '측정 대상 아님' if r['latency_median_ms'] == '' else f"{r['latency_median_ms']:.4f}"
        lines.append(f"| {r['method']} | {r['battery_macro_mae']:.6f} | {r['macro_mae_seed_sd']:.6f} | {r['pooled_mae']:.6f} | {length} | {latency} |")
    lines += ['', 'Oracle은 테스트 정답을 사용해 사후적으로 가장 작은 오차를 고른 참고값이다. 배포 가능한 성능이나 계산 비용이 아니다.', '',
        '## Adaptive 대 학습 측에서 고른 Fixed', '',
        f'- 배터리 평균 MAE 차이(Adaptive − GlobalBestOOF): **{delta:+.6f}** ({delta*100:+.3f}%p). 음수일 때 Adaptive 오차가 작다.',
        '| 테스트 셀 | Adaptive MAE | GlobalBestOOF MAE | 차이 | 선택 LB10 / LB20 / LB50 |',
        '|---|---:|---:|---:|---|']
    for outer in protocol['batteries']:
        rows = {r['method']: r for r in per_battery if r['outer_test_battery'] == outer}
        counts = Counter(int(r['selected_lookback']) for r in selections if r['outer_test_battery'] == outer)
        a, b = rows['Adaptive']['mae_mean'], rows['GlobalBestOOF']['mae_mean']
        lines.append(f"| {outer} | {a:.6f} | {b:.6f} | {a-b:+.6f} | {counts[10]} / {counts[20]} / {counts[50]} |")
    lines += ['', '## 비용 측정 범위', '',
        '- CPU 1 thread, batch=1. 방법·샘플 실행 순서를 섞어 워밍업 후 반복 측정했다.',
        '- Adaptive는 매 요청마다 Context 15개 재계산 → 정규화 → 길이 선택 → 선택한 신호 정규화 → Predictor 하나를 실행했다.',
        '- Context 계산에는 최근 10개 cycle 신호와 누적 운용 메타데이터를 사용한다. 최단 후보도 10이므로 필요한 신호 범위는 선택한 L을 넘지 않는다.',
        '- MAT 로딩, cycle 분할·리샘플링, 모델 로딩, 누적 Ah/시간 메타데이터의 최초 캐시 구축은 공통 입력 준비로 제외했다.',
        '- 따라서 원시 센서 입력부터의 전체 운영 지연이 아니라, 리샘플링된 cycle이 준비된 뒤의 온라인 추론 비용이다.',
        '- 한 기기의 짧은 CPU 측정이며 반복 간 변동이 있다. 평균 L 감소를 실제 속도 향상으로 대체해서 해석하지 않는다.', '',
        '## 해석 범위', '',
        '- 성능 개선 여부는 위 실제 오차 차이로 판단한다. Selector를 만들었다는 사실만으로 개선을 주장하지 않는다.',
        '- 독립 배터리는 4개뿐이다. 5개 seed는 학습 초기값 변화이며 독립 배터리 20개에 해당하지 않는다.',
        '- 기존 2개 셀 학습 Predictor가 만든 라벨과 3개 셀로 재학습한 Predictor의 최적 길이는 달라질 수 있다.',
        '- selection_diagnostics.csv의 정답은 재학습 Predictor의 테스트 오차에서 사후 생성한 진단값이며 학습/모델 선택에 쓰지 않았다.',
        '- RW3의 높은 온도 결측, SOH 기준 측정 지연, Early 샘플 부족, cycle 평균 pooling의 순서 정보 손실은 기존 제약이다.',
        '- 두 Context 모델의 학습 잔차/정확도는 학습 적합도일 뿐 내부 검증 성능으로 보고하지 않는다.',
        '- 이 결과를 보고 설정을 바꾸는 후속 연구는 탐색 실험이며, 확증에는 새로운 셀/독립 데이터가 필요하다.', '',
        '## 파일', '',
        '- protocol.json: 테스트 전에 고정한 설정과 데이터·코드 해시.',
        '- predictions.csv / selections.csv: 테스트 SOH 예측과 선택 확률.',
        '- summary.csv / metrics_by_battery.csv / metrics_by_stage.csv / paired_comparisons.csv: 집계와 비교.',
        '- latency_samples.csv: 요청별 원시 시간 기록.',
        '- models/: 학습에 사용한 셀·샘플 목록, 정규화 통계, 학습 이력과 모델.',
        '- verification.json: 독립 재계산 검증 결과(검증 스크립트 실행 후 생성).', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--labels', type=Path, default=DEFAULT_LABELS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--phase', choices=['fit', 'evaluate', 'all'], default='all')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument('--selector-l2', type=float, default=0.1)
    parser.add_argument('--ridge-l2', type=float, default=0.1)
    parser.add_argument('--train-threads', type=int, default=4)
    parser.add_argument('--benchmark-rounds', type=int, default=20)
    parser.add_argument('--benchmark-warmup', type=int, default=3)
    args = parser.parse_args()
    for name in ('data_root', 'labels', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds) or
        args.selector_l2 <= 0 or args.ridge_l2 <= 0 or args.benchmark_rounds < 1 or
        args.benchmark_warmup < 1 or args.train_threads < 1):
        parser.error('Use unique seeds, positive regularization, and positive thread/repetition counts')
    protocol = make_protocol(args)
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_file = args.output / 'protocol.json'
    if protocol_file.exists():
        if json.loads(protocol_file.read_text()) != protocol:
            raise ValueError('Protocol/data/code changed; use a new output directory')
    else:
        if any(args.output.iterdir()):
            raise ValueError('Refusing to start in a nonempty directory without a protocol')
        write_json(protocol_file, protocol)
        write_json(args.output / 'protocol_frozen.json', dict(protocol_sha256=signature,
            frozen_utc=utc_now(), statement='Configuration saved before model fitting and outer evaluation'))
    if args.phase in ('fit', 'all') and not (args.output / 'fitting_complete.json').exists():
        fit_all(args, protocol, signature)
    if args.phase in ('evaluate', 'all'):
        evaluate_all(args, protocol, signature)


if __name__ == '__main__':
    main()
