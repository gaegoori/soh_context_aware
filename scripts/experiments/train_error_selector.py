"""Freeze the existing predictor bank and compare an OOF-error ridge selector.

No hyperparameter search. Previously inspected four cells: exploratory evidence.
All four new selectors are fitted before evaluation; old results are read-only.
"""
from scripts.paths import PROJECT_ROOT, SCRIPTS, hash_matches, source_names
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
import torch

from scripts.models.adaptive_models import (ALL_METHODS as OLD_METHODS, CONTEXT_NAMES,
    CycleHistory, LOOKBACKS, context_at, fit_context_scaler, infer_one, transform_context)
from scripts.preprocessing.battery_data import DEFAULT_ROOT
from scripts.models.error_selector_models import fit_error_selector, infer_error_one, select_error_index
from scripts.experiments.train_adaptive_selector import align_selector_data, complete_matches, load_bank, load_final_split
from scripts.experiments.train_lookback_oof import read_csv, sha256, write_csv, write_json

PROJECT = PROJECT_ROOT
METHODS = tuple(OLD_METHODS[:-1]) + ('ErrorAware', 'Oracle')
DEPLOYABLE = METHODS[:-1]
DEFAULT_OUTPUT = PROJECT / 'outputs/error_selector_v1'
DEFAULT_BASELINE = PROJECT / 'outputs/adaptive_selector_v1'
L2 = 0.1


def now():
    return datetime.now(timezone.utc).isoformat()


def signature(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def assert_hashes(base, hashes):
    for name, digest in hashes.items():
        if not hash_matches(Path(base) / name, digest):
            raise ValueError(f'Changed artifact: {Path(base) / name}')


def check_baseline(root, baseline):
    p = json.loads((baseline / 'protocol.json').read_text())
    sig = signature(p)
    check = json.loads((baseline / 'verification.json').read_text())
    if check['status'] != 'PASS' or check['protocol_sha256'] != sig:
        raise ValueError('A verified original predictor/selector experiment is required')
    if Path(p['source_data_root']).resolve() != root:
        raise ValueError('Data root differs from the original experiment')
    labels = Path(p['source_labels_root'])
    for base, hashes in ((root, p['data_sha256']), (labels, p['labels_sha256']),
                         (SCRIPTS, p['code_sha256'])):
        assert_hashes(base, hashes)
    artifacts = json.loads((baseline / 'artifact_manifest.json').read_text())
    if artifacts['protocol_sha256'] != sig:
        raise ValueError('Baseline signature mismatch')
    assert_hashes(baseline, artifacts['files_sha256'])
    for outer in p['batteries']:
        if not complete_matches(baseline / 'models' / f'test_{outer}' / 'context_models', sig):
            raise ValueError('Missing baseline context models')
        for seed in p['seeds']:
            for length in LOOKBACKS:
                if not complete_matches(baseline / 'models' / f'test_{outer}' /
                                        f'seed_{seed}' / f'lb_{length}', sig):
                    raise ValueError('Missing original predictor')
    return p, sig, labels


def prepare_protocol(root, baseline):
    old, old_sig, labels = check_baseline(root, baseline)
    # Freeze all local artifacts actually consumed, including the original bank.
    inputs = set()
    for base, mapping in ((root, old['data_sha256']), (labels, old['labels_sha256'])):
        inputs.update(base / name for name in mapping)
    inputs.update(labels / name for name in ('predictions.csv', 'protocol.json', 'verification.json'))
    inputs.update(baseline / name for name in ('protocol.json', 'artifact_manifest.json',
                  'verification.json', 'predictions.csv', 'summary.csv'))
    for outer in old['batteries']:
        inputs.update((baseline / 'models' / f'test_{outer}').rglob('*.json'))
        inputs.update((baseline / 'models' / f'test_{outer}').rglob('*.npz'))
        inputs.update((baseline / 'models' / f'test_{outer}').rglob('*.pt'))
    source = SCRIPTS
    return dict(experiment='error_selector_v1', version=1, source_data_root=str(root),
        source_baseline_root=str(baseline), source_labels_root=str(labels),
        baseline_protocol_sha256=old_sig, batteries=old['batteries'], seeds=old['seeds'],
        lookbacks=LOOKBACKS.tolist(), features=CONTEXT_NAMES, methods=list(METHODS),
        selector=dict(model='three independent linear ridge error regressors, 15 -> 3',
            l2=L2, objective='per output: mean squared error + l2*||w||^2; unpenalized intercept',
            target='per sample/per lookback: mean absolute OOF prediction error over original five seeds',
            sample_weights='equal sample weights, same training rows as the original selector',
            target_transform='none; fractional SOH absolute errors',
            selection='argmin un-clipped error scores; exact ties choose shorter lookback',
            negative_estimates='possible for linear extrapolation; recorded, not clipped or calibrated'),
        predictor='reuse all 60 frozen original final checkpoints, no predictor training',
        preprocessing='same train-only context mean/std/imputation and signal scalers as original',
        tuning='none for this run; l2 fixed at 0.1 before fitting/evaluation',
        evidence='exploratory: original four outer cells have already informed method design',
        transfer_limit='OOF error teachers trained on 2 cells; final frozen predictors on 3 cells',
        primary_metric='battery-macro SOH MAE, paired across the same five predictor seeds',
        uncertainty='seed SD describes predictor initialization, not independent battery uncertainty',
        oracle='posthoc per sample/seed minimum; same three original predictions',
        benchmark=dict(rounds=20, warmup_rounds=3, threads=1, batch_size=1,
            shuffle_seed=20261027,
            scope='resampled past signals + causal metadata cache -> context if needed -> selector -> normalization -> one predictor',
            excludes='raw I/O, cycle segmentation/resampling, model loading, initial metadata cache',
            comparison='all deployable methods re-timed in this same run; no reuse of old timings'),
        input_sha256={str(p.resolve()): sha256(p) for p in sorted(inputs)},
        code_sha256={name: sha256(source / name) for name in source_names((
            'error_selector_models.py', 'train_error_selector.py', 'adaptive_models.py',
            'train_adaptive_selector.py', 'train_lookback_oof.py', 'battery_data.py', 'preprocess_main.py'))},
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                      torch=str(torch.__version__), platform=platform.platform()))


def check_inputs(protocol):
    assert_hashes(Path('/'), protocol['input_sha256'])
    assert_hashes(SCRIPTS, protocol['code_sha256'])


def fit_all(output, p):
    root, labels, baseline = (Path(p[k]) for k in
        ('source_data_root', 'source_labels_root', 'source_baseline_root'))
    samples = {r['sample_id']: r for r in read_csv(root / 'samples.csv')}
    for outer in p['batteries']:
        split, _ = load_final_split(root, outer, p['batteries'])
        rows = [samples[sid] for sid in split['train']]
        data = load_npz(labels / 'selector_data' / f'test_{outer}' / 'dataset.npz')
        raw, classes, errors, batteries = align_selector_data(data, rows, outer)
        histories = {b: CycleHistory.load(root, b) for b in split['train_batteries']}
        for row, stored in zip(rows, raw):
            actual = context_at(histories[row['battery']], int(row['history_end_exclusive']))
            np.testing.assert_allclose(actual, stored, rtol=1e-6, atol=1e-5, equal_nan=True)
        mean, std = fit_context_scaler(raw)
        x = transform_context(raw, mean, std)
        old = load_npz(baseline / 'models' / f'test_{outer}' / 'context_models' / 'parameters.npz')
        np.testing.assert_array_equal(mean, old['context_mean'])
        np.testing.assert_array_equal(std, old['context_std'])
        weight, bias = fit_error_selector(x, errors, p['selector']['l2'])
        residual = x @ weight + bias - errors
        folder = output / 'models' / f'test_{outer}'
        folder.mkdir(parents=True)
        np.savez(folder / 'parameters.npz', context_mean=mean, context_std=std,
                 error_weight=weight, error_bias=bias)
        write_json(folder / 'fit.json', dict(outer_test_battery=outer,
            train_batteries=split['train_batteries'], train_sample_ids=split['train'],
            feature_names=CONTEXT_NAMES, tuning_performed=False,
            target='OOF seed-mean absolute errors; not outer-test errors',
            training_mse_diagnostic=np.square(residual).mean(axis=0).tolist(),
            first_order_max_abs=float(np.abs(x.T @ residual / len(x) + L2 * weight).max()),
            negative_training_score_count=int((x @ weight + bias < 0).sum())))
        print(json.dumps(dict(event='error_selector_fitted', outer=outer, training_rows=len(rows))), flush=True)
    write_json(output / 'fitting_complete.json', dict(status='ALL_SELECTORS_FITTED',
        protocol_sha256=signature(p), completed_utc=now(), new_selectors=len(p['batteries']),
        new_predictors=0, model_sha256={str(f.relative_to(output)): sha256(f)
            for f in sorted((output / 'models').rglob('*')) if f.is_file()}))


def infer(method, history, n, bank, params):
    if method == 'ErrorAware':
        return infer_error_one(history, n, bank, params)
    return infer_one(history, n, bank, params, method)


def evaluate(output, p):
    if (output / 'run_summary.json').exists():
        raise FileExistsError('Evaluation already exists; preserve the saved experiment')
    check_inputs(p)
    fitted = json.loads((output / 'fitting_complete.json').read_text())
    if fitted['status'] != 'ALL_SELECTORS_FITTED' or fitted['protocol_sha256'] != signature(p):
        raise ValueError('Fit all selectors before evaluation')
    assert_hashes(output, fitted['model_sha256'])
    root, baseline = Path(p['source_data_root']), Path(p['source_baseline_root'])
    samples = {r['sample_id']: r for r in read_csv(root / 'samples.csv')}
    original = {(r['outer_test_battery'], int(r['seed']), r['sample_id'], r['method']): r
                for r in read_csv(baseline / 'predictions.csv')}
    predictions, selections, timings = [], [], []
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for outer in p['batteries']:
        split, _ = load_final_split(root, outer, p['batteries'])
        rows = [samples[sid] for sid in split['test']]
        params = load_npz(baseline / 'models' / f'test_{outer}' / 'context_models' / 'parameters.npz')
        params.update(load_npz(output / 'models' / f'test_{outer}' / 'parameters.npz'))
        history = CycleHistory.load(root, outer)
        for row in rows:
            context = transform_context(context_at(history, int(row['history_end_exclusive'])),
                                        params['context_mean'], params['context_std'])
            index, estimates = select_error_index(context, params['error_weight'], params['error_bias'])
            selections.append(dict(outer_test_battery=outer, sample_id=row['sample_id'],
                selected_lookback=int(LOOKBACKS[index]),
                **{f'estimated_error_lb{lb}': float(estimates[j]) for j, lb in enumerate(LOOKBACKS)},
                negative_score_count=int((estimates < 0).sum())))
        for seed in p['seeds']:
            bank = load_bank(baseline, outer, seed, p['baseline_protocol_sha256'])
            expected, fixed = {}, {}
            for method in DEPLOYABLE:
                for row in rows:
                    sid, n = row['sample_id'], int(row['history_end_exclusive'])
                    value, length = infer(method, history, n, bank, params)
                    if not np.isfinite(value):
                        raise ValueError('Nonfinite SOH estimate')
                    if method.startswith('Fixed'):
                        fixed[sid, length] = value
                    if method == 'ErrorAware':
                        np.testing.assert_allclose(value, fixed[sid, length], rtol=0, atol=1e-7)
                    else:
                        old = original[outer, seed, sid, method]
                        np.testing.assert_allclose(value, float(old['soh_pred']), rtol=0, atol=1e-7)
                        assert ('' if length is None else str(length)) == old['lookback']
                    expected[method, sid] = (value, length)
                    truth = float(row['soh'])
                    predictions.append(dict(outer_test_battery=outer, seed=seed, method=method,
                        sample_id=sid, stage=row['stage'], soh_true=truth, soh_pred=value,
                        absolute_error=abs(value-truth), lookback='' if length is None else length))
            for row in rows:
                sid, truth = row['sample_id'], float(row['soh'])
                length = min(LOOKBACKS.tolist(), key=lambda lb: abs(fixed[sid, lb]-truth))
                value = fixed[sid, length]
                np.testing.assert_allclose(value, float(original[outer, seed, sid, 'Oracle']['soh_pred']), rtol=0, atol=1e-7)
                predictions.append(dict(outer_test_battery=outer, seed=seed, method='Oracle',
                    sample_id=sid, stage=row['stage'], soh_true=truth, soh_pred=value,
                    absolute_error=abs(value-truth), lookback=length))
            requests = [(m, row) for m in DEPLOYABLE for row in rows]
            rng = np.random.default_rng(p['benchmark']['shuffle_seed'] + seed*100 + int(outer[2:]))
            for repeat in range(-p['benchmark']['warmup_rounds'], p['benchmark']['rounds']):
                for k in rng.permutation(len(requests)):
                    method, row = requests[k]
                    n = int(row['history_end_exclusive'])
                    start = time.perf_counter_ns()
                    value, length = infer(method, history, n, bank, params)
                    elapsed = time.perf_counter_ns() - start
                    want, want_length = expected[method, row['sample_id']]
                    if abs(value-want) > 1e-7 or length != want_length:
                        raise ValueError('Timed prediction changed')
                    if repeat >= 0:
                        timings.append(dict(outer_test_battery=outer, seed=seed, method=method,
                            sample_id=row['sample_id'], repetition=repeat,
                            lookback='' if length is None else length, latency_ns=elapsed))
            print(json.dumps(dict(event='evaluated', outer=outer, seed=seed, samples=len(rows))), flush=True)
    write_csv(output / 'predictions.csv', predictions)
    write_csv(output / 'selections.csv', selections)
    write_csv(output / 'latency_samples.csv', timings)
    summarize(output, p, predictions, selections, timings)
    check_inputs(p)
    write_json(output / 'run_summary.json', dict(status='EVALUATION_COMPLETE',
        protocol_sha256=signature(p), completed_utc=now(), unique_test_samples=len(samples),
        prediction_rows=len(predictions), timing_rows=len(timings), new_predictors=0,
        reused_predictors=60, new_selectors=4, current_run_test_tuning=False,
        evidence='exploratory; previously inspected four cells', methods=list(METHODS)))
    write_json(output / 'artifact_manifest.json', dict(protocol_sha256=signature(p),
        files_sha256={str(f.relative_to(output)): sha256(f) for f in sorted(output.rglob('*'))
                      if f.is_file() and f.name != 'artifact_manifest.json'}))


def summarize(output, p, predictions, selections, timings):
    groups, latency = defaultdict(list), defaultdict(list)
    for row in predictions:
        groups[row['outer_test_battery'], row['seed'], row['method']].append(row)
    for row in timings:
        latency[row['method']].append(row['latency_ns']/1e6)
    stats, seed_rows = {}, []
    for (outer, seed, method), rows in sorted(groups.items()):
        errors = np.array([r['absolute_error'] for r in rows])
        record = dict(outer_test_battery=outer, seed=seed, method=method, samples=len(rows),
                      mae=float(errors.mean()), rmse=float(np.sqrt(np.mean(errors**2))))
        stats[outer, seed, method] = record
        seed_rows.append(record)
    write_csv(output / 'metrics_per_seed.csv', seed_rows)
    battery_rows, summary, paired = [], [], []
    for method in METHODS:
        mae_matrix = np.array([[stats[b, s, method]['mae'] for b in p['batteries']] for s in p['seeds']])
        oracle_matrix = np.array([[stats[b, s, 'Oracle']['mae'] for b in p['batteries']] for s in p['seeds']])
        for j, b in enumerate(p['batteries']):
            lengths = [r['lookback'] for r in predictions if r['method']==method and
                       r['outer_test_battery']==b and r['lookback']!='']
            battery_rows.append(dict(outer_test_battery=b, method=method,
                samples_per_seed=stats[b, p['seeds'][0], method]['samples'],
                mae_mean=float(mae_matrix[:,j].mean()), mae_seed_sd=float(mae_matrix[:,j].std(ddof=1)),
                rmse_mean=float(np.mean([stats[b,s,method]['rmse'] for s in p['seeds']])),
                excess_mae_vs_oracle=float((mae_matrix[:,j]-oracle_matrix[:,j]).mean()),
                mean_lookback=float(np.mean(lengths)) if lengths else ''))
        lengths = [r['lookback'] for r in predictions if r['method']==method and r['lookback']!='']
        cost = latency.get(method, [])
        macro = mae_matrix.mean(axis=1)
        summary.append(dict(method=method, battery_macro_mae=float(macro.mean()),
            macro_mae_seed_sd=float(macro.std(ddof=1)),
            battery_macro_rmse=float(np.mean([stats[b,s,method]['rmse'] for b in p['batteries'] for s in p['seeds']])),
            excess_mae_vs_oracle=float((mae_matrix-oracle_matrix).mean()),
            mean_lookback=float(np.mean(lengths)) if lengths else '',
            latency_median_ms=float(np.median(cost)) if cost else '',
            latency_p95_ms=float(np.percentile(cost,95)) if cost else ''))
    for scope in [*p['batteries'], 'MACRO']:
        cells = p['batteries'] if scope=='MACRO' else [scope]
        for baseline in ('Adaptive','GlobalBestOOF','Fixed10','Fixed20','Fixed50','ContextOnlyRidge'):
            deltas = [np.mean([stats[b,s,'ErrorAware']['mae']-stats[b,s,baseline]['mae'] for b in cells])
                      for s in p['seeds']]
            paired.append(dict(scope=scope, baseline=baseline,
                error_aware_minus_baseline_mae=float(np.mean(deltas)),
                paired_delta_seed_sd=float(np.std(deltas,ddof=1))))
    write_csv(output / 'metrics_by_battery.csv', battery_rows)
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'paired_comparisons.csv', paired)
    pred = {(r['outer_test_battery'],r['seed'],r['sample_id'],r['method']):r for r in predictions}
    diagnostics=[]
    for row in selections:
        outer,sid = row['outer_test_battery'],row['sample_id']
        errors=np.array([np.mean([pred[outer,s,sid,f'Fixed{lb}']['absolute_error'] for s in p['seeds']]) for lb in LOOKBACKS])
        best=int(LOOKBACKS[np.argmin(errors)])
        old=int(pred[outer,p['seeds'][0],sid,'Adaptive']['lookback'])
        chosen=int(row['selected_lookback'])
        diagnostics.append(dict(outer_test_battery=outer,sample_id=sid,selected_lookback=chosen,
            old_selected_lookback=old,changed_from_old=int(old!=chosen),posthoc_seed_mean_best_lb=best,
            matches_posthoc_best=int(chosen==best),
            excess_mae_vs_seed_mean_oracle=float(errors[LOOKBACKS.tolist().index(chosen)]-min(errors)),
            **{f'actual_seed_mean_error_lb{lb}':float(errors[j]) for j,lb in enumerate(LOOKBACKS)}))
    write_csv(output / 'selection_diagnostics.csv',diagnostics)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--baseline',type=Path,default=DEFAULT_BASELINE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--phase',choices=('fit','evaluate','all'),default='all')
    args=parser.parse_args()
    output=args.output.resolve()
    if args.phase in ('fit','all'):
        if output.exists():
            raise FileExistsError('Use a fresh output directory; existing experiments are never overwritten')
        p=prepare_protocol(args.data_root.resolve(),args.baseline.resolve())
        output.mkdir(parents=True)
        write_json(output/'protocol.json',p)
        write_json(output/'protocol_frozen.json',dict(protocol_sha256=signature(p),frozen_utc=now()))
        fit_all(output,p)
    else:
        p=json.loads((output/'protocol.json').read_text())
    if args.phase in ('evaluate','all'):
        evaluate(output,p)


if __name__=='__main__':
    main()
