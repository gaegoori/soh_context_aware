"""Exploratory follow-up: context ablations and matched predictor input ladder.

Freeze protocol before fitting; finish every fit before scoring held-out cells.
These cells have already been inspected in a prior study, so this is NOT a new
confirmatory test. No test-driven tuning or checkpoint selection is performed.
"""
from scripts.paths import PROJECT_ROOT, SCRIPTS, source_names
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from scripts.models.adaptive_models import CycleHistory, fit_context_ridge
from scripts.preprocessing.battery_data import DEFAULT_ROOT
from scripts.models.context_history_models import (BATTERIES, LOOKBACKS, SEEDS, VARIANTS, CONTEXT_GROUPS,
    HistoryPredictor, train_bundle, input_arrays)
from scripts.preprocessing.preprocess_main import CONTEXT_NAMES
from scripts.experiments.train_adaptive_selector import begin_partial, commit_partial, complete_matches, utc_now
from scripts.experiments.train_lookback_oof import read_csv, write_csv, write_json, sha256, set_seed, state_hash

REPO = PROJECT_ROOT
DEFAULT_OUTPUT = REPO / 'outputs/context_history_v1'


def make_protocol(root, previous):
    files = [root / 'samples.csv', root / 'manifest.json', previous / 'predictions.csv', previous / 'summary.csv']
    for b in BATTERIES:
        files.extend([root / f'{b}.npz', root / f'{b}_cycles.csv'])
        files.extend(root / 'folds' / f'test_{b}__final' / n for n in ('split.json', 'scalers.json'))
    sources = ['context_history_models.py', 'train_context_history.py', 'adaptive_models.py',
               'train_adaptive_selector.py', 'train_lookback_oof.py', 'battery_data.py', 'preprocess_main.py']
    files.extend(SCRIPTS / n for n in source_names(sources))
    return dict(version=1, status='EXPLORATORY: same four cells inspected in previous adaptive study',
        batteries=BATTERIES, seeds=SEEDS, lookbacks=LOOKBACKS, variants=VARIANTS,
        context_groups=CONTEXT_GROUPS, context_names=CONTEXT_NAMES, epochs=200, learning_rate=.001,
        optimizer='Adam defaults; full-batch sample-weighted MSE; no early stopping',
        training_threads=4, device='cpu', deterministic=True, ridge_l2=.1,
        architecture='514->64->32 per cycle; mean32 + relative-position slope32 + context15; 79->32->1; ReLU',
        masks='Disabled metadata/trend/context replaced with zeros; same allocated parameters and seeded initialization',
        trend='sum(z_i*p_i)/sum(p_i^2), p=linspace(-1,1,L); simple linear order summary',
        metadata=['duration_s', 'discharge_Ah'],
        normalizers='Signal and metadata: union of training 50-cycle windows; context: training target endpoints only',
        targets='62 common measured reference SOH targets; no interpolated SOH or target-derived inputs',
        evaluation='3 training cells/1 held-out cell, rotate 4; battery-macro MAE; SD over 5 seeds, not cell CI',
        historical_reference='Previously computed Fixed10/20/50 only; different architecture from WaveMean',
        oracle='Posthoc minimum absolute error per sample/seed over three L; diagnostic only',
        cost='No new latency measurements or efficiency claims',
        selection='All settings fixed before new outcomes; no parameter search; all fits before evaluation',
        input_sha256={str(p.resolve()): sha256(p) for p in files},
        environment=dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__)))


def neural_folder(output, outer, variant, seed, lb):
    return output / 'models' / f'test_{outer}' / variant / f'seed_{seed}' / f'lb_{lb}'


def fit_all(root, output, signature):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    count = 0
    for outer in BATTERIES:
        split, rows, histories, scalers = train_bundle(root, outer)
        folder = output / 'models' / f'test_{outer}'
        write_json(folder / 'scalers.json', scalers)
        x = input_arrays(rows, histories, scalers, 50)
        y = torch.tensor([float(r['soh']) for r in rows], dtype=torch.float32)
        # Ridge keeps float64 context and targets, matching the previous baseline.
        from scripts.models.adaptive_models import context_at, transform_context
        raw_context = np.stack([context_at(histories[r['battery']], int(r['history_end_exclusive'])) for r in rows])
        cx = transform_context(raw_context, scalers['context_mean'], scalers['context_std'])
        target64 = np.array([float(r['soh']) for r in rows])
        for variant, features in CONTEXT_GROUPS.items():
            dest = folder / variant
            if complete_matches(dest, signature):
                continue
            weight, bias = fit_context_ridge(cx[:, features], target64, .1)
            partial = begin_partial(dest)
            np.savez(partial / 'parameters.npz', weight=weight, bias=bias, features=features)
            write_json(partial / 'fit.json', dict(train_sample_ids=split['train'], train_batteries=split['train_batteries'],
                training_mse=float(np.square(cx[:, features] @ weight + bias - target64).mean())))
            commit_partial(partial, dest, signature)
        for variant in VARIANTS:
            for seed in SEEDS:
                for lb in LOOKBACKS:
                    dest = neural_folder(output, outer, variant, seed, lb)
                    count += 1
                    if complete_matches(dest, signature):
                        continue
                    xx = (x[0][:, -lb:], x[1][:, -lb:], x[2])
                    set_seed(seed)
                    model = HistoryPredictor(variant)
                    initial_hash = state_hash(model)
                    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                    start, history = time.perf_counter(), []
                    for epoch in range(1, 201):
                        optimizer.zero_grad(set_to_none=True)
                        loss = torch.mean((model(*xx) - y).square())
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite loss')
                        loss.backward()
                        optimizer.step()
                        history.append(dict(epoch=epoch, training_mse_before_update=float(loss.detach())))
                    model.eval()
                    with torch.inference_mode():
                        final_mse = float((model(*xx) - y).square().mean())
                    seconds = time.perf_counter() - start
                    partial = begin_partial(dest)
                    torch.save(dict(state_dict=model.state_dict(), protocol_sha256=signature, variant=variant,
                        seed=seed, lookback=lb, outer_test_battery=outer), partial / 'checkpoint.pt')
                    write_csv(partial / 'training_history.csv', history)
                    write_json(partial / 'fit.json', dict(train_sample_ids=split['train'], train_batteries=split['train_batteries'],
                        initial_state_sha256=initial_hash, final_state_sha256=state_hash(model), epochs=200,
                        parameters=sum(p.numel() for p in model.parameters()), training_mse=final_mse, training_seconds=seconds))
                    commit_partial(partial, dest, signature)
                    print(f'FIT {count}/240 {outer} {variant} seed={seed} LB={lb} {seconds:.1f}s', flush=True)
    write_json(output / 'fitting_complete.json', dict(protocol_sha256=signature, neural_models=240,
        ridge_models=16, completed_utc=utc_now(), status='ALL_MODELS_FITTED'))


def prediction_rows(rows, prediction, method, family, seed, lb):
    return [dict(outer_test_battery=r['battery'], sample_id=r['sample_id'], stage=r['stage'],
        method=method, family=family, seed=seed, lookback=lb, soh_true=float(r['soh']), soh_pred=float(p),
        absolute_error=abs(float(p)-float(r['soh']))) for r, p in zip(rows, prediction)]


def evaluate(root, output, previous, signature):
    completion = json.loads((output / 'fitting_complete.json').read_text())
    if completion['protocol_sha256'] != signature or completion['status'] != 'ALL_MODELS_FITTED':
        raise ValueError('All models must finish fitting before evaluation')
    samples = {r['sample_id']: r for r in read_csv(root / 'samples.csv')}
    predictions = []
    from scripts.models.adaptive_models import context_at, transform_context
    for outer in BATTERIES:
        split = json.loads((root / 'folds' / f'test_{outer}__final' / 'split.json').read_text())
        rows = [samples[sid] for sid in split['test']]
        folder = output / 'models' / f'test_{outer}'
        scalers = json.loads((folder / 'scalers.json').read_text())
        histories = {outer: CycleHistory.load(root, outer)}
        x = input_arrays(rows, histories, scalers, 50)
        cx = transform_context(np.stack([context_at(histories[outer], int(r['history_end_exclusive'])) for r in rows]),
                               scalers['context_mean'], scalers['context_std'])
        for variant in CONTEXT_GROUPS:
            with np.load(folder / variant / 'parameters.npz') as p:
                pred = cx[:, p['features']] @ p['weight'] + p['bias']
            predictions += prediction_rows(rows, pred, variant, 'ridge', 'deterministic', '')
        for variant in VARIANTS:
            for seed in SEEDS:
                bank = []
                for lb in LOOKBACKS:
                    dest = neural_folder(output, outer, variant, seed, lb)
                    model = HistoryPredictor(variant)
                    checkpoint = torch.load(dest / 'checkpoint.pt', map_location='cpu', weights_only=True)
                    model.load_state_dict(checkpoint['state_dict'])
                    model.eval()
                    with torch.inference_mode():
                        pred = model(x[0][:, -lb:], x[1][:, -lb:], x[2]).numpy()
                    bank.append(pred)
                    predictions += prediction_rows(rows, pred, variant, 'neural', seed, lb)
                bank = np.stack(bank, axis=1)
                y = np.array([float(r['soh']) for r in rows])
                best = np.argmin(abs(bank-y[:, None]), axis=1)
                oracle = prediction_rows(rows, bank[np.arange(len(rows)), best], variant+'Oracle', 'oracle', seed, '')
                for row, index in zip(oracle, best):
                    row['lookback'] = LOOKBACKS[index]
                predictions.extend(oracle)
    old = read_csv(previous / 'predictions.csv')
    reference = {r['sample_id']: float(r['soh_pred']) for r in old if r['method'] == 'ContextOnlyRidge'}
    for r in predictions:
        if r['method'] == 'ContextAll':
            np.testing.assert_allclose(r['soh_pred'], reference[r['sample_id']], rtol=0, atol=1e-10)
    for r in old:
        if r['method'] in ('Fixed10', 'Fixed20', 'Fixed50'):
            predictions += prediction_rows([samples[r['sample_id']]], [float(r['soh_pred'])],
                'HistoricalFixed', 'historical', int(r['seed']), int(r['lookback']))
    write_csv(output / 'predictions.csv', predictions)
    summarize(output, predictions)
    write_json(output / 'run_summary.json', dict(status='COMPLETE', protocol_sha256=signature,
        completed_utc=utc_now(), unique_samples=len(samples), cells=len(BATTERIES), neural_models=240,
        ridge_models=16, prediction_rows=len(predictions), prior_context_ridge_reproduced=True))
    files = [p for p in output.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'
             and p.suffix in ('.json', '.csv', '.npz', '.pt') and 'figures' not in p.parts]
    write_json(output / 'artifact_manifest.json', dict(protocol_sha256=signature,
        files_sha256={str(p.relative_to(output)): sha256(p) for p in files}))


def summarize(output, rows):
    groups = defaultdict(list)
    for r in rows:
        lb = '' if r['family'] == 'oracle' else r['lookback']
        groups[(r['method'], r['family'], lb, str(r['seed']), r['outer_test_battery'])].append(r)
    cells = []
    for (method, family, lb, seed, battery), rr in groups.items():
        err = np.array([float(r['soh_pred'])-float(r['soh_true']) for r in rr])
        cells.append(dict(method=method, family=family, lookback=lb, seed=seed, battery=battery,
                          samples=len(rr), mae=float(abs(err).mean()), rmse=float(np.sqrt(np.square(err).mean()))))
    write_csv(output / 'metrics_by_battery_seed.csv', cells)
    macro = defaultdict(list)
    for r in cells:
        macro[(r['method'], r['family'], r['lookback'], r['seed'])].append(r)
    seed_rows = []
    for (method, family, lb, seed), rr in macro.items():
        if len(rr) != 4:
            raise ValueError('Incomplete 4-cell evaluation')
        seed_rows.append(dict(method=method, family=family, lookback=lb, seed=seed,
            macro_mae=float(np.mean([r['mae'] for r in rr])), macro_rmse=float(np.mean([r['rmse'] for r in rr]))))
    write_csv(output / 'metrics_per_seed.csv', seed_rows)
    summary_groups = defaultdict(list)
    for r in seed_rows:
        summary_groups[(r['method'], r['family'], r['lookback'])].append(r)
    summary = []
    for (method, family, lb), rr in summary_groups.items():
        values = [r['macro_mae'] for r in rr]
        summary.append(dict(method=method, family=family, lookback=lb, runs=len(rr),
            macro_mae=float(np.mean(values)), seed_sd=float(np.std(values, ddof=1)) if len(values)>1 else 0.,
            macro_rmse=float(np.mean([r['macro_rmse'] for r in rr]))))
    write_csv(output / 'summary.csv', summary)
    lookup = {(r['method'], str(r['lookback']), r['seed']): r['macro_mae'] for r in seed_rows}
    comparisons = []
    def compare(name, a, la, b, lb):
        values = []
        for seed in SEEDS:
            ka = (a, str(la), str(seed) if a not in CONTEXT_GROUPS else 'deterministic')
            kb = (b, str(lb), str(seed) if b not in CONTEXT_GROUPS else 'deterministic')
            values.append(lookup[ka]-lookup[kb])
        comparisons.append(dict(comparison=name, candidate=a, candidate_lb=la, reference=b, reference_lb=lb,
            mean_delta_mae=float(np.mean(values)), seed_sd_delta=float(np.std(values, ddof=1)),
            improved_seeds=int(sum(v < 0 for v in values)), total_seeds=5))
    variants = list(VARIANTS)
    for lb in LOOKBACKS:
        for a,b in zip(variants[1:], variants):
            compare('input_addition', a, lb, b, lb)
        for variant in VARIANTS:
            compare('vs_context_ridge', variant, lb, 'ContextAll', '')
    for variant in VARIANTS:
        for lb in (20,50):
            compare('longer_history', variant, lb, variant, 10)
    write_csv(output / 'paired_comparisons.csv', comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--previous', type=Path, default=REPO/'outputs/adaptive_selector_v1')
    args = parser.parse_args()
    protocol = json.loads(json.dumps(make_protocol(args.data_root, args.previous)))
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    path = args.output / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('Protocol changed; choose a new output directory')
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError('Nonempty output without protocol')
        write_json(path, protocol)
        write_json(args.output/'protocol_frozen.json', dict(protocol_sha256=signature, frozen_utc=utc_now()))
    print(f'EXPLORATORY protocol {signature}; 240 neural fits + 16 ridge fits', flush=True)
    fit_all(args.data_root, args.output, signature)
    evaluate(args.data_root, args.output, args.previous, signature)
    print('COMPLETE: all fits and evaluation saved', flush=True)


if __name__ == '__main__':
    main()
