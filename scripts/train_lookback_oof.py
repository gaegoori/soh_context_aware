"""Train fixed-epoch cycle predictors and create outer-fold-specific OOF labels.

The validation battery supplies labels only AFTER training. No outer-test data is
loaded by the training/evaluation code, and no held-out score selects a checkpoint.
"""
import argparse
import csv
import hashlib
import json
import platform
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from battery_data import BatteryDataset, DEFAULT_ROOT

LOOKBACKS = (10, 20, 50)
DEFAULT_SEEDS = (42, 0, 1, 2, 3)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'outputs/context_labels_v1'


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f'Cannot write an empty table: {path}')
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CyclePredictor(nn.Module):
    """Shared encoder across cycles; independently fitted weights per lookback."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(1), nn.Linear(512, 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[2:]) != (128, 4):
            raise ValueError('Expected [batch, lookback, 128, 4]')
        batch, length = x.shape[:2]
        z = self.encoder(x.reshape(batch * length, 128, 4))
        return self.head(z.reshape(batch, length, 32).mean(dim=1)).squeeze(-1)


def train_predictor(x, y, seed, epochs, learning_rate):
    """Only training tensors enter this function; every optimizer step is fixed."""
    set_seed(seed)
    model = CyclePredictor()
    initial_hash = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x) - y) ** 2)
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite training loss at epoch {epoch}')
        loss.backward()
        optimizer.step()
        history.append(dict(epoch=epoch, train_mse_before_update=float(loss.detach())))
    model.eval()
    with torch.inference_mode():
        final_mse = float(torch.mean((model(x) - y) ** 2))
    return model, history, initial_hash, final_mse, time.perf_counter() - started


def get_fold(root, name, expected_batteries):
    split = json.loads((root / 'folds' / name / 'split.json').read_text())
    tr, va, te = [set(split[k + '_batteries']) for k in ('train', 'validation', 'test')]
    if (len(tr), len(va), len(te)) != (2, 1, 1):
        raise ValueError(f'{name}: expected 2 training, 1 labeling, 1 test battery')
    if tr & va or tr & te or va & te or tr | va | te != set(expected_batteries):
        raise ValueError(f'{name}: invalid battery split')
    ids = [set(split[k]) for k in ('train', 'validation', 'test')]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError(f'{name}: overlapping sample IDs')
    return split


def load_batch(root, fold, role):
    if role not in ('train', 'validation'):
        raise ValueError('This experiment never loads outer-test tensors')
    ds = BatteryDataset(fold, role, lookback=50, root=root)
    items = [ds[i] for i in range(len(ds))]
    if not items:
        raise ValueError(f'{fold}: empty {role} batch')
    x = torch.from_numpy(np.stack([item['x'] for item in items]))
    y = torch.from_numpy(np.array([item['soh'] for item in items], dtype=np.float32))
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Nonfinite normalized input or target')
    return x, y, ds.rows


def metrics(rows):
    y = np.array([float(r['soh_true']) for r in rows])
    p = np.array([float(r['soh_pred']) for r in rows])
    return dict(samples=len(rows), mae=float(np.abs(p-y).mean()),
                rmse=float(np.sqrt(np.mean((p-y)**2))),
                mape_percent=float(np.mean(np.abs((p-y)/y))*100))


def make_protocol(args, folds, manifest):
    root = args.data_root
    # Bind runs to data AND code. Seed subsets may be extended without refitting.
    files = [root / name for name in ('manifest.json', 'samples.csv', 'audit.json')]
    files += [root / f'{b}.npz' for b in manifest['batteries']]
    for name in folds:
        folder = root / 'folds' / name
        split = get_fold(root, name, manifest['batteries'])
        files += [folder / 'split.json', folder / 'scalers.json']
        files += [folder / f'{b}.npz' for b in split['train_batteries'] + split['validation_batteries']]
    return dict(version=1, experiment='cycle-level fixed-epoch nested OOF label generation',
                epochs=args.epochs, learning_rate=args.learning_rate, threads=args.threads,
                optimizer='Adam; default betas/epsilon; weight_decay=0',
                batch='full training batch; equal weight per reference sample',
                loss='MSE', lookbacks=list(LOOKBACKS), device='cpu', deterministic=True,
                architecture='Flatten(128x4)-Linear(512,64)-ReLU-Linear(64,32)-ReLU; mean over cycles; Linear(32,32)-ReLU-Linear(32,1)',
                heldout_use='predict once at fixed final epoch; no early stopping or checkpoint selection',
                target='measured reference SOH; no secondary interpolated labels',
                label='argmin of mean per-seed absolute error; exact ties choose shortest lookback',
                context='raw canonical 15 features; no mixing of inner-fold-normalized context',
                folds=folds, data_root=str(root),
                files_sha256={str(p.relative_to(root)): sha256(p) for p in sorted(set(files))},
                code_sha256={name: sha256(Path(__file__).with_name(name))
                             for name in ('train_lookback_oof.py', 'battery_data.py')},
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              torch=str(torch.__version__), machine=platform.machine()))


def run_one(args, fold, split, seed, lb, train, protocol_hash):
    folder = args.output / 'runs' / fold / f'seed_{seed}' / f'lb_{lb}'
    complete = folder / 'complete.json'
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved['protocol_sha256'] != protocol_hash:
            raise ValueError(f'Protocol mismatch: {folder}')
        for filename, digest in saved['files_sha256'].items():
            if sha256(folder / filename) != digest:
                raise ValueError(f'Changed artifact: {folder / filename}')
        return read_csv(folder / 'predictions.csv'), json.loads((folder / 'metrics.json').read_text()), True
    # A failed run is rebuilt in a dedicated partial folder, never mistaken for complete.
    partial = folder.with_name(folder.name + '.partial')
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    x50, y, train_rows = train
    x = x50[:, -lb:].contiguous()
    model, history, init_hash, train_mse, seconds = train_predictor(
        x, y, seed, args.epochs, args.learning_rate)
    # Held-out tensors are first loaded after the final optimizer step.
    vx50, _, heldout_rows = load_batch(args.data_root, fold, 'validation')
    with torch.inference_mode():
        prediction = model(vx50[:, -lb:].contiguous()).numpy()
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite held-out prediction')
    outer, heldout = split['test_batteries'][0], split['validation_batteries'][0]
    predictions = []
    for row, p in zip(heldout_rows, prediction):
        truth = float(row['soh'])
        predictions.append(dict(outer_test_battery=outer, heldout_battery=heldout, fold=fold,
                                sample_id=row['sample_id'], seed=seed, lookback=lb, stage=row['stage'],
                                soh_true=truth, soh_pred=float(p), absolute_error=abs(float(p)-truth)))
    result = dict(outer_test_battery=outer, heldout_battery=heldout, fold=fold, seed=seed,
                  lookback=lb, **metrics(predictions), epochs=args.epochs,
                  final_train_mse=train_mse, training_seconds=seconds,
                  initial_state_sha256=init_hash, final_state_sha256=state_hash(model))
    torch.save(dict(state_dict=model.state_dict(), protocol_sha256=protocol_hash, fold=fold,
                    seed=seed, lookback=lb, epochs=args.epochs,
                    train_sample_ids=[r['sample_id'] for r in train_rows],
                    train_batteries=split['train_batteries'], heldout_battery=heldout,
                    outer_test_battery=outer, initial_state_sha256=init_hash,
                    scalers=json.loads((args.data_root / 'folds' / fold / 'scalers.json').read_text())),
               partial / 'checkpoint.pt')
    write_csv(partial / 'training_history.csv', history)
    write_csv(partial / 'predictions.csv', predictions)
    write_json(partial / 'metrics.json', result)
    write_json(partial / 'complete.json', dict(protocol_sha256=protocol_hash,
               files_sha256={p.name: sha256(p) for p in partial.iterdir() if p.is_file()}))
    if folder.exists():
        raise FileExistsError(f'Incomplete destination exists; inspect before continuing: {folder}')
    partial.rename(folder)
    return predictions, result, False


def build_labels(predictions, seeds):
    groups = defaultdict(dict)
    for r in predictions:
        key = (r['outer_test_battery'], r['sample_id'])
        index = (int(r['seed']), int(r['lookback']))
        if index in groups[key]:
            raise ValueError(f'Duplicate prediction: {key}, {index}')
        groups[key][index] = r
    expected = {(seed, lb) for seed in seeds for lb in LOOKBACKS}
    labels, seed_labels = [], []
    for (outer, sid), entries in sorted(groups.items()):
        if set(entries) != expected:
            raise ValueError(f'Incomplete seed/lookback grid: {outer}, {sid}')
        metadata = next(iter(entries.values()))
        if metadata['heldout_battery'] == outer:
            raise ValueError('Outer-test samples cannot generate selector training labels')
        if any(r['heldout_battery'] != metadata['heldout_battery'] or
               r['stage'] != metadata['stage'] or float(r['soh_true']) != float(metadata['soh_true'])
               for r in entries.values()):
            raise ValueError('Mismatched sample metadata across lookbacks')
        errors = np.array([[abs(float(entries[s, lb]['soh_pred']) - float(entries[s, lb]['soh_true']))
                            for lb in LOOKBACKS] for s in seeds])
        if not np.isfinite(errors).all():
            raise ValueError('Nonfinite labeling error')
        means = errors.mean(axis=0)
        best_index = int(np.argmin(means))
        winners = np.argmin(errors, axis=1)
        sorted_errors = np.sort(means)
        labels.append(dict(outer_test_battery=outer, heldout_battery=metadata['heldout_battery'],
                           sample_id=sid, stage=metadata['stage'], soh_true=float(metadata['soh_true']),
                           seed_count=len(seeds), **{f'mean_error_lb{lb}': float(means[j]) for j, lb in enumerate(LOOKBACKS)},
                           best_lookback=LOOKBACKS[best_index], class_label=best_index,
                           error_gap_to_second_best=float(sorted_errors[1]-sorted_errors[0]),
                           seed_agreement=float(np.mean(winners == best_index)),
                           **{f'seed_wins_lb{lb}': int(np.sum(winners == j)) for j, lb in enumerate(LOOKBACKS)}))
        for seed, row_errors, winner in zip(seeds, errors, winners):
            seed_labels.append(dict(outer_test_battery=outer, heldout_battery=metadata['heldout_battery'],
                                    sample_id=sid, seed=seed,
                                    **{f'error_lb{lb}': float(row_errors[j]) for j, lb in enumerate(LOOKBACKS)},
                                    best_lookback=LOOKBACKS[int(winner)]))
    return labels, seed_labels


def summarize_metrics(predictions):
    per_seed = defaultdict(list)
    for r in predictions:
        # Overall and stage metrics are computed separately, keeping sample count visible.
        for stage in ('all', r['stage']):
            key = (r['outer_test_battery'], r['heldout_battery'], stage, int(r['lookback']), int(r['seed']))
            per_seed[key].append(r)
    groups = defaultdict(list)
    for key, rows in per_seed.items():
        groups[key[:-1]].append(metrics(rows))
    result = []
    for (outer, heldout, stage, lb), values in sorted(groups.items()):
        row = dict(outer_test_battery=outer, heldout_battery=heldout, stage=stage,
                   lookback=lb, samples_per_seed=values[0]['samples'], seed_count=len(values))
        for metric in ('mae', 'rmse', 'mape_percent'):
            measurements = np.array([v[metric] for v in values])
            row[f'{metric}_mean'] = float(measurements.mean())
            row[f'{metric}_std'] = float(measurements.std(ddof=1 if len(values)>1 else 0))
        result.append(row)
    return result


def export_outputs(args, predictions, model_metrics, manifest, protocol_hash):
    predictions.sort(key=lambda r: (r['outer_test_battery'], r['sample_id'], int(r['seed']), int(r['lookback'])))
    labels, seed_labels = build_labels(predictions, args.seeds)
    summary = summarize_metrics(predictions)
    write_csv(args.output / 'predictions.csv', predictions)
    write_csv(args.output / 'model_metrics.csv', model_metrics)
    write_csv(args.output / 'metrics_by_battery_stage.csv', summary)
    write_csv(args.output / 'labels.csv', labels)
    write_csv(args.output / 'labels_by_seed.csv', seed_labels)
    by_id = {r['sample_id']: r for r in read_csv(args.data_root / 'samples.csv')}
    contexts = {}
    for b in manifest['batteries']:
        with np.load(args.data_root / f'{b}.npz', allow_pickle=False) as data:
            contexts[b] = data['context']
    distributions, outer_metrics = [], []
    for outer in manifest['batteries']:
        rows = [r for r in labels if r['outer_test_battery'] == outer]
        expected = {sid for sid, r in by_id.items() if r['battery'] != outer}
        if {r['sample_id'] for r in rows} != expected or len(rows) != len(expected):
            raise ValueError(f'{outer}: labels must cover every outer-training sample exactly once')
        folder = args.output / 'selector_data' / f'test_{outer}'
        folder.mkdir(parents=True, exist_ok=True)
        raw_context = np.stack([contexts[r['heldout_battery']][int(by_id[r['sample_id']]['history_end_exclusive'])-1]
                                for r in rows])
        joined = []
        for r, features in zip(rows, raw_context):
            joined.append(dict(r, **{name: float(v) if np.isfinite(v) else ''
                                     for name, v in zip(manifest['context_names'], features)}))
        write_csv(folder / 'labels_with_context.csv', joined)
        np.savez_compressed(folder / 'dataset.npz', context_raw=raw_context,
                            context_observed_mask=np.isfinite(raw_context),
                            context_names=np.array(manifest['context_names']),
                            sample_id=np.array([r['sample_id'] for r in rows]),
                            battery=np.array([r['heldout_battery'] for r in rows]),
                            best_lookback=np.array([r['best_lookback'] for r in rows], dtype=np.int64),
                            class_label=np.array([r['class_label'] for r in rows], dtype=np.int64),
                            mean_errors=np.array([[r[f'mean_error_lb{lb}'] for lb in LOOKBACKS] for r in rows]),
                            seed_agreement=np.array([r['seed_agreement'] for r in rows]))
        write_json(folder / 'schema.json', dict(outer_test_battery=outer, samples=len(rows),
                   input_keys=['context_raw'], mask_key='context_observed_mask',
                   feature_names=manifest['context_names'], class_to_lookback=list(LOOKBACKS),
                   preprocessing='Fit imputation and scaling using selector training data only. NaNs retained.',
                   metadata_not_features=['sample_id', 'battery', 'best_lookback', 'class_label', 'mean_errors', 'seed_agreement'],
                   csv_metadata_not_features=[k for k in rows[0]],
                   warning='Do not concatenate outer-fold datasets for a common selector; each excludes a different test battery.'))
        for heldout in ['all'] + sorted(set(r['heldout_battery'] for r in rows)):
            for stage in ('all', 'early', 'middle', 'late'):
                selected = [r for r in rows if (heldout == 'all' or r['heldout_battery'] == heldout)
                            and (stage == 'all' or r['stage'] == stage)]
                counts = Counter(r['best_lookback'] for r in selected)
                distributions.append(dict(outer_test_battery=outer, heldout_battery=heldout, stage=stage,
                                          samples=len(selected), **{f'lb{lb}_count': counts[lb] for lb in LOOKBACKS},
                                          mean_seed_agreement=float(np.mean([r['seed_agreement'] for r in selected])) if selected else '',
                                          median_error_gap=float(np.median([r['error_gap_to_second_best'] for r in selected])) if selected else ''))
        # Equal battery weighting: these are inner OOF diagnostics, NOT outer-test scores.
        heldouts = sorted(set(r['heldout_battery'] for r in rows))
        mean_errors = [float(np.mean([np.mean([r[f'mean_error_lb{lb}'] for r in rows if r['heldout_battery']==b])
                                     for b in heldouts])) for lb in LOOKBACKS]
        oracle = float(np.mean([np.mean([min(r[f'mean_error_lb{lb}'] for lb in LOOKBACKS)
                                        for r in rows if r['heldout_battery']==b]) for b in heldouts]))
        winner = int(np.argmin(mean_errors))
        outer_metrics.append(dict(outer_test_battery=outer, samples=len(rows),
                                  **{f'macro_mae_lb{lb}': mean_errors[j] for j, lb in enumerate(LOOKBACKS)},
                                  selected_fixed_lookback=LOOKBACKS[winner],
                                  oracle_on_mean_seed_errors_macro_mae=oracle,
                                  oracle_gap_from_best_fixed=mean_errors[winner]-oracle,
                                  mean_seed_agreement=float(np.mean([r['seed_agreement'] for r in rows]))))
    write_csv(args.output / 'label_distribution.csv', distributions)
    write_csv(args.output / 'inner_oof_summary.csv', outer_metrics)
    write_json(args.output / 'run_summary.json', dict(status='TRAINING_AND_LABELS_COMPLETE',
               protocol_sha256=protocol_hash, seeds=args.seeds, epochs=args.epochs,
               models=len(model_metrics), prediction_rows=len(predictions), label_rows=len(labels),
               unique_reference_samples=len({r['sample_id'] for r in labels}),
               stage='inner OOF labeling only; outer-test evaluation has not been performed'))
    lines = ['# Cycle 단위 Predictor 학습 및 OOF Lookback 라벨', '',
             f'- 설정: 12 내부 folds × 3 lookbacks × {len(args.seeds)} seeds = {len(model_metrics)} models.',
             f'- Seeds: {args.seeds}; 고정 {args.epochs} epochs; Adam lr={args.learning_rate}; CPU.',
             '- 학습 2개 셀, 라벨 생성 1개 셀, 외부 테스트 1개 셀. 라벨 생성 셀로 early stopping하지 않는다.',
             '- 기존 cycle-level main_v1 데이터와 fold별 학습 전용 정규화 사용. 모델은 LB별 독립 학습.',
             '- 초기 가중치는 같은 seed의 LB10/20/50에 동일하다. MSE는 reference sample에 동일 가중치를 준다.',
             '- 외부 테스트 예측은 수행하지 않았다. 아래 수치는 내부 OOF 진단이며 최종 일반화 성능이 아니다.', '',
             '## 내부 OOF 성능 (배터리별 MAE를 동일 가중 평균)', '',
             '| 외부 테스트 제외 셀 | 라벨 수 | LB10 | LB20 | LB50 | 내부 선택 Fixed | Oracle* |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in outer_metrics:
        lines.append(f"| {row['outer_test_battery']} | {row['samples']} | {row['macro_mae_lb10']:.5f} | {row['macro_mae_lb20']:.5f} | {row['macro_mae_lb50']:.5f} | {row['selected_fixed_lookback']} | {row['oracle_on_mean_seed_errors_macro_mae']:.5f} |")
    lines += ['', '*Oracle: 샘플별 seed 평균 절대오차가 최소인 길이를 정답을 알고 선택한 사후 값. 학습 가능한 성능을 보장하지 않는다.',
              'SOH는 비율 단위이며 MAE 0.01은 SOH 1 percentage point에 해당한다.', '',
              '## 라벨 분포', '', '| 외부 테스트 제외 셀 | LB10 | LB20 | LB50 | 평균 seed 일치율 |',
              '|---|---:|---:|---:|---:|']
    for row in distributions:
        if row['heldout_battery']=='all' and row['stage']=='all':
            lines.append(f"| {row['outer_test_battery']} | {row['lb10_count']} | {row['lb20_count']} | {row['lb50_count']} | {row['mean_seed_agreement']:.1%} |")
    lines += ['', '## 해석 및 다음 단계', '',
              '- 186개 라벨 행은 독립 측정 186개가 아니다. 62개 기준 측정을 외부 fold별로 3회씩 사용했다.',
              '- seed 평균 절대오차가 최소인 후보를 라벨로 사용한다. 정확한 동률만 짧은 길이로 결정한다.',
              '- seed_agreement는 대표 라벨과 일치한 seed 비율이며 통계적 신뢰도나 selector 정확도가 아니다.',
              f'- 고정 {args.epochs} epochs는 사전에 정한 설정이며 이 결과로 epoch를 최적화하지 않았다.',
              '- 학습 후 관찰한 순위는 모델/학습/데이터 조건에 의존한다. 평균 pooling은 cycle 순서를 구별하지 않는다.',
              '- Stage는 사후 분석 메타데이터다. 고유 표본은 Early 6 / Middle 18 / Late 38로 Early 분석이 제한된다.',
              '- RW3 온도 결측과 소수 셀은 해석 제약이다. 온도 제외 검증과 Context-only baseline은 후속 실험이다.',
              '- Context 원본 15개와 결측 마스크를 보존했다. 정답 SOH, Stage, 오차, Battery ID는 selector feature에 넣지 않는다.',
              '- selector 하이퍼파라미터를 검증할 때에는 해당 검증 셀을 포함하지 않는 predictor/label 생성 절차도 고려해야 한다.',
              '- 외부 fold별 dataset.npz를 이용해 selector를 학습하고, 이후 학습 3개 셀로 predictor를 재학습해 외부 테스트를 평가한다.', '',
              '## 결과 파일', '',
              '- predictions.csv: sample/seed/lookback별 예측과 오차.',
              '- labels.csv / labels_by_seed.csv: seed 평균 대표 라벨 / seed별 라벨.',
              '- metrics_by_battery_stage.csv: 외부 fold/라벨 생성 셀/Stage/LB별 mean ± seed SD.',
              '- label_distribution.csv: 배터리 및 Stage별 라벨 분포와 일치율.',
              '- selector_data/test_RW*/dataset.npz: 외부 fold별 원본 Context와 라벨.',
              '- runs/: 모델 가중치, 학습 손실, 예측, 파일 해시. 중단 후 같은 명령으로 재개 가능.',
              '- verification.json: 별도 검증 스크립트를 통과한 경우 생성.', '']
    (args.output / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    args.data_root, args.output = args.data_root.resolve(), args.output.resolve()
    if (args.epochs < 1 or args.threads < 1 or not args.learning_rate > 0 or
            not np.isfinite(args.learning_rate) or len(args.seeds) != len(set(args.seeds)) or
            any(s < 0 or s >= 2**32 for s in args.seeds)):
        parser.error('epochs/threads/lr must be positive; seeds must be unique uint32 values')
    if args.output == args.data_root or args.data_root in args.output.parents:
        parser.error('output must be outside processed input data')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((args.data_root / 'manifest.json').read_text())
    batteries = manifest['batteries']
    if len(batteries) != 4 or len(set(batteries)) != 4:
        raise ValueError('This protocol requires four distinct batteries')
    folds = [f'test_{outer}__val_{heldout}' for outer in batteries for heldout in batteries if outer != heldout]
    protocol = make_protocol(args, folds, manifest)
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Existing output has a different protocol/data/code; choose a new output directory')
    else:
        write_json(protocol_path, protocol)
    # Expanding seeds invalidates the old aggregate verification until verified again.
    (args.output / 'verification.json').unlink(missing_ok=True)
    write_json(args.output / 'run_summary.json', dict(status='RUNNING', seeds=args.seeds,
               protocol_sha256=signature, expected_models=len(folds)*len(LOOKBACKS)*len(args.seeds)))
    models, predictions, completed = [], [], 0
    for seed in args.seeds:
        for fold in folds:
            split = get_fold(args.data_root, fold, batteries)
            train = load_batch(args.data_root, fold, 'train')
            for lb in LOOKBACKS:
                rows, result, reused = run_one(args, fold, split, seed, lb, train, signature)
                predictions.extend(rows)
                models.append(result)
                completed += 1
                print(json.dumps(dict(completed=completed, total=len(folds)*3*len(args.seeds),
                                      fold=fold, seed=seed, lookback=lb, reused=reused,
                                      heldout_mae=result['mae'], train_mse=result['final_train_mse'],
                                      training_seconds=result['training_seconds'])), flush=True)
    export_outputs(args, predictions, models, manifest, signature)
    print(f'DONE: {args.output}', flush=True)


if __name__ == '__main__':
    main()
