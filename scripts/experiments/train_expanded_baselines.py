"""Development-only LOBO Ridge baselines and leakage-safe inner OOF errors.

Uses eight development cells; the four reserved final-test payloads are never
loaded. All models are fitted before evaluating the development validation folds.
No Selector is fitted and no threshold or hyperparameter is optimized in this run.
"""
from scripts.paths import PROJECT_ROOT, SCRIPTS, resolve_file, hash_matches, source_names
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from scripts.models.adaptive_models import CycleHistory
from scripts.models.temporal_context_models import context_at_L,fit_ridge,predict_ridge,L2

ROOT=PROJECT_ROOT
LENGTHS=(10,20,50)
METHODS=('FixedContext10','FixedContext20','FixedContext50','GlobalBestOOF','Oracle')


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(resolve_file(path).read_bytes()).hexdigest()
def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def read(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def write_json(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
def write_csv(path,values):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
def load(path):
    with np.load(path,allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}
def group(cell):return 'uniform' if cell in ('RW3','RW4','RW5','RW6') else 'low_current' if cell in ('RW13','RW14') else 'high_current'


def check_inputs(protocol):
    for filename,digest in protocol['input_sha256'].items():
        if not hash_matches(filename, digest):raise ValueError('Input changed: '+filename)
    for filename,digest in protocol['code_sha256'].items():
        if not hash_matches(SCRIPTS/filename, digest):raise ValueError('Code changed: '+filename)


def make_protocol(root):
    split=json.loads((root/'split_plan.json').read_text())
    verification=json.loads((root/'verification.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    assert verification['status']=='PASS' and manifest['version']=='expanded_qc_v1'
    cells=split['development_batteries'];reserved=split['final_test_batteries']
    assert len(cells)==8 and len(reserved)==4 and not set(cells)&set(reserved)
    samples=read(root/'samples_development.csv')
    assert {r['battery'] for r in samples}==set(cells)
    inputs=[root/n for n in ('split_plan.json','manifest.json','quality_policy.json','verification.json','samples_development.csv')]
    inputs += [root/f'{cell}{suffix}' for cell in cells for suffix in ('.npz','_cycles.csv')]
    return dict(experiment='expanded_predictor_baseline_v1',source_data_root=str(root),
        development_batteries=cells,reserved_final_test_batteries=reserved,
        lookbacks=list(LENGTHS),methods=list(METHODS),primary_samples=len(samples),
        model=dict(kind='Context Ridge',dimensions=15,l2=L2,
            objective='mean squared SOH error + 0.1*||w||^2; unpenalized intercept',
            preprocessing='Training-target rows only: mean/std, mean imputation, inactive all-missing columns.',
            seeds='None: deterministic fits; no duplicated random-seed runs.'),
        features='Same 6 cumulative/last-cycle features + 9 statistics over the last L cycles; no SOH, stage or cell ID.',
        outer_validation='Eight development LOBO folds: seven fit cells and one validation cell.',
        inner_labels='Within each outer fitting set only: six Predictor fit cells and one OOF-label cell; seven rotations.',
        global_best='Choose the smallest equal-cell mean inner OOF absolute error; exact ties shorter L.',
        oracle='Posthoc minimum absolute error across the three final candidates on each development validation sample.',
        fitting_order='Fit all 168 inner and 24 outer candidate models before any outer validation predictions.',
        frozen_settings=True,selector_trained=False,threshold_tuned=False,final_test_evaluated=False,
        counts=dict(inner_models=168,outer_models=24),
        primary_metric='Mean of eight per-cell MAEs; RMSE also averaged over cells.',
        evidence='Development diagnostics, not final-test confirmation. Unequal sample counts; cells, not cycles or seeds, are the comparison units.',
        transfer_limit='Inner OOF teachers fit six cells; validation candidates fit seven. Store labels separately for each outer fold.',
        input_sha256={str(p.resolve()):sha(p) for p in inputs},
        code_sha256={name:sha(SCRIPTS/name) for name in source_names((
            'train_expanded_baselines.py','temporal_context_models.py','adaptive_models.py',
            'error_selector_models.py','preprocess_main.py'))})


def save_model(folder,model,info):
    folder.mkdir(parents=True)
    np.savez(folder/'parameters.npz',**model)
    write_json(folder/'fit.json',info)


def fit_all(output,p):
    root=Path(p['source_data_root']);samples=read(root/'samples_development.csv')
    cells=p['development_batteries'];lookup={r['sample_id']:r for r in samples}
    histories={cell:CycleHistory.load(root,cell) for cell in cells}
    features={}
    for r in samples:
        cell=r['battery'];n=int(r['history_end_exclusive'])
        feature=np.asarray([context_at_L(histories[cell],n,L) for L in LENGTHS])
        with np.load(root/f'{cell}.npz',allow_pickle=False) as data:
            assert n>=50 and data['cycle_valid'][n-50:n].all()
            np.testing.assert_array_equal(feature[0],data['context'][n-1])
        features[r['sample_id']]=feature
    np.savez(output/'features.npz',sample_id=np.array(list(features)),context_by_length=np.array(list(features.values())),lookbacks=np.array(LENGTHS))
    X=lambda ids,j:np.array([features[sid][j] for sid in ids])
    Y=lambda ids:np.array([float(lookup[sid]['soh']) for sid in ids])
    ids_for=lambda selected:[r['sample_id'] for r in samples if r['battery'] in selected]
    oof=[];labels=[];choices=[];model_count=0
    for outer in cells:
        fit_cells=[c for c in cells if c!=outer]
        fit_ids=ids_for(fit_cells);val_ids=ids_for([outer]);cell_errors=[]
        folder=output/'models'/f'validation_{outer}'
        write_json(output/f'split_validation_{outer}.json',dict(
            train_batteries=fit_cells,validation_batteries=[outer],
            train_sample_ids=fit_ids,validation_sample_ids=val_ids))
        for label_cell in fit_cells:
            train_cells=[c for c in fit_cells if c!=label_cell]
            train_ids=ids_for(train_cells);label_ids=ids_for([label_cell]);preds=[]
            for j,L in enumerate(LENGTHS):
                model=fit_ridge(X(train_ids,j),Y(train_ids))
                save_model(folder/f'oof_{label_cell}'/f'lb_{L}',model,dict(
                    role='inner_oof',validation_cell=outer,label_cell=label_cell,lookback=L,
                    train_batteries=train_cells,train_sample_ids=train_ids,label_sample_ids=label_ids))
                pred=predict_ridge(X(label_ids,j),model);preds.append(pred);model_count+=1
                for sid,value in zip(label_ids,pred):
                    truth=float(lookup[sid]['soh'])
                    oof.append(dict(validation_battery=outer,label_battery=label_cell,sample_id=sid,
                        lookback=L,soh_true=truth,soh_pred=float(value),absolute_error=float(abs(value-truth))))
            errors=np.abs(np.asarray(preds).T-Y(label_ids)[:,None]);cell_errors.append(errors.mean(axis=0))
            for sid,err in zip(label_ids,errors):
                labels.append(dict(validation_battery=outer,label_battery=label_cell,sample_id=sid,
                    best_lookback=LENGTHS[int(np.argmin(err))],error_gap_to_second_best=float(np.sort(err)[1]-min(err)),
                    **{f'error_lb{L}':float(err[j]) for j,L in enumerate(LENGTHS)}))
        means=np.mean(cell_errors,axis=0);best=LENGTHS[int(np.argmin(means))]
        choices.append(dict(validation_battery=outer,selected_lookback=best,
            **{f'inner_oof_mae_lb{L}':float(means[j]) for j,L in enumerate(LENGTHS)}))
        for j,L in enumerate(LENGTHS):
            model=fit_ridge(X(fit_ids,j),Y(fit_ids))
            save_model(folder/'candidates'/f'lb_{L}',model,dict(role='outer_candidate',validation_cell=outer,
                lookback=L,train_batteries=fit_cells,train_sample_ids=fit_ids,validation_sample_ids=val_ids))
            model_count+=1
        print(json.dumps(dict(event='fold_fitted',validation=outer,inner_selected_length=best,model_count=model_count)),flush=True)
    assert model_count==192
    write_csv(output/'oof_predictions.csv',oof);write_csv(output/'oof_labels.csv',labels)
    write_csv(output/'global_choices.csv',choices)
    write_json(output/'fitting_complete.json',dict(status='ALL_MODELS_FITTED',completed_utc=now(),
        protocol_sha256=signature(p),models=model_count,files_sha256={str(f.relative_to(output)):sha(f)
            for f in sorted(output.rglob('*')) if f.is_file() and f.name not in ('protocol.json','protocol_frozen.json')}))


def evaluate(output,p):
    check_inputs(p)
    complete=json.loads((output/'fitting_complete.json').read_text())
    assert complete['protocol_sha256']==signature(p) and complete['models']==192
    for name,digest in complete['files_sha256'].items():assert hash_matches(output/name, digest)
    root=Path(p['source_data_root']);samples=read(root/'samples_development.csv')
    data=load(output/'features.npz');features={str(sid):x for sid,x in zip(data['sample_id'],data['context_by_length'])}
    choices={r['validation_battery']:int(r['selected_lookback']) for r in read(output/'global_choices.csv')}
    predictions=[];diagnostics=[]
    for cell in p['development_batteries']:
        bank={L:load(output/'models'/f'validation_{cell}'/'candidates'/f'lb_{L}'/'parameters.npz') for L in LENGTHS}
        for r in [r for r in samples if r['battery']==cell]:
            sid=r['sample_id'];truth=float(r['soh'])
            values=np.array([float(predict_ridge(features[sid][j],bank[L])) for j,L in enumerate(LENGTHS)])
            errors=np.abs(values-truth);best=int(np.argmin(errors))
            diagnostics.append(dict(battery=cell,sample_id=sid,oracle_lookback=LENGTHS[best],
                best_second_error_gap=float(np.sort(errors)[1]-errors[best]),
                **{f'absolute_error_lb{L}':float(errors[j]) for j,L in enumerate(LENGTHS)}))
            for method in METHODS:
                L=int(method.replace('FixedContext','')) if method.startswith('FixedContext') else choices[cell] if method=='GlobalBestOOF' else LENGTHS[best]
                value=float(values[LENGTHS.index(L)])
                predictions.append(dict(battery=cell,condition=group(cell),sample_id=sid,method=method,
                    selected_lookback=L,soh_true=truth,soh_pred=value,absolute_error=abs(value-truth),
                    squared_error=(value-truth)**2,stage=r['stage']))
    write_csv(output/'predictions.csv',predictions);write_csv(output/'selection_potential.csv',diagnostics)
    metrics=[];summary=[];by_condition=[]
    for method in METHODS:
        mm=[]
        for cell in p['development_batteries']:
            subset=[r for r in predictions if r['method']==method and r['battery']==cell]
            row=dict(battery=cell,condition=group(cell),method=method,samples=len(subset),
                mae=float(np.mean([r['absolute_error'] for r in subset])),
                rmse=float(np.sqrt(np.mean([r['squared_error'] for r in subset]))))
            metrics.append(row);mm.append(row)
        summary.append(dict(method=method,battery_macro_mae=float(np.mean([r['mae'] for r in mm])),
            battery_macro_rmse=float(np.mean([r['rmse'] for r in mm])),development_cells=len(mm),samples=len(samples)))
        for condition in ('uniform','low_current','high_current'):
            sub=[r for r in mm if r['condition']==condition]
            by_condition.append(dict(condition=condition,method=method,cells=len(sub),
                battery_macro_mae=float(np.mean([r['mae'] for r in sub]))))
    write_csv(output/'metrics_by_battery.csv',metrics);write_csv(output/'summary.csv',summary)
    write_csv(output/'metrics_by_condition.csv',by_condition)
    gaps=[]
    for cell in p['development_batteries']:
        d={r['method']:r['mae'] for r in metrics if r['battery']==cell}
        posthoc=min(LENGTHS,key=lambda L:d[f'FixedContext{L}'])
        dd=[r for r in diagnostics if r['battery']==cell]
        gaps.append(dict(battery=cell,posthoc_best_fixed_lookback=posthoc,
            global_oof_lookback=choices[cell],global_oof_mae=d['GlobalBestOOF'],oracle_mae=d['Oracle'],
            global_to_oracle_gap=d['GlobalBestOOF']-d['Oracle'],
            posthoc_best_fixed_to_oracle_gap=d[f'FixedContext{posthoc}']-d['Oracle'],
            **{f'oracle_count_lb{L}':sum(r['oracle_lookback']==L for r in dd) for L in LENGTHS}))
    write_csv(output/'oracle_gap_by_battery.csv',gaps)
    result={r['method']:r['battery_macro_mae'] for r in summary}
    best_fixed=min(LENGTHS,key=lambda L:result[f'FixedContext{L}'])
    report=['# 확장 데이터의 개발용 Predictor 기준 실험','',
        '개발용 8개 배터리 안에서 한 셀씩 검증했습니다. 각 검증 후보는 나머지 7개 셀로 학습했습니다.',
        '고정 길이 선택은 그 7개 셀 내부의 OOF 오차로 결정했습니다. 최종 평가 셀은 불러오거나 예측하지 않았습니다.','',
        '| 방법 | 개발 셀 평균 SOH MAE (%p) |','|---|---:|']
    for row in summary:report.append(f"| {row['method']} | {row['battery_macro_mae']*100:.3f} |")
    report+=['',f'- 개발 검증 결과에서 평균 오차가 가장 작은 고정 길이는 LB{best_fixed}입니다. 이 순위는 개발 결과로 정한 것이며 독립 테스트 결론이 아닙니다.',
        f"- 해당 고정 길이와 Oracle의 차이는 {(result[f'FixedContext{best_fixed}']-result['Oracle'])*100:.3f}%p입니다.",
        f"- 학습 측 OOF로 고른 고정 길이와 Oracle의 차이는 {(result['GlobalBestOOF']-result['Oracle'])*100:.3f}%p입니다.",
        '- Oracle은 검증 정답을 알고 선택한 사후 진단입니다. 이 차이가 Selector로 학습 가능하다는 뜻은 아닙니다.',
        '- 이번에는 Predictor 기준선 및 후속 Selector용 내부 OOF 오차를 생성했습니다. Selector 재학습이나 threshold 적용 성능은 아직 측정하지 않았습니다.',
        '- Context는 15개이며, 길이에 따라 최근 통계 9개만 바뀝니다. 누적 사용 이력은 모든 후보에 들어갑니다.',
        '- 0~100°C 범위 및 입력 종료 후 최대 120시간의 측정 범위는 성능 확인 전에 고정했습니다.',
        '- 기존 4셀 실험과 셀 구성·학습 수·품질 처리 조건이 달라 수치를 직접 빼서 데이터 추가 효과라고 주장하지 않습니다.',
        '- 실제 측정 SOH 샘플은 155개이며, 같은 셀 내 샘플이나 내부 OOF 반복을 독립 표본으로 세지 않습니다.',
        '- 내부 라벨 생성 후보는 6개 셀, 검증 후보는 7개 셀로 학습하므로 재학습에 따른 후보 변화는 여전히 존재합니다.',
        '- 원인 분석 및 모델 선택을 위한 개발 결과입니다. 최종 평가용 4개 셀은 이후 설계를 고정한 뒤 사용합니다.','']
    (output/'FINDINGS.md').write_text('\n'.join(report))
    write_json(output/'run_summary.json',dict(status='DEVELOPMENT_EVALUATION_COMPLETE',completed_utc=now(),
        protocol_sha256=signature(p),models=192,development_samples=len(samples),
        predictions=len(predictions),inner_oof_predictions=len(read(output/'oof_predictions.csv')),
        selector_trained=False,final_test_payload_loaded=False,final_test_evaluated=False))
    write_json(output/'artifact_manifest.json',dict(protocol_sha256=signature(p),
        files_sha256={str(f.relative_to(output)):sha(f) for f in sorted(output.rglob('*')) if f.is_file()}))
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/processed/expanded_qc_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/expanded_predictor_baseline_v1')
    args=parser.parse_args();root=args.data_root.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError('Use a new output directory; completed runs are immutable')
    protocol=make_protocol(root);out.mkdir(parents=True)
    write_json(out/'protocol.json',protocol)
    write_json(out/'protocol_frozen.json',dict(frozen_utc=now(),protocol_sha256=signature(protocol)))
    check_inputs(protocol);fit_all(out,protocol);evaluate(out,protocol)


if __name__=='__main__':main()
