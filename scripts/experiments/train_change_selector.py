"""Predeclared 2x2 feature/model ablation on development cells only."""
from scripts.paths import PROJECT_ROOT, SCRIPTS, hash_matches, source_names
import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
import numpy as np
from scripts.models.adaptive_models import CycleHistory
from scripts.preprocessing.preprocess_main import CONTEXT_NAMES
from scripts.models.temporal_context_models import predict_ridge
from scripts.experiments.train_expanded_baselines import (now,sha,signature,read,write_json,write_csv,load,group,save_model,check_inputs)
from scripts.models.change_selector_models import (CHANGE_NAMES,RF_PARAMS,operating_changes,
    fit_regressor,predict_regressor,choose_length)

ROOT=PROJECT_ROOT
LENGTHS=(10,20,50)
SELECTORS=('Selector_Base_Linear','Selector_Change_Linear','Selector_Base_Forest','Selector_Change_Forest')
DIRECT=('Direct_Base_Linear','Direct_Change_Linear','Direct_Base_Forest','Direct_Change_Forest')
METHODS=('FixedContext10','FixedContext20','FixedContext50','GlobalBestOOF',*SELECTORS,*DIRECT,'Oracle')


def verified_manifest(root):
    assert json.loads((root/'verification.json').read_text())['status']=='PASS'
    manifest=json.loads((root/'artifact_manifest.json').read_text())
    inputs={}
    for name,digest in manifest['files_sha256'].items():
        assert hash_matches(root/name, digest),name
        inputs[str(root/name)]=digest
    for name in ('artifact_manifest.json','verification.json'):
        inputs[str(root/name)]=sha(root/name)
    return inputs


def make_protocol(baseline,previous):
    b=json.loads((baseline/'protocol.json').read_text());check_inputs(b)
    inputs={**b['input_sha256'],**verified_manifest(baseline),**verified_manifest(previous)}
    code_names=('change_selector_models.py','train_change_selector.py','verify_change_selector.py',
                'test_change_selector.py','temporal_context_models.py','adaptive_models.py',
                'error_selector_models.py','preprocess_main.py','train_expanded_baselines.py')
    return dict(experiment='change_selector_v1',source_baseline=str(baseline),source_previous_selector=str(previous),
        source_data_root=b['source_data_root'],development_batteries=b['development_batteries'],
        reserved_final_test_batteries=b['reserved_final_test_batteries'],
        baseline_protocol_sha256=signature(b),methods=list(METHODS),
        selector_methods=list(SELECTORS),direct_methods=list(DIRECT),
        input_sha256=inputs,code_sha256={n:sha(SCRIPTS/n) for n in source_names(code_names)},
        requirements_sha256=sha(ROOT/'requirements-selector-extra.txt'),
        runtime=dict(python=platform.python_version(),**{n:version(n) for n in ('numpy','scipy','scikit-learn','joblib','threadpoolctl')}),
        inputs=dict(base='C10: existing 15 Context features, including usage, moments and duration slope',
            augmented='Same 15 plus six signed recent-5 minus preceding-5 changes; total 21',
            added_feature_names=list(CHANGE_NAMES),
            missing='V/I/T group means use available resampled points; if either T block is all missing, change stays NaN. Missing-rate change is explicit.',
            past_only='Completed cycles [n-10:n-5] and [n-5:n]; no current/future SOH, stage, battery ID, target delay or condition label'),
        models=dict(linear='Mean MSE + 0.1 * squared weight norm; independent target columns; unpenalized intercept',
            forest=RF_PARAMS,forest_targets='One multioutput forest for 3 candidate errors; single-output forest for direct SOH',
            scaler='Fit only each model training rows, mean imputation, deactivate all-missing columns',
            randomness='Single predeclared RF seed 42; deterministic replay, not a seed-stability study',
            sample_weight='Equal training samples; evaluation equal cells'),
        selection='Argmin estimated OOF absolute error, exact ties shortest; no clipping; no threshold',
        splitting='Eight LOBO development validation folds; Selector trains on seven cells with six-cell-teacher OOF errors. Same frozen seven-cell candidates for all Selector variants.',
        direct_controls='Same seven training cells and same feature/model 2x2, predict SOH directly instead of selecting length.',
        tuning='No hyperparameter search or method choice using outer validation; all four variants reported.',
        fitting_order='Freeze specification; create target-free features; fit all 32 Selectors and 32 direct models; seal; then evaluate.',
        primary_metric='Equal-cell macro MAE on 155 measured targets; per-cell and per-condition results also reported.',
        planned_contrasts=['Change minus Base within Linear','Change minus Base within Forest',
                           'Forest minus Linear within Base','Forest minus Linear within Change',
                           'Combined changes minus Base Linear','Matching Selector minus direct SOH control'],
        limitations=['Prior development outcomes have been examined; exploratory development evidence only',
            'Same cell samples and LOBO folds are dependent; no significance claim',
            'Six-cell OOF teacher versus seven-cell final Predictor mismatch remains',
            'Lower mean lookback is not measured speed or reduced cumulative-history access',
            'All six change features added as one bundle; individual causal feature attribution not claimed'],
        final_test_evaluated=False,inference_cost_measured=False)


def check_inputs_frozen(p):
    for name,digest in p['input_sha256'].items():assert hash_matches(name, digest),name
    for name,digest in p['code_sha256'].items():assert hash_matches(SCRIPTS/name, digest),name
    assert hash_matches(ROOT/'requirements-selector-extra.txt', p['requirements_sha256'])


def load_data(p):
    baseline=Path(p['source_baseline']); root=Path(p['source_data_root'])
    samples=read(root/'samples_development.csv');lookup={r['sample_id']:r for r in samples}
    cached=load(baseline/'features.npz')
    features={str(s):v for s,v in zip(cached['sample_id'],cached['context_by_length'])}
    assert len(samples)==len(lookup)==len(features)==155
    assert {r['battery'] for r in samples}==set(p['development_batteries'])
    return samples,lookup,features


def prepare_features(out,p):
    samples,lookup,features=load_data(p);root=Path(p['source_data_root'])
    histories={c:CycleHistory.load(root,c) for c in p['development_batteries']}
    ids=[r['sample_id'] for r in samples]
    change=np.array([operating_changes(histories[r['battery']],int(r['history_end_exclusive'])) for r in samples])
    base=np.array([features[s][0] for s in ids]);aug=np.column_stack((base,change))
    assert base.shape==(155,15) and aug.shape==(155,21)
    np.savez(out/'features.npz',sample_id=np.array(ids),base=base,changes=change,augmented=aug)
    write_json(out/'feature_names.json',dict(base=list(CONTEXT_NAMES),added=list(CHANGE_NAMES)))
    write_csv(out/'change_features.csv',[dict(battery=lookup[s]['battery'],sample_id=s,
        **{n:float(v) if np.isfinite(v) else '' for n,v in zip(CHANGE_NAMES,ch)}) for s,ch in zip(ids,change)])


def feature_lookup(out):
    f=load(out/'features.npz')
    return {'Base':{str(s):x for s,x in zip(f['sample_id'],f['base'])},
            'Change':{str(s):x for s,x in zip(f['sample_id'],f['augmented'])}}


def fit_all(out,p):
    samples,lookup,_=load_data(p);features=feature_lookup(out)
    rows=read(Path(p['source_baseline'])/'oof_labels.csv')
    labels={(r['validation_battery'],r['sample_id']):r for r in rows}
    counts=dict(Selector=0,Direct=0)
    for outer in p['development_batteries']:
        train_cells=[c for c in p['development_batteries'] if c!=outer]
        ss=[r['sample_id'] for r in samples if r['battery'] in train_cells]
        errors=np.array([[float(labels[outer,s][f'error_lb{L}']) for L in LENGTHS] for s in ss])
        y=np.array([float(lookup[s]['soh']) for s in ss])
        for method in (*SELECTORS,*DIRECT):
            role,feature_set,kind=method.split('_');raw=np.array([features[feature_set][s] for s in ss])
            model=fit_regressor(raw,errors if role=='Selector' else y,kind)
            save_model(out/'models'/f'validation_{outer}'/method,model,dict(
                method=method,validation_battery=outer,train_batteries=train_cells,train_sample_ids=ss,
                feature_set=feature_set,kind=kind,role=role,
                target='OOF absolute errors, six-cell teachers' if role=='Selector' else 'Measured SOH on training cells',
                oof_validation_filter=outer if role=='Selector' else None))
            counts[role]+=1
        print(json.dumps(dict(fitted_validation_fold=outer,counts=counts)),flush=True)
    assert counts==dict(Selector=32,Direct=32)
    write_json(out/'fitting_complete.json',dict(protocol_sha256=signature(p),completed_utc=now(),counts=counts,
        files_sha256={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()}))


def evaluate(out,p):
    check_inputs_frozen(p)
    seal=json.loads((out/'fitting_complete.json').read_text())
    for name,digest in seal['files_sha256'].items():assert hash_matches(out/name, digest)
    samples,lookup,candidate_features=load_data(p);features=feature_lookup(out)
    baseline=Path(p['source_baseline']);cells=p['development_batteries']
    defaults={r['validation_battery']:int(r['selected_lookback']) for r in read(baseline/'global_choices.csv')}
    predictions=[];decisions=[]
    for outer in cells:
        ss=[r['sample_id'] for r in samples if r['battery']==outer]
        selector_outputs={};direct_outputs={}
        for method in (*SELECTORS,*DIRECT):
            role,feature_set,kind=method.split('_')
            model=load(out/'models'/f'validation_{outer}'/method/'parameters.npz')
            raw=np.array([features[feature_set][s] for s in ss])
            if role=='Selector':
                selected,scores=choose_length(raw,model);selector_outputs[method]=selected
                for i,sid in enumerate(ss):
                    decisions.append(dict(battery=outer,sample_id=sid,method=method,
                        selected_lookback=int(selected[i]),
                        **{f'estimated_error_lb{L}':float(scores[i,j]) for j,L in enumerate(LENGTHS)}))
            else:direct_outputs[method]=predict_regressor(raw,model)[:,0]
        bank=[load(baseline/'models'/f'validation_{outer}'/'candidates'/f'lb_{L}'/'parameters.npz') for L in LENGTHS]
        values=np.column_stack([predict_ridge(np.array([candidate_features[s][j] for s in ss]),m) for j,m in enumerate(bank)])
        truth=np.array([float(lookup[s]['soh']) for s in ss]);errors=np.abs(values-truth[:,None])
        default=LENGTHS.index(defaults[outer]);oracle=np.argmin(errors,axis=1)
        for method in METHODS:
            for i,sid in enumerate(ss):
                if method in DIRECT:pred=float(direct_outputs[method][i]);L=''
                else:
                    L=int(method.replace('FixedContext','')) if method.startswith('FixedContext') else defaults[outer] if method=='GlobalBestOOF' else LENGTHS[int(oracle[i])] if method=='Oracle' else int(selector_outputs[method][i])
                    pred=float(values[i,LENGTHS.index(L)])
                err=abs(pred-truth[i])
                predictions.append(dict(battery=outer,condition=group(outer),sample_id=sid,method=method,
                    selected_lookback=L,default_lookback=defaults[outer],soh_true=float(truth[i]),soh_pred=pred,
                    absolute_error=float(err),squared_error=float((pred-truth[i])**2),
                    improvement_over_default=float(errors[i,default]-err)))
    write_csv(out/'predictions.csv',predictions);write_csv(out/'decisions.csv',decisions)
    metrics=[];summary=[];conditions=[]
    for method in METHODS:
        for c in cells:
            rows=[r for r in predictions if r['method']==method and r['battery']==c]
            gain=np.array([r['improvement_over_default'] for r in rows])
            metrics.append(dict(battery=c,condition=group(c),method=method,samples=len(rows),
                mae=float(np.mean([r['absolute_error'] for r in rows])),
                rmse=float(np.sqrt(np.mean([r['squared_error'] for r in rows]))),
                mean_lookback='' if method in DIRECT else float(np.mean([r['selected_lookback'] for r in rows])),
                switches='' if method in DIRECT else sum(r['selected_lookback']!=r['default_lookback'] for r in rows),
                improved_samples=int((gain>1e-12).sum()),worsened_samples=int((gain < -1e-12).sum()),
                mean_improvement_over_default=float(gain.mean())))
        mm=[r for r in metrics if r['method']==method]
        summary.append(dict(method=method,battery_macro_mae=float(np.mean([r['mae'] for r in mm])),
            battery_macro_rmse=float(np.mean([r['rmse'] for r in mm])),
            mean_lookback='' if method in DIRECT else float(np.mean([r['mean_lookback'] for r in mm])),
            cells_improved=sum(r['mean_improvement_over_default']>1e-12 for r in mm),
            cells_worsened=sum(r['mean_improvement_over_default'] < -1e-12 for r in mm),
            improved_samples=sum(r['improved_samples'] for r in mm),worsened_samples=sum(r['worsened_samples'] for r in mm),
            switches='' if method in DIRECT else sum(r['switches'] for r in mm)))
        for cond in ('uniform','low_current','high_current'):
            sub=[r for r in mm if r['condition']==cond]
            conditions.append(dict(condition=cond,method=method,cells=len(sub),
                battery_macro_mae=float(np.mean([r['mae'] for r in sub]))))
    write_csv(out/'metrics_by_battery.csv',metrics);write_csv(out/'summary.csv',summary)
    write_csv(out/'metrics_by_condition.csv',conditions)
    pairs=[('Change_Linear','Base_Linear'),('Change_Forest','Base_Forest'),
           ('Base_Forest','Base_Linear'),('Change_Forest','Change_Linear'),('Change_Forest','Base_Linear')]
    contrasts=[]
    for a,b in [('Selector_'+a,'Selector_'+b) for a,b in pairs]+[(s,s.replace('Selector','Direct')) for s in SELECTORS]:
        for cell in cells:
            v={r['method']:r['mae'] for r in metrics if r['battery']==cell}
            contrasts.append(dict(battery=cell,comparison=f'{a} minus {b}',method_a=a,method_b=b,
                                  mae_a=v[a],mae_b=v[b],mae_difference=v[a]-v[b]))
    write_csv(out/'contrasts_by_battery.csv',contrasts)
    contrast_summary=[]
    for a,b in dict.fromkeys((r['method_a'],r['method_b']) for r in contrasts):
        values=[r['mae_difference'] for r in contrasts if r['method_a']==a and r['method_b']==b]
        contrast_summary.append(dict(method_a=a,method_b=b,macro_mae_difference=float(np.mean(values)),
            cells_a_better=sum(v < -1e-12 for v in values),cells_a_worse=sum(v>1e-12 for v in values)))
    write_csv(out/'contrasts_summary.csv',contrast_summary)
    report=['# 운용 변화 Context와 비선형 Selector 비교','',
        '개발용 8개 셀의 155개 측정 SOH를 배터리 단위로 분리해 평가했습니다. 기존 Predictor 후보는 그대로 유지했습니다.',
        '최종 평가용 RW15·16·19·20은 평가하지 않았습니다. 이전 결과를 본 뒤 설계한 탐색 연구입니다.','',
        '| 방법 | 셀 평균 SOH MAE (%p) | 고정 기준 대비 개선/악화 셀 수 |','|---|---:|---:|']
    for r in summary:report.append(f"| {r['method']} | {r['battery_macro_mae']*100:.4f} | {r['cells_improved']}/{r['cells_worsened']} |")
    report+=['','- Base는 기존 C10 15개, Change는 기존 15개 + 최근 5회와 이전 5회의 변화 6개입니다.',
        '- Linear는 기존 강도 0.1의 Ridge, Forest는 200개 트리·최대 깊이 3·최소 leaf 8·seed 42의 Random Forest입니다.',
        '- 네 Selector 모두 같은 OOF 절대오차를 학습하며, 예측 오차가 가장 작을 것으로 예상되는 길이를 선택합니다. Threshold는 사용하지 않았습니다.',
        '- Direct는 같은 정보와 모델로 SOH를 직접 예측합니다. 추가 정보 자체의 효과와 길이 선택의 기여를 구분하기 위한 기준선입니다.',
        '- 기본 Predictor의 예측값 및 기존 Base Linear Selector 재현 여부는 독립 검증에서 확인합니다.',
        '- 여섯 변화 특징을 한 묶음으로 추가했습니다. 개별 특징의 기여나 물리적 인과 관계를 이 비교만으로 단정하지 않습니다.',
        '- 새 모델 설정은 평가 전에 고정했으며, 결과를 보고 탐색하거나 수정하지 않았습니다. 단일 RF seed 실험이므로 seed 안정성은 확인하지 않았습니다.',
        '- 평균 입력 길이는 추론 비용 측정값이 아닙니다. 이번 실험은 지연 시간이나 최종 테스트 성능을 보고하지 않습니다.','']
    (out/'FINDINGS.md').write_text('\n'.join(report))
    write_json(out/'run_summary.json',dict(protocol_sha256=signature(p),completed_utc=now(),
        status='DEVELOPMENT_EVALUATION_COMPLETE',selector_models=32,direct_models=32,
        development_samples=155,prediction_rows=len(predictions),selector_decisions=len(decisions),
        final_test_payload_loaded=False,final_test_evaluated=False,inference_cost_measured=False))
    write_json(out/'artifact_manifest.json',dict(protocol_sha256=signature(p),
        files_sha256={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()}))
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=ROOT/'outputs/expanded_predictor_baseline_v1')
    parser.add_argument('--previous-selector',type=Path,default=ROOT/'outputs/expanded_selector_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/change_selector_v1')
    a=parser.parse_args();out=a.output.resolve()
    if out.exists():raise FileExistsError('Use a new output directory; preserve completed experiments')
    p=make_protocol(a.baseline.resolve(),a.previous_selector.resolve());out.mkdir(parents=True)
    write_json(out/'protocol.json',p)
    write_json(out/'protocol_frozen.json',dict(protocol_sha256=signature(p),frozen_utc=now()))
    check_inputs_frozen(p);prepare_features(out,p);fit_all(out,p);evaluate(out,p)


if __name__=='__main__':main()
