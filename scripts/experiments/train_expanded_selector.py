"""Frozen Predictor bank + nested development-only Selector comparison.

Never load final-test cell payloads. Gate calibration rebuilds OOF labels with
both outer validation and inner calibration cells excluded from every teacher.
"""
from scripts.paths import PROJECT_ROOT, SCRIPTS, hash_matches, source_names
import argparse
from itertools import combinations
from pathlib import Path
import json
import numpy as np

from scripts.experiments.train_expanded_baselines import (now, sha, signature, read, write_json,
    write_csv, load, group, save_model, check_inputs)
from scripts.models.temporal_context_models import fit_ridge, predict_ridge
from scripts.models.expanded_selector_models import (LENGTHS, THRESHOLDS, fit_selector,
    error_scores, classify, gate, select_threshold)

ROOT = PROJECT_ROOT
METHODS = ('FixedContext10', 'FixedContext20', 'FixedContext50', 'GlobalBestOOF',
           'Classifier', 'ErrorAware', 'GatedError', 'Oracle')


def make_protocol(source):
    previous = json.loads((source/'protocol.json').read_text())
    checked = json.loads((source/'verification.json').read_text())
    assert checked['status'] == 'PASS' and not checked['final_test_evaluated']
    check_inputs(previous)
    manifest = json.loads((source/'artifact_manifest.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        assert hash_matches(source/name, digest), name
    return dict(experiment='expanded_selector_v1', source_baseline=str(source),
        source_data_root=previous['source_data_root'],
        development_batteries=previous['development_batteries'],
        reserved_final_test_batteries=previous['reserved_final_test_batteries'],
        methods=list(METHODS), lookbacks=list(LENGTHS),
        baseline_protocol_sha256=signature(previous),
        inputs_sha256={**previous['input_sha256'],
            **{str(source/name):sha(source/name) for name in manifest['files_sha256']},
            **{str(source/name):sha(source/name) for name in ('artifact_manifest.json','verification.json')}},
        code_sha256={name:sha(SCRIPTS/name) for name in source_names((
            'expanded_selector_models.py','train_expanded_selector.py',
            'verify_expanded_selector.py','test_expanded_selector.py',
            'temporal_context_models.py','adaptive_models.py','error_selector_models.py',
            'train_expanded_baselines.py','preprocess_main.py'))},
        features='Existing C10: six cumulative/last-cycle features + nine recent-10 statistics. No new features in this comparison.',
        selectors=dict(classifier='Observed-class multinomial logistic, mean CE + 0.1/2 * ||W||^2',
            error_aware='Three Ridge error regressions: mean MSE + 0.1*||w||^2; no clipping',
            gated_error='Same error regressor; change from inner-OOF default only if predicted absolute-error reduction strictly exceeds threshold',
            standardization='Each Selector uses only its own fit rows; mean-impute and deactivate all-missing columns',
            fit_weights='Equal samples, deterministic; no seed repetitions'),
        thresholds_fraction_soh=list(THRESHOLDS),
        threshold_tie_rule='MAE within 1e-12 of minimum: fewer calibration switches, then larger threshold (never last)',
        primary_metric='Equal-cell macro MAE on eight held-out development cells',
        outer_protocol='Outer validation 1; seven training cells. Final Selector uses existing six-cell-teacher OOF labels; frozen seven-cell candidates.',
        threshold_protocol='Within each outer training seven: hold one calibration cell; generate fresh labels on other six using five-cell teachers. Fit error Selector on six; score calibration cell using frozen six-cell candidates. Pool seven cell-level calibration losses to choose threshold.',
        default_protocol='Default length always chosen using equal-cell mean OOF errors of the Selector fit cells; no hard-coded LB50.',
        candidate_transfer='Five-to-six cells during calibration; six-to-seven for outer validation. Predictor refit mismatch remains.',
        fitting_order='Freeze all settings; fit all nested teachers/Selectors and thresholds; then evaluate all outer validation folds.',
        context_limit='L controls recent statistics; cumulative usage features include earlier history. Mean L is not measured inference cost.',
        evidence='Development baseline outcomes were already seen. This is exploratory development validation, not an untouched final-test claim.',
        final_test_evaluated=False, inference_time_measured=False)


def check_frozen(p):
    for path, digest in p['inputs_sha256'].items():
        assert hash_matches(path, digest), path
    for name, digest in p['code_sha256'].items():
        assert hash_matches(SCRIPTS/name, digest), name


def data(p):
    source = Path(p['source_baseline'])
    samples = read(Path(p['source_data_root'])/'samples_development.csv')
    f = load(source/'features.npz')
    feature = {str(sid):v for sid,v in zip(f['sample_id'],f['context_by_length'])}
    lookup = {r['sample_id']:r for r in samples}
    assert set(lookup) == set(feature) and len(lookup) == 155
    assert {r['battery'] for r in samples} == set(p['development_batteries'])
    return samples, feature, lookup


def teacher_key(cells):
    return '_'.join(cells)


def fit_all(out, p):
    source = Path(p['source_baseline']); cells = p['development_batteries']
    samples, feature, lookup = data(p)
    ids = lambda cs:[r['sample_id'] for r in samples if r['battery'] in cs]
    X = lambda ss,j:np.array([feature[s][j] for s in ss])
    Y = lambda ss:np.array([float(lookup[s]['soh']) for s in ss])
    teachers = {}
    for train_cells in combinations(cells, 5):
        key = teacher_key(train_cells); train_ids = ids(train_cells); teachers[key] = []
        for j,L in enumerate(LENGTHS):
            m = fit_ridge(X(train_ids,j),Y(train_ids)); teachers[key].append(m)
            save_model(out/'teachers'/key/f'lb_{L}',m,dict(
                train_batteries=list(train_cells),train_sample_ids=train_ids,lookback=L))
    assert len(teachers) == 56
    print('Fitted 168 five-cell teachers for leakage-safe threshold calibration.',flush=True)
    base_labels = read(source/'oof_labels.csv')
    base_oof = {(r['validation_battery'],r['sample_id'],int(r['lookback'])):float(r['soh_pred'])
                for r in read(source/'oof_predictions.csv')}
    nested_labels=[]; calibration=[]; threshold_scores=[]; choices=[]
    for outer in cells:
        fit_cells=[c for c in cells if c!=outer]
        fold_cal=[]
        for cal in fit_cells:
            selector_cells=[c for c in fit_cells if c!=cal]; by_id={}; teacher_refs={}
            for label_cell in selector_cells:
                train_cells=[c for c in selector_cells if c!=label_cell]
                key=teacher_key(train_cells); teacher_refs[label_cell]=key
                label_ids=ids([label_cell])
                values=np.column_stack([predict_ridge(X(label_ids,j),m)
                                        for j,m in enumerate(teachers[key])])
                errors=np.abs(values-Y(label_ids)[:,None])
                for sid,err in zip(label_ids,errors):
                    by_id[sid]=err
                    nested_labels.append(dict(validation_battery=outer,calibration_battery=cal,
                        label_battery=label_cell,sample_id=sid,teacher_fit_key=key,
                        **{f'error_lb{L}':float(err[j]) for j,L in enumerate(LENGTHS)}))
            fit_ids=ids(selector_cells); errors=np.array([by_id[s] for s in fit_ids])
            model,stats=fit_selector(X(fit_ids,0),errors,[lookup[s]['battery'] for s in fit_ids])
            save_model(out/'selectors'/f'validation_{outer}'/f'calibration_{cal}',model,dict(
                validation_battery=outer,calibration_battery=cal,train_batteries=selector_cells,
                train_sample_ids=fit_ids,teacher_references=teacher_refs,**stats))
            cal_ids=ids([cal]); scores=error_scores(X(cal_ids,0),model)
            default=int(model['default_index'])
            true_errors=np.array([[abs(base_oof[outer,s,L]-float(lookup[s]['soh']))
                                   for L in LENGTHS] for s in cal_ids])
            for sid,estimated,err in zip(cal_ids,scores,true_errors):
                row=dict(validation_battery=outer,calibration_battery=cal,sample_id=sid,
                    default_lookback=LENGTHS[default],
                    **{f'estimated_error_lb{L}':float(estimated[j]) for j,L in enumerate(LENGTHS)},
                    **{f'actual_error_lb{L}':float(err[j]) for j,L in enumerate(LENGTHS)})
                calibration.append(row);fold_cal.append(row)
        threshold_rows=[]
        for ti,threshold in enumerate(THRESHOLDS):
            maes=[]; switches=0
            for cal in fit_cells:
                rr=[r for r in fold_cal if r['calibration_battery']==cal]
                scores=np.array([[r[f'estimated_error_lb{L}'] for L in LENGTHS] for r in rr])
                errors=np.array([[r[f'actual_error_lb{L}'] for L in LENGTHS] for r in rr])
                default=LENGTHS.index(rr[0]['default_lookback'])
                selected,_=gate(scores,default,threshold)
                maes.append(float(errors[np.arange(len(rr)),selected].mean()))
                switches+=int((selected!=default).sum())
            row=dict(validation_battery=outer,threshold_index=ti,
                threshold_fraction='never' if threshold is None else threshold,
                calibration_macro_mae=float(np.mean(maes)),calibration_switches=switches)
            threshold_scores.append(row);threshold_rows.append(row)
        ti=select_threshold([r['calibration_macro_mae'] for r in threshold_rows],
                            [r['calibration_switches'] for r in threshold_rows])
        rows={r['sample_id']:r for r in base_labels if r['validation_battery']==outer}
        fit_ids=ids(fit_cells)
        errors=np.array([[float(rows[s][f'error_lb{L}']) for L in LENGTHS] for s in fit_ids])
        model,stats=fit_selector(X(fit_ids,0),errors,[lookup[s]['battery'] for s in fit_ids],classifier=True)
        save_model(out/'selectors'/f'validation_{outer}'/'final',model,dict(
            validation_battery=outer,train_batteries=fit_cells,train_sample_ids=fit_ids,
            labels_source=str(source/'oof_labels.csv'),labels_validation_filter=outer,
            threshold_index=ti,threshold_fraction=THRESHOLDS[ti],**stats))
        choices.append(dict(validation_battery=outer,default_lookback=LENGTHS[int(model['default_index'])],
            threshold_index=ti,threshold_fraction='never' if THRESHOLDS[ti] is None else THRESHOLDS[ti]))
        print(json.dumps(dict(fitted_outer_selector=outer,threshold=THRESHOLDS[ti],
                              default=LENGTHS[int(model['default_index'])])),flush=True)
    write_csv(out/'nested_oof_labels.csv',nested_labels)
    write_csv(out/'calibration_predictions.csv',calibration)
    write_csv(out/'threshold_validation.csv',threshold_scores)
    write_csv(out/'choices.csv',choices)
    write_json(out/'fitting_complete.json',dict(completed_utc=now(),protocol_sha256=signature(p),
        new_predictor_teachers=168,nested_error_selectors=56,outer_error_selectors=8,outer_classifiers=8,
        files_sha256={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()}))


def evaluate(out,p):
    check_frozen(p)
    fitted=json.loads((out/'fitting_complete.json').read_text())
    for name,digest in fitted['files_sha256'].items():assert hash_matches(out/name, digest)
    cells=p['development_batteries']; source=Path(p['source_baseline'])
    samples,feature,lookup=data(p)
    choices={r['validation_battery']:r for r in read(out/'choices.csv')}
    predictions=[];decisions=[]
    for cell in cells:
        ss=[r['sample_id'] for r in samples if r['battery']==cell]
        model=load(out/'selectors'/f'validation_{cell}'/'final'/'parameters.npz')
        x=np.array([feature[s][0] for s in ss]); scores=error_scores(x,model)
        default=int(model['default_index']);threshold=THRESHOLDS[int(choices[cell]['threshold_index'])]
        indices={'Classifier':classify(x,model),'ErrorAware':np.argmin(scores,axis=1)}
        indices['GatedError'],gain=gate(scores,default,threshold)
        # All deployed choices above depend only on Context/model parameters.
        bank=[load(source/'models'/f'validation_{cell}'/'candidates'/f'lb_{L}'/'parameters.npz') for L in LENGTHS]
        values=np.column_stack([predict_ridge(np.array([feature[s][j] for s in ss]),m) for j,m in enumerate(bank)])
        truth=np.array([float(lookup[s]['soh']) for s in ss]); errors=np.abs(values-truth[:,None])
        oracle=np.argmin(errors,axis=1)
        for i,sid in enumerate(ss):
            decisions.append(dict(battery=cell,sample_id=sid,default_lookback=LENGTHS[default],
                threshold_index=int(choices[cell]['threshold_index']),predicted_gain=float(gain[i]),
                oracle_lookback=LENGTHS[int(oracle[i])],
                **{f'{m}_lookback':LENGTHS[int(v[i])] for m,v in indices.items()},
                **{f'estimated_error_lb{L}':float(scores[i,j]) for j,L in enumerate(LENGTHS)}))
            for method in METHODS:
                j=LENGTHS.index(int(method.replace('FixedContext',''))) if method.startswith('FixedContext') else default if method=='GlobalBestOOF' else int(oracle[i]) if method=='Oracle' else int(indices[method][i])
                pred=float(values[i,j]); err=float(errors[i,j])
                predictions.append(dict(battery=cell,condition=group(cell),sample_id=sid,method=method,
                    selected_lookback=LENGTHS[j],default_lookback=LENGTHS[default],soh_true=float(truth[i]),
                    soh_pred=pred,absolute_error=err,squared_error=(pred-truth[i])**2,
                    improvement_over_default=float(errors[i,default]-err),
                    selection_regret=float(err-errors[i,oracle[i]])))
    write_csv(out/'predictions.csv',predictions);write_csv(out/'decisions.csv',decisions)
    metrics=[];summary=[];by_condition=[]
    for method in METHODS:
        for cell in cells:
            sub=[r for r in predictions if r['method']==method and r['battery']==cell]
            metrics.append(dict(battery=cell,condition=group(cell),method=method,samples=len(sub),
                mae=float(np.mean([r['absolute_error'] for r in sub])),
                rmse=float(np.sqrt(np.mean([r['squared_error'] for r in sub]))),
                mean_lookback=float(np.mean([r['selected_lookback'] for r in sub])),
                switches=sum(r['selected_lookback']!=r['default_lookback'] for r in sub),
                helpful_switches=sum(r['improvement_over_default']>1e-12 for r in sub),
                harmful_switches=sum(r['improvement_over_default'] < -1e-12 for r in sub),
                mean_regret=float(np.mean([r['selection_regret'] for r in sub])),
                mean_improvement_over_default=float(np.mean([r['improvement_over_default'] for r in sub]))))
        mm=[r for r in metrics if r['method']==method]
        summary.append(dict(method=method,battery_macro_mae=float(np.mean([r['mae'] for r in mm])),
            battery_macro_rmse=float(np.mean([r['rmse'] for r in mm])),
            battery_macro_mean_lookback=float(np.mean([r['mean_lookback'] for r in mm])),
            battery_macro_improvement_over_default=float(np.mean([r['mean_improvement_over_default'] for r in mm])),
            cells_better_than_default=sum(r['mean_improvement_over_default']>1e-12 for r in mm),
            cells_worse_than_default=sum(r['mean_improvement_over_default'] < -1e-12 for r in mm),
            switches=sum(r['switches'] for r in mm),helpful_switches=sum(r['helpful_switches'] for r in mm),
            harmful_switches=sum(r['harmful_switches'] for r in mm)))
        for condition in ('uniform','low_current','high_current'):
            sub=[r for r in mm if r['condition']==condition]
            by_condition.append(dict(condition=condition,method=method,cells=len(sub),
                battery_macro_mae=float(np.mean([r['mae'] for r in sub]))))
    write_csv(out/'metrics_by_battery.csv',metrics);write_csv(out/'summary.csv',summary)
    write_csv(out/'metrics_by_condition.csv',by_condition)
    report=['# 확장 데이터 Selector 개발 검증','',
        '개발용 8개 셀을 하나씩 제외하여 비교했습니다. 각 셀의 SOH 예측값은 기존 Predictor 후보의 값을 그대로 사용했습니다.',
        '최종 평가용 RW15·16·19·20은 평가하지 않았습니다. 이전 개발 결과를 본 뒤 정한 탐색 실험입니다.','',
        '| 방법 | 셀 평균 MAE (%p) | 셀 평균 선택 길이 | 고정 기준보다 개선/악화 셀 수 |',
        '|---|---:|---:|---:|']
    for r in summary:
        report.append(f"| {r['method']} | {100*r['battery_macro_mae']:.4f} | {r['battery_macro_mean_lookback']:.2f} | {r['cells_better_than_default']}/{r['cells_worse_than_default']} |")
    report+=['','## 비교 방법','',
        '- Classifier: C10의 15개 Context로 오차 최소 길이를 분류합니다.',
        '- ErrorAware: 같은 Context로 세 길이의 절대오차를 예측하고 가장 작은 것을 고릅니다.',
        '- GatedError: 같은 오차 예측을 사용하되, 학습 측 OOF로 고른 기본 길이 대비 예상 개선이 threshold보다 클 때만 전환합니다.',
        '- threshold 후보는 SOH MAE 기준 0, 0.05, 0.1, 0.2, 0.5, 1, 2%p 및 전환하지 않음입니다. 검증 전에 고정했습니다.',
        '- 내부 보정 셀을 Predictor 라벨 생성에서도 제외하여 5개 셀 학습 → 1개 라벨 생성 → 1개 보정 → 1개 외부 개발 검증을 분리했습니다.',
        '- 기본 길이와 threshold는 각 외부 분할의 학습 자료만 사용합니다. 외부 개발 검증 정답으로 threshold를 조정하지 않습니다.','',
        '## 해석 범위','',
        '- Predictor는 Context Ridge입니다. 원신호 MLP 모델을 다시 학습한 실험은 아닙니다.',
        '- 새 운용 변화 변수를 추가하지 않았습니다. 기존 입력을 유지해 Selector 선택 규칙을 비교했습니다.',
        '- 후보 Predictor가 5→6, 6→7개 학습 셀로 바뀌는 OOF-재학습 차이는 남아 있습니다.',
        '- Ridge 오차 예측은 음수가 될 수 있어 확률이나 보장된 실제 오차로 해석하지 않습니다. 임의 clipping은 하지 않았습니다.',
        '- Oracle은 정답을 알고 고른 사후 참고값입니다. 배포 가능한 Selector가 아닙니다.',
        '- 평균 선택 길이는 계산 시간 측정값이 아닙니다. 누적 사용 이력은 모든 후보에 유지됩니다.',
        '- 학습·검증 셀 일부가 분할 간 겹치므로 셀별 차이만 보고 통계적 유의성이나 독립 일반화를 주장하지 않습니다.','']
    (out/'FINDINGS.md').write_text('\n'.join(report))
    write_json(out/'run_summary.json',dict(status='DEVELOPMENT_EVALUATION_COMPLETE',completed_utc=now(),
        protocol_sha256=signature(p),development_samples=155,prediction_rows=len(predictions),
        new_predictor_teachers=168,nested_error_selectors=56,outer_error_selectors=8,outer_classifiers=8,
        final_test_payload_loaded=False,final_test_evaluated=False,inference_time_measured=False,
        negative_outer_error_scores=sum(r[f'estimated_error_lb{L}']<0 for r in decisions for L in LENGTHS)))
    write_json(out/'artifact_manifest.json',dict(protocol_sha256=signature(p),
        files_sha256={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()}))
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=ROOT/'outputs/expanded_predictor_baseline_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/expanded_selector_v1')
    args=parser.parse_args(); out=args.output.resolve()
    if out.exists():raise FileExistsError('Use a new output directory; completed runs are immutable')
    p=make_protocol(args.baseline.resolve());out.mkdir(parents=True)
    write_json(out/'protocol.json',p)
    write_json(out/'protocol_frozen.json',dict(frozen_utc=now(),protocol_sha256=signature(p)))
    check_frozen(p);fit_all(out,p);evaluate(out,p)


if __name__=='__main__':main()
