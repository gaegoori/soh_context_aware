"""Paired study of candidate refitting and selector transfer.

For each test cell and calibration cell: 2 cells fit the frozen candidate bank,
1 other development cell trains each selector, 1 cell is untouched for fitting.
The SAME selector is applied to 2-cell and 3-cell banks on the SAME test targets.
True test SOH is used only to score methods and construct posthoc diagnostics.
"""
from scripts.paths import PROJECT_ROOT, SCRIPTS, source_names
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
import torch

from scripts.models.adaptive_models import CycleHistory,LOOKBACKS
from scripts.models.temporal_context_models import L2,context_at_L,fit_selectors,choose,infer_one
from scripts.experiments.train_temporal_context import (check_inputs as check_parent_inputs,hashes_ok,
                                    load_npz,signature)
from scripts.experiments.train_lookback_oof import read_csv,sha256,write_csv,write_json

PROJECT=PROJECT_ROOT
DEFAULT_SOURCE=PROJECT/'outputs/temporal_context_ridge_v1'
DEFAULT_OUTPUT=PROJECT/'outputs/selector_transfer_v1'
BANKS=('FrozenPair','RefitThree')
METHODS=('FixedContext10','FixedContext20','FixedContext50','CalibrationBest',
         'LabelMajority','Adaptive','ErrorAware','Oracle')


def now():return datetime.now(timezone.utc).isoformat()


def prepare_protocol(source):
    parent=json.loads((source/'protocol.json').read_text())
    check_parent_inputs(parent)
    check=json.loads((source/'verification.json').read_text())
    manifest=json.loads((source/'artifact_manifest.json').read_text())
    parent_sig=signature(parent)
    if check['status']!='PASS' or check['protocol_sha256']!=parent_sig or manifest['protocol_sha256']!=parent_sig:
        raise ValueError('Verified temporal Context Ridge experiment required')
    hashes_ok(source,manifest['files_sha256'])
    root=Path(parent['source_data_root']);files=set(source/name for name in manifest['files_sha256'])
    files.update((source/'artifact_manifest.json',source/'verification.json'))
    files.update(Path(name) for name in parent['input_sha256'])
    return dict(experiment='selector_transfer_v1',version=1,source_root=str(source),
        source_data_root=str(root),parent_protocol_sha256=parent_sig,batteries=parent['batteries'],
        lookbacks=LOOKBACKS.tolist(),banks=list(BANKS),methods=list(METHODS),
        task='diagnose candidate refit effects on the same unseen target and frozen selector',
        design='4 outer cells x 3 calibration-cell rotations; 2 predictor-fit / 1 selector-fit / 1 outer-test cell',
        frozen_bank='saved 2-cell OOF Ridge candidates; identical to those generating calibration errors',
        refit_bank='saved final Ridge candidates on all 3 development cells; adds calibration cell',
        selectors=dict(input='C10, 15 features, scaler fitted only on calibration-cell target rows',
            target='new Ridge OOF errors from that exact frozen bank on its excluded calibration cell',
            classification='mean cross entropy + 0.1/2*||W||²; observed classes only',
            error_regression='each output mean squared OOF error residual + 0.1*||w||²',
            l2=L2,refitting_between_banks=False,calibration_cell_search=False),
        comparison='same selector parameters/decisions and test rows in both bank conditions',
        fixed_choice='CalibrationBest uses mean errors on the one calibration cell, not pooled three-cell OOF',
        diagnostic='test truth gives best length for each bank; frozen-bank oracle choice applied to refit bank is posthoc only',
        tie_rule='exact argmin/argmax ties choose shortest observed length; error scores not clipped',
        reporting='average over 3 calibration rotations within each outer cell, then equal mean of 4 outer cells',
        independent_units='4 cells and 62 distinct targets, not 186 independent samples or 12 independent test cells',
        previous_comparison='selectors now learn on 1 cell; previous experiment pooled 3 OOF cells. Do not attribute cross-experiment differences solely to refitting.',
        latency='not measured: controlled diagnostic, no efficiency claims',
        tuning='none; fixed L2=0.1, all 24 selectors fitted before evaluation',
        evidence='exploratory, already inspected four cells; no guarantee of learnable selection or generalization',
        input_sha256={str(f.resolve()):sha256(f) for f in sorted(files)},
        code_sha256={name:sha256(SCRIPTS/name) for name in source_names((
            'train_selector_transfer.py','temporal_context_models.py','error_selector_models.py',
            'adaptive_models.py','preprocess_main.py','train_lookback_oof.py','train_temporal_context.py'))})


def check_inputs(p):
    hashes_ok('/',p['input_sha256']);hashes_ok(SCRIPTS,p['code_sha256'])


def fit_all(output,p):
    source,root=Path(p['source_root']),Path(p['source_data_root'])
    samples={r['sample_id']:r for r in read_csv(root/'samples.csv')}
    oof={(r['outer_test_battery'],r['sample_id'],int(r['lookback'])):r
         for r in read_csv(source/'oof_predictions.csv')}
    features=load_npz(source/'features.npz')
    context={str(sid):x[0] for sid,x in zip(features['sample_id'],features['context_by_length'])}
    splits=[]
    for outer in p['batteries']:
        for calibration in p['batteries']:
            if calibration==outer:continue
            split=json.loads((root/'folds'/f'test_{outer}__val_{calibration}'/'split.json').read_text())
            if (set(split['train_batteries'])!=set(p['batteries'])-{outer,calibration}
                or split['validation_batteries']!=[calibration] or split['test_batteries']!=[outer]):
                raise ValueError('Expected disjoint 2/1/1 battery split')
            ids=split['validation']
            if set(ids)&(set(split['train'])|set(split['test'])):raise ValueError('Overlapping roles')
            raw=np.array([context[sid] for sid in ids])
            errors=np.array([[float(oof[outer,sid,int(lb)]['absolute_error']) for lb in LOOKBACKS] for sid in ids])
            if any(samples[sid]['battery']!=calibration for sid in ids):raise ValueError('Wrong selector training cell')
            selector,stats=fit_selectors(raw,errors,np.array([calibration]*len(ids)))
            folder=output/'models'/f'test_{outer}'/f'calibration_{calibration}'
            folder.mkdir(parents=True)
            np.savez(folder/'parameters.npz',**selector)
            np.savez(folder/'training_data.npz',sample_id=np.array(ids),context_raw=raw,
                     errors=errors,class_label=np.argmin(errors,axis=1))
            meta=dict(outer_test_battery=outer,calibration_battery=calibration,
                predictor_fit_batteries=split['train_batteries'],predictor_fit_sample_ids=split['train'],
                selector_fit_sample_ids=ids,test_sample_ids=split['test'],l2=L2,**stats)
            write_json(folder/'fit.json',meta)
            splits.append(meta)
            print(json.dumps(dict(event='selectors_fitted',outer=outer,calibration=calibration,
                training_rows=len(ids),classes=stats['class_counts'])),flush=True)
    write_json(output/'splits.json',splits)
    write_json(output/'fitting_complete.json',dict(status='ALL_SELECTORS_FITTED',protocol_sha256=signature(p),
        completed_utc=now(),new_selectors=24,new_predictors=0,
        files_sha256={str(f.relative_to(output)):sha256(f) for f in sorted((output/'models').rglob('*')) if f.is_file()}))


def evaluate(output,p):
    if (output/'run_summary.json').exists():raise FileExistsError('Existing evaluation is preserved')
    check_inputs(p)
    fitted=json.loads((output/'fitting_complete.json').read_text())
    if fitted['status']!='ALL_SELECTORS_FITTED' or fitted['protocol_sha256']!=signature(p):raise ValueError('Fit all selectors first')
    hashes_ok(output,fitted['files_sha256'])
    source,root=Path(p['source_root']),Path(p['source_data_root'])
    samples={r['sample_id']:r for r in read_csv(root/'samples.csv')}
    splits=json.loads((output/'splits.json').read_text())
    predictions=[];selections=[];shift=[];diagnostics=[]
    history={b:CycleHistory.load(root,b) for b in p['batteries']}
    for split in splits:
        outer,cal=split['outer_test_battery'],split['calibration_battery']
        selector=load_npz(output/'models'/f'test_{outer}'/f'calibration_{cal}'/'parameters.npz')
        banks={mode:{int(lb):load_npz(source/'models'/f'test_{outer}'/
            (f'oof_{cal}' if mode=='FrozenPair' else 'final')/f'lb_{lb}'/'parameters.npz')
            for lb in LOOKBACKS} for mode in BANKS}
        for sid in split['test_sample_ids']:
            row=samples[sid];n=int(row['history_end_exclusive'])
            if row['battery']!=outer:raise ValueError('Wrong outer sample')
            raw=context_at_L(history[outer],n,10)
            chosen={}
            for method in ('Adaptive','ErrorAware'):
                length,scores=choose(raw,selector,method);chosen[method]=length
                selections.append(dict(outer_test_battery=outer,calibration_battery=cal,sample_id=sid,
                    method=method,selected_lookback=length,
                    **{f'score_lb{lb}':float(scores[j]) for j,lb in enumerate(LOOKBACKS)},
                    negative_score_count=int((scores<0).sum())))
            # Predictions are made without passing truth into inference/selection.
            mode_values={}
            for mode,bank in banks.items():
                values={}
                for method in METHODS[:-1]:
                    call_method='GlobalBestOOF' if method=='CalibrationBest' else method
                    value,length=infer_one(history[outer],n,bank,selector,call_method)
                    if not np.isfinite(value):raise ValueError('Nonfinite prediction')
                    if method in chosen and length!=chosen[method]:raise ValueError('Selector changed between banks')
                    values[method]=(value,length)
                mode_values[mode]=values
            truth=float(row['soh']);bank_errors={};best={}
            for mode,values in mode_values.items():
                errors=np.array([abs(values[f'FixedContext{lb}'][0]-truth) for lb in LOOKBACKS])
                bank_errors[mode]=errors;best[mode]=int(LOOKBACKS[np.argmin(errors)])
                values['Oracle']=values[f'FixedContext{best[mode]}']
                for method,(value,length) in values.items():
                    predictions.append(dict(outer_test_battery=outer,calibration_battery=cal,bank=mode,
                        sample_id=sid,stage=row['stage'],method=method,soh_true=truth,soh_pred=value,
                        absolute_error=abs(value-truth),lookback=length))
            pair,refit=bank_errors['FrozenPair'],bank_errors['RefitThree']
            pair_best,refit_best=best['FrozenPair'],best['RefitThree']
            transfer=float(refit[LOOKBACKS.tolist().index(pair_best)]-min(refit))
            shift.append(dict(outer_test_battery=outer,calibration_battery=cal,sample_id=sid,
                frozen_best_lookback=pair_best,refit_best_lookback=refit_best,
                best_length_changed=int(pair_best!=refit_best),
                frozen_gap_to_second=float(np.sort(pair)[1]-min(pair)),
                refit_gap_to_second=float(np.sort(refit)[1]-min(refit)),
                frozen_oracle_mae=float(min(pair)),refit_oracle_mae=float(min(refit)),
                frozen_oracle_choice_on_refit_error=float(refit[LOOKBACKS.tolist().index(pair_best)]),
                frozen_oracle_choice_on_refit_regret=transfer,
                **{f'frozen_error_lb{lb}':float(pair[j]) for j,lb in enumerate(LOOKBACKS)},
                **{f'refit_error_lb{lb}':float(refit[j]) for j,lb in enumerate(LOOKBACKS)}))
            for method,length in chosen.items():
                j=LOOKBACKS.tolist().index(length)
                rp,rf=float(pair[j]-min(pair)),float(refit[j]-min(refit))
                diagnostics.append(dict(outer_test_battery=outer,calibration_battery=cal,sample_id=sid,
                    method=method,selected_lookback=length,matches_frozen_best=int(length==pair_best),
                    matches_refit_best=int(length==refit_best),frozen_selection_regret=rp,
                    refit_selection_regret=rf,regret_change_after_refit=rf-rp))
        print(json.dumps(dict(event='paired_evaluation',outer=outer,calibration=cal,
                              samples=len(split['test_sample_ids']))),flush=True)
    write_csv(output/'predictions.csv',predictions);write_csv(output/'selections.csv',selections)
    write_csv(output/'label_shift.csv',shift);write_csv(output/'selector_diagnostics.csv',diagnostics)
    summarize(output,p,predictions,shift,diagnostics)
    check_inputs(p)
    write_json(output/'run_summary.json',dict(status='EVALUATION_COMPLETE',protocol_sha256=signature(p),
        completed_utc=now(),new_predictors=0,new_selectors=24,paired_splits=12,
        unique_test_samples=len(samples),test_sample_rotations=len(shift),prediction_rows=len(predictions),
        current_run_test_tuning=False,evidence='exploratory controlled diagnostic; repeated rotations are not independent cells'))
    write_json(output/'artifact_manifest.json',dict(protocol_sha256=signature(p),
        files_sha256={str(f.relative_to(output)):sha256(f) for f in sorted(output.rglob('*'))
                      if f.is_file() and f.name!='artifact_manifest.json'}))


def summarize(output,p,predictions,shift,diagnostics):
    groups=defaultdict(list)
    for r in predictions:groups[r['outer_test_battery'],r['calibration_battery'],r['bank'],r['method']].append(r)
    per_split=[]
    for (outer,cal,bank,method),rows in sorted(groups.items()):
        errors=np.array([r['absolute_error'] for r in rows])
        oracle=np.mean([r['absolute_error'] for r in groups[outer,cal,bank,'Oracle']])
        per_split.append(dict(outer_test_battery=outer,calibration_battery=cal,bank=bank,method=method,
            samples=len(rows),mae=float(errors.mean()),rmse=float(np.sqrt(np.mean(errors**2))),
            regret=float(errors.mean()-oracle),mean_lookback=float(np.mean([r['lookback'] for r in rows]))))
    write_csv(output/'metrics_by_split.csv',per_split)
    per_cell=[];summary=[]
    for bank in BANKS:
        for method in METHODS:
            for outer in p['batteries']:
                rows=[r for r in per_split if (r['outer_test_battery'],r['bank'],r['method'])==(outer,bank,method)]
                assert len(rows)==3
                per_cell.append(dict(outer_test_battery=outer,bank=bank,method=method,
                    unique_targets=rows[0]['samples'],rotations=3,
                    **{key:float(np.mean([r[key] for r in rows])) for key in ('mae','rmse','regret','mean_lookback')}))
            rows=[r for r in per_cell if (r['bank'],r['method'])==(bank,method)]
            summary.append(dict(bank=bank,method=method,
                battery_macro_mae=float(np.mean([r['mae'] for r in rows])),
                battery_macro_rmse=float(np.mean([r['rmse'] for r in rows])),
                battery_macro_regret=float(np.mean([r['regret'] for r in rows])),
                battery_macro_mean_lookback=float(np.mean([r['mean_lookback'] for r in rows]))))
    write_csv(output/'metrics_by_battery.csv',per_cell);write_csv(output/'summary.csv',summary)
    indexes={(r['outer_test_battery'],r['bank'],r['method']):r for r in per_cell}
    effects=[]
    for scope in [*p['batteries'],'MACRO']:
        cells=p['batteries'] if scope=='MACRO' else [scope]
        for method in METHODS:
            effects.append(dict(scope=scope,method=method,
                refit_minus_frozen_mae=float(np.mean([indexes[b,'RefitThree',method]['mae']-indexes[b,'FrozenPair',method]['mae'] for b in cells])),
                refit_minus_frozen_regret=float(np.mean([indexes[b,'RefitThree',method]['regret']-indexes[b,'FrozenPair',method]['regret'] for b in cells]))))
    write_csv(output/'bank_effects.csv',effects)
    comparisons=[]
    for scope in [*p['batteries'],'MACRO']:
        cells=p['batteries'] if scope=='MACRO' else [scope]
        for bank in BANKS:
            for method in ('Adaptive','ErrorAware'):
                for baseline in ('FixedContext10','CalibrationBest','Oracle'):
                    comparisons.append(dict(scope=scope,bank=bank,method=method,baseline=baseline,
                        mae_difference=float(np.mean([indexes[b,bank,method]['mae']-indexes[b,bank,baseline]['mae'] for b in cells]))))
    write_csv(output/'paired_comparisons.csv',comparisons)
    shift_summary=[]
    for outer in p['batteries']:
        rows=[r for r in shift if r['outer_test_battery']==outer]
        shift_summary.append(dict(outer_test_battery=outer,unique_targets=len(rows)//3,target_rotations=len(rows),
            label_flip_rate=float(np.mean([r['best_length_changed'] for r in rows])),
            frozen_gap_median=float(np.median([r['frozen_gap_to_second'] for r in rows])),
            refit_gap_median=float(np.median([r['refit_gap_to_second'] for r in rows])),
            transferred_oracle_regret=float(np.mean([r['frozen_oracle_choice_on_refit_regret'] for r in rows]))))
    write_csv(output/'label_shift_by_battery.csv',shift_summary)
    diag_summary=[]
    for outer in p['batteries']:
        for method in ('Adaptive','ErrorAware'):
            rows=[r for r in diagnostics if (r['outer_test_battery'],r['method'])==(outer,method)]
            diag_summary.append(dict(outer_test_battery=outer,method=method,
                **{key:float(np.mean([r[key] for r in rows])) for key in ('matches_frozen_best','matches_refit_best',
                    'frozen_selection_regret','refit_selection_regret','regret_change_after_refit')}))
    write_csv(output/'selector_diagnostics_by_battery.csv',diag_summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--phase',choices=('fit','evaluate','all'),default='all')
    args=parser.parse_args();output=args.output.resolve();torch.set_num_threads(1)
    if args.phase in ('fit','all'):
        if output.exists():raise FileExistsError('Preserve existing runs; choose a fresh output path')
        p=prepare_protocol(args.source.resolve());output.mkdir(parents=True)
        write_json(output/'protocol.json',p);write_json(output/'protocol_frozen.json',dict(protocol_sha256=signature(p),frozen_utc=now()))
        fit_all(output,p)
    else:p=json.loads((output/'protocol.json').read_text())
    if args.phase in ('evaluate','all'):evaluate(output,p)


if __name__=='__main__':main()
