"""Fixed-protocol temporal Context Ridge study, with two fresh OOF selectors.

All outer folds are fitted before any outer evaluation. Existing experiments are
read-only. These already inspected four cells provide exploratory evidence only.
"""
from scripts.paths import PROJECT_ROOT, SCRIPTS, hash_matches, source_names
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
import torch

from scripts.models.adaptive_models import CONTEXT_NAMES, CycleHistory, LOOKBACKS, context_at
from scripts.preprocessing.battery_data import DEFAULT_ROOT
from scripts.models.temporal_context_models import (L2, METHODS, DEPLOYABLE, choose, context_at_L,
    fit_ridge, fit_selectors, infer_one, predict_ridge)
from scripts.experiments.train_lookback_oof import get_fold, read_csv, sha256, write_csv, write_json
from scripts.experiments.train_adaptive_selector import load_final_split

PROJECT = PROJECT_ROOT
DEFAULT_OUTPUT = PROJECT/'outputs/temporal_context_ridge_v1'
DEFAULT_BASELINE = PROJECT/'outputs/adaptive_selector_v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def signature(p):
    return hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()


def load_npz(path):
    with np.load(path,allow_pickle=False) as d:
        return {k:d[k].copy() for k in d.files}


def hashes_ok(base, hashes):
    for name,digest in hashes.items():
        if not hash_matches(Path(base)/name, digest):
            raise ValueError(f'Changed file: {Path(base)/name}')


def check_inputs(p):
    hashes_ok('/',p['input_sha256'])
    hashes_ok(SCRIPTS,p['code_sha256'])


def prepare_protocol(root, baseline):
    old=json.loads((baseline/'protocol.json').read_text())
    old_check=json.loads((baseline/'verification.json').read_text())
    artifacts=json.loads((baseline/'artifact_manifest.json').read_text())
    old_sig=signature(old)
    if old_check['status']!='PASS' or old_check['protocol_sha256']!=old_sig or artifacts['protocol_sha256']!=old_sig:
        raise ValueError('Verified original Context Ridge baseline required')
    if Path(old['source_data_root']).resolve()!=root:
        raise ValueError('Baseline data root mismatch')
    hashes_ok(baseline,artifacts['files_sha256'])
    hashes_ok(root,old['data_sha256'])
    hashes_ok(SCRIPTS,old['code_sha256'])
    manifest=json.loads((root/'manifest.json').read_text())
    batteries=manifest['batteries']
    if batteries!=old['batteries'] or manifest['context_names']!=CONTEXT_NAMES:
        raise ValueError('Canonical cells/features changed')
    inputs={root/name for name in ('manifest.json','samples.csv','audit.json')}
    inputs.update(baseline/name for name in ('protocol.json','verification.json','predictions.csv','artifact_manifest.json'))
    for outer in batteries:
        inputs.update((root/f'{outer}.npz',root/f'{outer}_cycles.csv'))
        load_final_split(root,outer,batteries)
        inputs.update(root/'folds'/f'test_{outer}__final'/name for name in ('split.json','scalers.json'))
        context_dir=baseline/'models'/f'test_{outer}'/'context_models'
        complete=json.loads((context_dir/'complete.json').read_text())
        if complete['protocol_sha256']!=old_sig:raise ValueError('Changed baseline context signature')
        hashes_ok(context_dir,complete['files_sha256'])
        inputs.update(context_dir/name for name in ('parameters.npz','fit.json','complete.json'))
        for heldout in batteries:
            if outer==heldout:continue
            name=f'test_{outer}__val_{heldout}'
            get_fold(root,name,batteries)
            inputs.add(root/'folds'/name/'split.json')
    return dict(experiment='temporal_context_ridge_v1',version=1,
        source_data_root=str(root),source_baseline_root=str(baseline),batteries=batteries,
        lookbacks=LOOKBACKS.tolist(),methods=list(METHODS),
        features=dict(selector=CONTEXT_NAMES,
            predictor=[n.replace('last10_','window_') for n in CONTEXT_NAMES],
            shared='first 6: cumulative use and last-cycle values, identical for all L',
            varying='last 9: duration mean/std/slope, pooled resampled V/I/T mean/std over L cycles',
            std_ddof=0,slope='actual cycle indices 0..L-1',dtype='canonical float32 features; float64 models',
            length_meaning='recent statistics window; cumulative features summarize older history too'),
        preprocessing='fit mean/std on training target rows; all-missing training columns inactive in every role; mean-impute; constant std=1',
        predictor=dict(model='Ridge regression',l2=L2,
            objective='mean squared SOH error + l2*||w||^2; unpenalized intercept',
            counts=dict(oof=36,final_candidates=12,controls=8),
            seeds='none: deterministic linear solution; no repeated identical fits'),
        classification_selector=dict(input='raw C10 -> train-only scaler',l2=L2,
            objective='mean cross entropy + l2/2*||W||^2; unpenalized intercepts',
            optimizer='zero init L-BFGS-B, maxiter=500, ftol=1e-12, gtol=1e-8',
            absent_class='fit observed classes only; one class uses constant choice',
            selection='probability argmax, exact ties shortest'),
        error_selector=dict(input='same C10 and scaler as classification selector',l2=L2,
            target='new deterministic OOF absolute errors for three Ridge candidates; no old neural labels or seed averaging',
            objective='per output mean squared error + l2*||w||^2; unpenalized intercept',
            selection='argmin raw error scores, no clipping, exact ties shortest'),
        label_rule='argmin new OOF absolute error per sample; exact ties shorter L',
        weighting='equal training sample weights; global-best uses equal cell weighting of OOF MAEs',
        model_selection='no hyperparameter search, no outer-result tuning in this run',
        evidence='exploratory: previously inspected RW3/RW4/RW5/RW6 and 62 reference targets',
        transfer_limit='OOF teacher fits 2 cells; final candidate fits 3 development cells',
        target_setup='measured reference SOH, initial reference capacity Q0 available; no current/future SOH input',
        fitting_order='all 36 OOF + 20 final/control regressors and all 8 selectors before any outer predictions',
        primary_metric='equal-cell macro SOH MAE; also per-cell RMSE, paired differences, Oracle gap and selection diagnostics',
        benchmark=dict(rounds=100,warmup_rounds=5,threads=1,batch_size=1,shuffle_seed=20261027,
            scope='causal cached metadata and resampled signals -> window context -> selector if needed -> normalization -> one predictor',
            reuse='C10 reused for selected L10; otherwise reuse 6 fixed values and compute only new 9 window statistics',
            excludes='raw I/O, segmentation/resampling, model loading and initial cumulative cache construction'),
        input_sha256={str(f.resolve()):sha256(f) for f in sorted(inputs)},
        code_sha256={name:sha256(SCRIPTS/name) for name in source_names((
            'temporal_context_models.py','train_temporal_context.py','adaptive_models.py',
            'error_selector_models.py','preprocess_main.py','battery_data.py',
            'train_lookback_oof.py','train_adaptive_selector.py'))},
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                      torch=str(torch.__version__),platform=platform.platform()))


def save_fit(folder,model,meta):
    folder.mkdir(parents=True)
    np.savez(folder/'parameters.npz',**model)
    write_json(folder/'fit.json',meta)


def fit_all(output,p):
    root=Path(p['source_data_root']);baseline=Path(p['source_baseline_root'])
    rows=read_csv(root/'samples.csv');by_id={r['sample_id']:r for r in rows}
    histories={b:CycleHistory.load(root,b) for b in p['batteries']}
    features={};reproduction_max=0.
    for b in p['batteries']:
        canonical=load_npz(root/f'{b}.npz')
        for row in rows:
            if row['battery']!=b:continue
            n=int(row['history_end_exclusive'])
            if n<50 or not canonical['cycle_valid'][n-50:n].all():
                raise ValueError('All methods need the same 50 valid cycles')
            vectors=np.array([context_at_L(histories[b],n,int(lb)) for lb in LOOKBACKS])
            np.testing.assert_array_equal(vectors[0],context_at(histories[b],n))
            np.testing.assert_array_equal(vectors[0],canonical['context'][n-1])
            np.testing.assert_array_equal(vectors[:,:6],np.tile(vectors[0,:6],(3,1)))
            features[row['sample_id']]=vectors
    np.savez(output/'features.npz',sample_id=np.array([r['sample_id'] for r in rows]),
             lookbacks=LOOKBACKS,context_by_length=np.array([features[r['sample_id']] for r in rows]))
    oof_predictions=[];labels=[];fit_count=0
    def inputs(ids,j):return np.array([features[sid][j] for sid in ids])
    def targets(ids):return np.array([float(by_id[sid]['soh']) for sid in ids])
    for outer in p['batteries']:
        final,_=load_final_split(root,outer,p['batteries'])
        oof_by_id={}
        for heldout in final['train_batteries']:
            fold=f'test_{outer}__val_{heldout}'
            split=get_fold(root,fold,p['batteries'])
            predictions_by_length=[]
            for j,length in enumerate(LOOKBACKS.tolist()):
                model=fit_ridge(inputs(split['train'],j),targets(split['train']))
                meta=dict(role='oof',outer=outer,heldout=heldout,lookback=length,
                    train_sample_ids=split['train'],train_batteries=split['train_batteries'],
                    label_sample_ids=split['validation'],l2=L2,
                    inactive_features=np.flatnonzero(~model['active']).tolist())
                save_fit(output/'models'/f'test_{outer}'/f'oof_{heldout}'/f'lb_{length}',model,meta)
                values=predict_ridge(inputs(split['validation'],j),model)
                predictions_by_length.append(values);fit_count+=1
                for sid,value in zip(split['validation'],values):
                    row=by_id[sid];truth=float(row['soh'])
                    oof_predictions.append(dict(outer_test_battery=outer,heldout_battery=heldout,
                        sample_id=sid,lookback=length,soh_true=truth,soh_pred=float(value),
                        absolute_error=float(abs(value-truth))))
            prediction_matrix=np.array(predictions_by_length).T
            errors=np.abs(prediction_matrix-targets(split['validation'])[:,None])
            for sid,values,error in zip(split['validation'],prediction_matrix,errors):
                oof_by_id[sid]=error
                best=int(np.argmin(error))
                labels.append(dict(outer_test_battery=outer,heldout_battery=heldout,sample_id=sid,
                    best_lookback=int(LOOKBACKS[best]),class_label=best,
                    error_gap_to_second_best=float(np.sort(error)[1]-np.min(error)),
                    **{f'error_lb{lb}':float(error[j]) for j,lb in enumerate(LOOKBACKS)}))
        if set(oof_by_id)!=set(final['train']):raise ValueError('OOF row coverage mismatch')
        for j,length in enumerate(LOOKBACKS.tolist()):
            model=fit_ridge(inputs(final['train'],j),targets(final['train']))
            if length==10:
                old=load_npz(baseline/'models'/f'test_{outer}'/'context_models'/'parameters.npz')
                for new_key,old_key in (('mean','context_mean'),('std','context_std'),('weight','ridge_weight'),('bias','ridge_bias')):
                    np.testing.assert_allclose(model[new_key],old[old_key],rtol=1e-10,atol=1e-12)
                    reproduction_max=max(reproduction_max,float(np.max(np.abs(model[new_key]-old[old_key]))))
            save_fit(output/'models'/f'test_{outer}'/'final'/f'lb_{length}',model,
                dict(role='final',outer=outer,lookback=length,train_sample_ids=final['train'],
                     train_batteries=final['train_batteries'],l2=L2,inactive_features=np.flatnonzero(~model['active']).tolist()))
            fit_count+=1
        for method,dimension in (('Usage3',3),('LastState6',6)):
            model=fit_ridge(inputs(final['train'],0)[:,:dimension],targets(final['train']))
            save_fit(output/'models'/f'test_{outer}'/'final'/method,model,
                dict(role='control',outer=outer,dimensions=dimension,train_sample_ids=final['train'],
                     train_batteries=final['train_batteries'],l2=L2,inactive_features=np.flatnonzero(~model['active']).tolist()))
            fit_count+=1
        errors=np.array([oof_by_id[sid] for sid in final['train']])
        cell_labels=np.array([by_id[sid]['battery'] for sid in final['train']])
        raw=inputs(final['train'],0)
        selector,stats=fit_selectors(raw,errors,cell_labels)
        save_fit(output/'models'/f'test_{outer}'/'selectors',selector,
            dict(role='selectors',outer=outer,train_sample_ids=final['train'],
                 train_batteries=final['train_batteries'],l2=L2,**stats))
        np.savez(output/'models'/f'test_{outer}'/'selectors'/'training_data.npz',
            sample_id=np.array(final['train']),battery=cell_labels,context_raw=raw,
            errors=errors,class_label=np.argmin(errors,axis=1))
        print(json.dumps(dict(event='fold_fitted',outer=outer,ridge_models_so_far=fit_count,
            selector_training_rows=len(raw),class_counts=stats['class_counts'])),flush=True)
    write_csv(output/'oof_predictions.csv',oof_predictions);write_csv(output/'labels.csv',labels)
    write_json(output/'reproduction.json',dict(status='PASS',scope='all 62 canonical C10 feature rows and all four original final Ridge fits',
        max_abs_parameter_difference=reproduction_max))
    write_json(output/'fitting_complete.json',dict(status='ALL_MODELS_FITTED',protocol_sha256=signature(p),
        completed_utc=now(),ridge_models=fit_count,classifiers=4,error_selectors=4,
        files_sha256={str(f.relative_to(output)):sha256(f) for f in sorted(output.rglob('*'))
            if f.is_file() and f.name not in ('protocol.json','protocol_frozen.json')}))


def evaluate(output,p):
    if (output/'run_summary.json').exists():raise FileExistsError('Evaluation already complete')
    check_inputs(p)
    fitted=json.loads((output/'fitting_complete.json').read_text())
    if fitted['status']!='ALL_MODELS_FITTED' or fitted['protocol_sha256']!=signature(p):
        raise ValueError('Fit all outer folds before evaluation')
    hashes_ok(output,fitted['files_sha256'])
    root=Path(p['source_data_root']);baseline=Path(p['source_baseline_root'])
    samples={r['sample_id']:r for r in read_csv(root/'samples.csv')}
    old_rows=[r for r in read_csv(baseline/'predictions.csv') if r['method']=='ContextOnlyRidge']
    old_predictions=defaultdict(list)
    for r in old_rows:old_predictions[r['sample_id']].append(float(r['soh_pred']))
    predictions=[];selections=[];timings=[];reproduction_max=0.
    for outer in p['batteries']:
        final,_=load_final_split(root,outer,p['batteries'])
        rows=[samples[sid] for sid in final['test']]
        history=CycleHistory.load(root,outer)
        base=output/'models'/f'test_{outer}'
        bank={int(lb):load_npz(base/'final'/f'lb_{lb}'/'parameters.npz') for lb in LOOKBACKS}
        bank.update({m:load_npz(base/'final'/m/'parameters.npz') for m in ('Usage3','LastState6')})
        selector=load_npz(base/'selectors'/'parameters.npz')
        expected={};fixed={}
        for row in rows:
            raw=context_at_L(history,int(row['history_end_exclusive']),10)
            for method in ('Adaptive','ErrorAware'):
                length,scores=choose(raw,selector,method)
                selections.append(dict(outer_test_battery=outer,sample_id=row['sample_id'],method=method,
                    selected_lookback=length,score_kind='probability' if method=='Adaptive' else 'estimated_absolute_error',
                    **{f'score_lb{lb}':float(scores[j]) for j,lb in enumerate(LOOKBACKS)},
                    negative_score_count=int((scores<0).sum())))
        for method in DEPLOYABLE:
            for row in rows:
                sid,n=row['sample_id'],int(row['history_end_exclusive'])
                value,length=infer_one(history,n,bank,selector,method)
                if not np.isfinite(value):raise ValueError('Nonfinite prediction')
                if method.startswith('FixedContext'):fixed[sid,length]=value
                if method in ('GlobalBestOOF','LabelMajority','Adaptive','ErrorAware'):
                    np.testing.assert_allclose(value,fixed[sid,length],rtol=0,atol=1e-12)
                if method=='FixedContext10':
                    np.testing.assert_allclose(old_predictions[sid],value,rtol=1e-10,atol=1e-12)
                    reproduction_max=max(reproduction_max,max(abs(v-value) for v in old_predictions[sid]))
                expected[method,sid]=(value,length)
                truth=float(row['soh'])
                predictions.append(dict(outer_test_battery=outer,sample_id=sid,stage=row['stage'],method=method,
                    soh_true=truth,soh_pred=value,absolute_error=abs(value-truth),lookback='' if length is None else length))
        for row in rows:
            sid=row['sample_id'];truth=float(row['soh'])
            length=min(LOOKBACKS.tolist(),key=lambda lb:abs(fixed[sid,lb]-truth));value=fixed[sid,length]
            predictions.append(dict(outer_test_battery=outer,sample_id=sid,stage=row['stage'],method='Oracle',
                soh_true=truth,soh_pred=value,absolute_error=abs(value-truth),lookback=length))
        requests=[(m,r) for m in DEPLOYABLE for r in rows]
        rng=np.random.default_rng(p['benchmark']['shuffle_seed']+int(outer[2:]))
        for repeat in range(-p['benchmark']['warmup_rounds'],p['benchmark']['rounds']):
            for k in rng.permutation(len(requests)):
                method,row=requests[k];n=int(row['history_end_exclusive'])
                start=time.perf_counter_ns()
                value,length=infer_one(history,n,bank,selector,method)
                elapsed=time.perf_counter_ns()-start
                want,want_length=expected[method,row['sample_id']]
                if abs(value-want)>1e-12 or length!=want_length:raise ValueError('Timed result mismatch')
                if repeat>=0:
                    timings.append(dict(outer_test_battery=outer,sample_id=row['sample_id'],method=method,
                        repetition=repeat,lookback='' if length is None else length,latency_ns=elapsed))
        print(json.dumps(dict(event='outer_evaluated',outer=outer,samples=len(rows))),flush=True)
    write_csv(output/'predictions.csv',predictions);write_csv(output/'selections.csv',selections)
    write_csv(output/'latency_samples.csv',timings)
    summarize(output,p,predictions,selections,timings)
    check_inputs(p)
    write_json(output/'run_summary.json',dict(status='EVALUATION_COMPLETE',protocol_sha256=signature(p),
        completed_utc=now(),unique_test_samples=len(samples),prediction_rows=len(predictions),timing_rows=len(timings),
        ridge_models=56,classifiers=4,error_selectors=4,seed_repetitions=0,
        max_abs_original_context_ridge_prediction_difference=reproduction_max,
        current_run_test_tuning=False,evidence='exploratory; previously inspected four cells'))
    write_json(output/'artifact_manifest.json',dict(protocol_sha256=signature(p),
        files_sha256={str(f.relative_to(output)):sha256(f) for f in sorted(output.rglob('*'))
                      if f.is_file() and f.name!='artifact_manifest.json'}))


def summarize(output,p,predictions,selections,timings):
    groups=defaultdict(list);stages=defaultdict(list);latency=defaultdict(list)
    for r in predictions:
        groups[r['outer_test_battery'],r['method']].append(r)
        stages[r['outer_test_battery'],r['stage'],r['method']].append(r)
    for r in timings:latency[r['method']].append(r['latency_ns']/1e6)
    def metrics(rows):
        errors=np.array([r['absolute_error'] for r in rows])
        lengths=[r['lookback'] for r in rows if r['lookback']!='']
        return dict(samples=len(rows),mae=float(errors.mean()),rmse=float(np.sqrt(np.mean(errors**2))),
                    mean_lookback=float(np.mean(lengths)) if lengths else '')
    by_cell=[dict(outer_test_battery=b,method=m,**metrics(groups[b,m])) for b in p['batteries'] for m in METHODS]
    stats={(r['outer_test_battery'],r['method']):r for r in by_cell}
    write_csv(output/'metrics_by_battery.csv',by_cell)
    write_csv(output/'metrics_by_stage.csv',[dict(outer_test_battery=b,stage=s,method=m,**metrics(rows))
                                          for (b,s,m),rows in sorted(stages.items())])
    summary=[]
    for m in METHODS:
        rows=[r for r in predictions if r['method']==m];cost=latency.get(m,[])
        summary.append(dict(method=m,battery_macro_mae=float(np.mean([stats[b,m]['mae'] for b in p['batteries']])),
            battery_macro_rmse=float(np.mean([stats[b,m]['rmse'] for b in p['batteries']])),
            pooled_mae=metrics(rows)['mae'],mean_lookback=metrics(rows)['mean_lookback'],
            latency_median_ms=float(np.median(cost)) if cost else '',latency_p95_ms=float(np.percentile(cost,95)) if cost else ''))
    write_csv(output/'summary.csv',summary)
    paired=[]
    for scope in [*p['batteries'],'MACRO']:
        cells=p['batteries'] if scope=='MACRO' else [scope]
        for method in ('Adaptive','ErrorAware'):
            for baseline in ('FixedContext10','FixedContext20','FixedContext50','GlobalBestOOF','Oracle','Adaptive'):
                if method==baseline:continue
                paired.append(dict(scope=scope,method=method,baseline=baseline,
                    mae_difference=float(np.mean([stats[b,method]['mae']-stats[b,baseline]['mae'] for b in cells]))))
    write_csv(output/'paired_comparisons.csv',paired)
    indexed={(r['outer_test_battery'],r['sample_id'],r['method']):r for r in predictions}
    diagnostics=[]
    for r in selections:
        b,sid,m=r['outer_test_battery'],r['sample_id'],r['method']
        errors=np.array([indexed[b,sid,f'FixedContext{lb}']['absolute_error'] for lb in LOOKBACKS])
        best=int(LOOKBACKS[np.argmin(errors)]);chosen=r['selected_lookback']
        diagnostics.append(dict(outer_test_battery=b,sample_id=sid,method=m,selected_lookback=chosen,
            posthoc_best_lookback=best,matches_posthoc_best=int(best==chosen),
            regret=float(errors[LOOKBACKS.tolist().index(chosen)]-min(errors)),
            oracle_gap_to_second=float(np.sort(errors)[1]-min(errors))))
    write_csv(output/'selection_diagnostics.csv',diagnostics)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--baseline',type=Path,default=DEFAULT_BASELINE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--phase',choices=('fit','evaluate','all'),default='all')
    args=parser.parse_args();output=args.output.resolve()
    torch.set_num_threads(1)
    if args.phase in ('fit','all'):
        if output.exists():raise FileExistsError('Use a fresh output path; preserve existing runs')
        p=prepare_protocol(args.data_root.resolve(),args.baseline.resolve())
        output.mkdir(parents=True)
        write_json(output/'protocol.json',p)
        write_json(output/'protocol_frozen.json',dict(protocol_sha256=signature(p),frozen_utc=now()))
        fit_all(output,p)
    else:p=json.loads((output/'protocol.json').read_text())
    if args.phase in ('evaluate','all'):evaluate(output,p)


if __name__=='__main__':main()
