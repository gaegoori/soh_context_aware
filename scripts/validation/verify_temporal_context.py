"""Independent feature, fit, split, prediction, metric and timing-grid audit."""
from scripts.paths import hash_matches
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import warnings

import numpy as np

from scripts.experiments.train_lookback_oof import read_csv,sha256,write_json
from scripts.experiments.train_temporal_context import DEFAULT_OUTPUT,check_inputs,signature

LENGTHS=(10,20,50)
METHODS=('Usage3','LastState6','FixedContext10','FixedContext20','FixedContext50',
         'GlobalBestOOF','LabelMajority','Adaptive','ErrorAware','Oracle')


def near(a,b,atol=1e-10):
    np.testing.assert_allclose(a,b,atol=atol,rtol=1e-8,equal_nan=True)


def load(path):
    with np.load(path,allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}


def scale_fit(raw):
    x=np.array(raw,dtype=np.float64);active=np.isfinite(x).any(axis=0)
    mu=np.zeros(x.shape[1]);sd=np.ones(x.shape[1])
    mu[active]=np.nanmean(x[:,active],axis=0);sd[active]=np.nanstd(x[:,active],axis=0)
    return mu,np.where(sd>1e-6,sd,1.),active


def scaled(raw,model):
    raw=np.array(raw,dtype=np.float64)
    result=(np.where(np.isfinite(raw),raw,model['mean'])-model['mean'])/model['std']
    return np.where(model['active'],result,0.)


def audit_ridge(path,raw,y,l2):
    model=load(path/'parameters.npz')
    mu,sd,active=scale_fit(raw)
    near(model['mean'],mu);near(model['std'],sd)
    np.testing.assert_array_equal(model['active'],active)
    x=scaled(raw,model);n,d=x.shape
    design=np.column_stack((x,np.ones(n)))
    penalty=np.diag(np.r_[np.full(d,np.sqrt(n*l2)),0.])
    solved=np.linalg.lstsq(np.vstack((design,penalty)),np.r_[y,np.zeros(d+1)],rcond=None)[0]
    near(solved[:-1],model['weight']);near(solved[-1],model['bias'])
    return model


def verify(output):
    p=json.loads((output/'protocol.json').read_text());check_inputs(p);sig=signature(p)
    run=json.loads((output/'run_summary.json').read_text())
    manifest=json.loads((output/'artifact_manifest.json').read_text())
    frozen=json.loads((output/'protocol_frozen.json').read_text())
    fitted=json.loads((output/'fitting_complete.json').read_text())
    assert run['status']=='EVALUATION_COMPLETE' and not run['current_run_test_tuning']
    assert all(r['protocol_sha256']==sig for r in (run,manifest,frozen,fitted))
    assert datetime.fromisoformat(frozen['frozen_utc'])<=datetime.fromisoformat(fitted['completed_utc'])<=datetime.fromisoformat(run['completed_utc'])
    for mapping in (manifest['files_sha256'],fitted['files_sha256']):
        for name,digest in mapping.items():assert hash_matches(output/name, digest),name
    root=Path(p['source_data_root']);baseline=Path(p['source_baseline_root'])
    samples={r['sample_id']:r for r in read_csv(root/'samples.csv')};cells=p['batteries']
    data={b:load(root/f'{b}.npz') for b in cells}
    metadata={b:read_csv(root/f'{b}_cycles.csv') for b in cells}
    feature_file=load(output/'features.npz')
    assert len(feature_file['sample_id'])==len(set(feature_file['sample_id']))==len(samples)
    feature={str(sid):v for sid,v in zip(feature_file['sample_id'],feature_file['context_by_length'])}
    # Reconstruct from saved cycle arrays/metadata, not the feature extractor.
    for sid,row in samples.items():
        b=row['battery'];n=int(row['history_end_exclusive']);m=metadata[b]
        assert n>=50 and data[b]['cycle_valid'][n-50:n].all()
        duration=np.array([float(r['duration_s']) for r in m]);last=m[n-1]
        cumulative=np.cumsum([float(r['discharge_Ah']) for r in m])
        common=[n,(float(last['end_time_s'])-float(m[0]['start_time_s']))/86400,
                cumulative[n-1],duration[n-1],float(last['discharge_Ah']),float(last['temperature_missing_fraction'])]
        for j,length in enumerate(LENGTHS):
            v=duration[n-length:n];t=np.arange(length,dtype=float)
            slope=float(np.dot(t-t.mean(),v-v.mean())/np.square(t-t.mean()).sum())
            window=data[b]['x'][n-length:n]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                moments=[float(fn(window[:,:,ch])) for ch in (0,1,3) for fn in (np.nanmean,np.nanstd)]
            expected=np.array([*common,v.mean(),v.std(),slope,*moments],dtype=np.float32)
            # Near-zero slope can differ between polyfit and centered covariance.
            np.testing.assert_allclose(feature[sid][j],expected,rtol=2e-6,atol=1e-5,equal_nan=True)
        np.testing.assert_array_equal(feature[sid][0],data[b]['context'][n-1])
    pred_rows=read_csv(output/'predictions.csv')
    pred={(r['outer_test_battery'],r['sample_id'],r['method']):r for r in pred_rows}
    expected_keys={(r['battery'],sid,m) for sid,r in samples.items() for m in METHODS}
    assert set(pred)==expected_keys and len(pred)==len(pred_rows)==run['prediction_rows']
    oof_rows=read_csv(output/'oof_predictions.csv')
    oof={(r['outer_test_battery'],r['sample_id'],int(r['lookback'])):r for r in oof_rows}
    assert len(oof_rows)==len(oof)==len(samples)*3*3
    labels=read_csv(output/'labels.csv');labelmap={(r['outer_test_battery'],r['sample_id']):r for r in labels}
    assert len(labels)==len(labelmap)==len(samples)*3
    selection_rows=read_csv(output/'selections.csv')
    selections={(r['outer_test_battery'],r['sample_id'],r['method']):r for r in selection_rows}
    assert len(selections)==len(selection_rows)==len(samples)*2
    old={r['sample_id']:float(r['soh_pred']) for r in read_csv(baseline/'predictions.csv') if r['method']=='ContextOnlyRidge'}
    checked=0
    raw_for=lambda ids,j:np.array([feature[sid][j] for sid in ids])
    target_for=lambda ids:np.array([float(samples[sid]['soh']) for sid in ids])
    for outer in cells:
        split=json.loads((root/'folds'/f'test_{outer}__final'/'split.json').read_text())
        assert set(split['train_batteries'])==set(cells)-{outer} and split['test_batteries']==[outer]
        assert set(split['train']).isdisjoint(split['test'])
        assert all(samples[sid]['battery']!=outer for sid in split['train'])
        reconstructed={}
        for held in split['train_batteries']:
            inner=json.loads((root/'folds'/f'test_{outer}__val_{held}'/'split.json').read_text())
            assert set(inner['train_batteries'])==set(cells)-{outer,held}
            assert inner['validation_batteries']==[held] and inner['test_batteries']==[outer]
            assert set(inner['train']).isdisjoint(inner['validation'])
            values=[]
            for j,length in enumerate(LENGTHS):
                folder=output/'models'/f'test_{outer}'/f'oof_{held}'/f'lb_{length}'
                meta=json.loads((folder/'fit.json').read_text())
                assert meta['train_sample_ids']==inner['train'] and meta['train_batteries']==inner['train_batteries']
                assert meta['label_sample_ids']==inner['validation']
                model=audit_ridge(folder,raw_for(inner['train'],j),target_for(inner['train']),p['predictor']['l2']);checked+=1
                predicted=scaled(raw_for(inner['validation'],j),model)@model['weight']+model['bias']
                values.append(predicted)
                for sid,value in zip(inner['validation'],predicted):
                    row=oof[outer,sid,length]
                    assert row['heldout_battery']==held
                    near(float(row['soh_pred']),value);near(float(row['soh_true']),float(samples[sid]['soh']))
                    near(float(row['absolute_error']),abs(value-float(samples[sid]['soh'])))
            errors=np.abs(np.array(values).T-target_for(inner['validation'])[:,None])
            for sid,error in zip(inner['validation'],errors):
                reconstructed[sid]=error;r=labelmap[outer,sid]
                near(error,[float(r[f'error_lb{lb}']) for lb in LENGTHS])
                assert int(r['class_label'])==int(np.argmin(error))
                assert int(r['best_lookback'])==LENGTHS[np.argmin(error)]
                near(float(r['error_gap_to_second_best']),np.sort(error)[1]-min(error))
        assert set(reconstructed)==set(split['train'])
        bank={}
        for key,j,dim in [(lb,j,15) for j,lb in enumerate(LENGTHS)]+[('Usage3',0,3),('LastState6',0,6)]:
            name=f'lb_{key}' if isinstance(key,int) else key
            folder=output/'models'/f'test_{outer}'/'final'/name
            meta=json.loads((folder/'fit.json').read_text())
            assert meta['train_sample_ids']==split['train'] and meta['train_batteries']==split['train_batteries']
            bank[key]=audit_ridge(folder,raw_for(split['train'],j)[:,:dim],target_for(split['train']),p['predictor']['l2']);checked+=1
        folder=output/'models'/f'test_{outer}'/'selectors'
        selector=load(folder/'parameters.npz');training=load(folder/'training_data.npz')
        meta=json.loads((folder/'fit.json').read_text())
        assert meta['train_sample_ids']==split['train'] and meta['train_batteries']==split['train_batteries']
        assert list(training['sample_id'])==split['train']
        raw=raw_for(split['train'],0);errors=np.array([reconstructed[sid] for sid in split['train']])
        np.testing.assert_array_equal(raw,training['context_raw']);near(errors,training['errors'])
        labels=np.argmin(errors,axis=1);np.testing.assert_array_equal(labels,training['class_label'])
        mu,sd,active=scale_fit(raw);near(selector['mean'],mu);near(selector['std'],sd)
        np.testing.assert_array_equal(selector['active'],active);x=scaled(raw,selector)
        cls,targets=np.unique(labels,return_inverse=True)
        np.testing.assert_array_equal(cls,selector['class_labels'])
        if len(cls)>1:
            logits=x@selector['class_weight']+selector['class_bias']
            probability=np.exp(logits-logits.max(axis=1,keepdims=True));probability/=probability.sum(axis=1,keepdims=True)
            residual=probability-np.eye(len(cls))[targets]
            assert np.abs(x.T@residual/len(x)+p['classification_selector']['l2']*selector['class_weight']).max()<1e-5
            assert np.abs(residual.mean(axis=0)).max()<1e-5
        else:
            near(selector['class_weight'],0.);near(selector['class_bias'],0.)
        d=x.shape[1];design=np.column_stack((x,np.ones(len(x))))
        penalty=np.diag(np.r_[np.full(d,np.sqrt(len(x)*p['error_selector']['l2'])),0.])
        solution=np.linalg.lstsq(np.vstack((design,penalty)),np.vstack((errors,np.zeros((d+1,3)))),rcond=None)[0]
        near(solution[:-1],selector['error_weight']);near(solution[-1],selector['error_bias'])
        cell_labels=np.array([samples[sid]['battery'] for sid in split['train']])
        np.testing.assert_array_equal(cell_labels,training['battery'])
        macro=np.mean([errors[cell_labels==b].mean(axis=0) for b in split['train_batteries']],axis=0)
        assert int(selector['global_best_lb'])==LENGTHS[np.argmin(macro)]
        assert int(selector['majority_lb'])==LENGTHS[np.argmax(np.bincount(labels,minlength=3))]
        near(macro,meta['oof_cell_macro_mae'])
        for sid in split['test']:
            assert samples[sid]['battery']==outer
            expected={key:float(scaled(feature[sid][LENGTHS.index(key)] if isinstance(key,int) else feature[sid][0][:3 if key=='Usage3' else 6],model)@model['weight']+model['bias']) for key,model in bank.items()}
            for key,value in expected.items():
                name=f'FixedContext{key}' if isinstance(key,int) else key
                near(float(pred[outer,sid,name]['soh_pred']),value)
            near(expected[10],old[sid],atol=1e-12)
            x=scaled(feature[sid][0],selector)
            logits=x@selector['class_weight']+selector['class_bias']
            probability=np.exp(logits-logits.max());probability/=probability.sum()
            full=np.zeros(3);full[cls]=probability
            scores=x@selector['error_weight']+selector['error_bias']
            for method,score,chosen in (('Adaptive',full,LENGTHS[np.argmax(full)]),('ErrorAware',scores,LENGTHS[np.argmin(scores)])):
                selection=selections[outer,sid,method]
                near(score,[float(selection[f'score_lb{lb}']) for lb in LENGTHS])
                assert int(selection['selected_lookback'])==chosen and int(selection['negative_score_count'])==int((score<0).sum())
                near(float(pred[outer,sid,method]['soh_pred']),expected[chosen])
                assert int(pred[outer,sid,method]['lookback'])==chosen
            truth=float(samples[sid]['soh'])
            oracle=min(LENGTHS,key=lambda lb:abs(expected[lb]-truth))
            for method,lb in (('Oracle',oracle),('GlobalBestOOF',int(selector['global_best_lb'])),('LabelMajority',int(selector['majority_lb']))):
                near(float(pred[outer,sid,method]['soh_pred']),expected[lb]);assert int(pred[outer,sid,method]['lookback'])==lb
    assert checked==56
    for row in pred_rows:
        truth=float(samples[row['sample_id']]['soh'])
        near(float(row['soh_true']),truth);near(float(row['absolute_error']),abs(float(row['soh_pred'])-truth))
    def stats(rows):
        errors=np.array([float(r['soh_pred'])-float(r['soh_true']) for r in rows])
        return float(np.abs(errors).mean()),float(np.sqrt(np.mean(errors**2)))
    cell_stats={}
    metric_rows=read_csv(output/'metrics_by_battery.csv');assert len(metric_rows)==len(cells)*len(METHODS)
    for row in metric_rows:
        b,m=row['outer_test_battery'],row['method']
        records=[r for r in pred_rows if (r['outer_test_battery'],r['method'])==(b,m)]
        cell_stats[b,m]=stats(records);near([float(row['mae']),float(row['rmse'])],cell_stats[b,m])
        assert int(row['samples'])==len(records)
    for row in read_csv(output/'metrics_by_stage.csv'):
        records=[r for r in pred_rows if (r['outer_test_battery'],r['stage'],r['method'])==(row['outer_test_battery'],row['stage'],row['method'])]
        near([float(row['mae']),float(row['rmse'])],stats(records));assert int(row['samples'])==len(records)
    for row in read_csv(output/'paired_comparisons.csv'):
        bs=cells if row['scope']=='MACRO' else [row['scope']]
        near(float(row['mae_difference']),np.mean([cell_stats[b,row['method']][0]-cell_stats[b,row['baseline']][0] for b in bs]))
    for row in read_csv(output/'selection_diagnostics.csv'):
        b,sid,m=row['outer_test_battery'],row['sample_id'],row['method']
        errors=np.array([float(pred[b,sid,f'FixedContext{lb}']['absolute_error']) for lb in LENGTHS])
        chosen=int(row['selected_lookback']);best=LENGTHS[np.argmin(errors)]
        assert int(row['posthoc_best_lookback'])==best and int(row['matches_posthoc_best'])==int(chosen==best)
        near(float(row['regret']),errors[LENGTHS.index(chosen)]-errors.min())
        near(float(row['oracle_gap_to_second']),np.sort(errors)[1]-errors.min())
    timing=read_csv(output/'latency_samples.csv');grid=set()
    for row in timing:
        key=(row['outer_test_battery'],row['sample_id'],row['method'])
        assert key in pred and row['method']!='Oracle' and int(row['latency_ns'])>0
        assert row['lookback']==pred[key]['lookback'];grid.add((*key,int(row['repetition'])))
    expected_grid={(*key,r) for key in expected_keys if key[-1]!='Oracle' for r in range(p['benchmark']['rounds'])}
    assert grid==expected_grid and len(grid)==len(timing)==run['timing_rows']
    for row in read_csv(output/'summary.csv'):
        m=row['method'];near([float(row['battery_macro_mae']),float(row['battery_macro_rmse'])],np.mean([cell_stats[b,m] for b in cells],axis=0))
        records=[r for r in pred_rows if r['method']==m];near(float(row['pooled_mae']),stats(records)[0])
        lengths=[int(r['lookback']) for r in records if r['lookback']!='']
        if lengths:near(float(row['mean_lookback']),np.mean(lengths))
        if m!='Oracle':
            costs=[int(r['latency_ns'])/1e6 for r in timing if r['method']==m]
            near(float(row['latency_median_ms']),np.median(costs));near(float(row['latency_p95_ms']),np.percentile(costs,95))
    result=dict(status='PASS',protocol_sha256=sig,checked_ridge_models=checked,classifiers=4,error_selectors=4,
        unique_test_samples=len(samples),predictions=len(pred_rows),oof_predictions=len(oof_rows),timings=len(timing),
        checks=['frozen sources and artifacts','independent feature reconstruction; exact canonical C10',
                'cell-disjoint OOF teachers; new error targets and labels','train-only missingness/scaling',
                '56 ridge fits by independent augmented least-squares solution','classifier first-order optimality and observed classes',
                'independent error-selector fits','original Context Ridge reproduction','all prediction routing and posthoc Oracle',
                'cell/stage/macro metrics, paired differences and regret','complete same-run timing grid'],
        evidence='exploratory four previously inspected cells; deterministic fits, no artificial seed repeats')
    write_json(output/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    verify(parser.parse_args().output.resolve())
