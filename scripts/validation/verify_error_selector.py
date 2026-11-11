"""Independently audit OOF targets, train-only fits, routing and result aggregates."""
from scripts.paths import hash_matches
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.train_lookback_oof import CyclePredictor, read_csv, sha256, write_json
from scripts.experiments.train_error_selector import DEFAULT_OUTPUT, METHODS, check_inputs, signature


def near(a, b, atol=1e-10):
    np.testing.assert_allclose(a, b, atol=atol, rtol=1e-8)


def verify(output):
    p = json.loads((output/'protocol.json').read_text())
    check_inputs(p)
    sig = signature(p)
    run = json.loads((output/'run_summary.json').read_text())
    frozen = json.loads((output/'protocol_frozen.json').read_text())
    fitted = json.loads((output/'fitting_complete.json').read_text())
    manifest = json.loads((output/'artifact_manifest.json').read_text())
    assert run['status']=='EVALUATION_COMPLETE' and not run['current_run_test_tuning']
    assert all(record['protocol_sha256']==sig for record in (run,frozen,fitted,manifest))
    assert datetime.fromisoformat(frozen['frozen_utc']) <= datetime.fromisoformat(fitted['completed_utc']) <= datetime.fromisoformat(run['completed_utc'])
    for name, digest in manifest['files_sha256'].items():
        assert hash_matches(output/name, digest), name
    for name, digest in fitted['model_sha256'].items():
        assert hash_matches(output/name, digest), name
    root, baseline, labels = (Path(p[k]) for k in ('source_data_root','source_baseline_root','source_labels_root'))
    samples = {r['sample_id']:r for r in read_csv(root/'samples.csv')}
    batteries, seeds, lengths = p['batteries'], p['seeds'], p['lookbacks']
    predictions = read_csv(output/'predictions.csv')
    key = lambda r:(r['outer_test_battery'],int(r['seed']),r['sample_id'],r['method'])
    pred = {key(r):r for r in predictions}
    expected = {(r['battery'],s,sid,m) for sid,r in samples.items() for s in seeds for m in METHODS}
    assert len(pred)==len(predictions)==len(expected)==run['prediction_rows'] and set(pred)==expected
    old = {key(r):r for r in read_csv(baseline/'predictions.csv')}
    selected_rows = read_csv(output/'selections.csv')
    selected = {(r['outer_test_battery'],r['sample_id']):r for r in selected_rows}
    assert len(selected)==len(selected_rows)==len(samples)==run['unique_test_samples']
    assert set(selected)=={(r['battery'],sid) for sid,r in samples.items()}
    oof_rows = read_csv(labels/'predictions.csv')
    oof = {(r['outer_test_battery'],r['sample_id'],int(r['seed']),int(r['lookback'])):r for r in oof_rows}
    assert len(oof)==len(oof_rows)
    canonical={}
    for b in batteries:
        with np.load(root/f'{b}.npz',allow_pickle=False) as d:
            canonical[b] = {k:d[k].copy() for k in ('context','x')}
    torch.set_num_threads(1)
    max_residual, checked_models = 0., 0
    for outer in batteries:
        split=json.loads((root/'folds'/f'test_{outer}__final'/'split.json').read_text())
        scale=json.loads((root/'folds'/f'test_{outer}__final'/'scalers.json').read_text())
        train=[samples[sid] for sid in split['train']]
        test=[samples[sid] for sid in split['test']]
        assert set(split['train_batteries'])==set(batteries)-{outer}
        assert set(split['train']).isdisjoint(split['test'])
        assert all(r['battery']!=outer for r in train) and all(r['battery']==outer for r in test)
        fit=json.loads((output/'models'/f'test_{outer}'/'fit.json').read_text())
        assert fit['train_sample_ids']==split['train'] and fit['train_batteries']==split['train_batteries']
        with np.load(output/'models'/f'test_{outer}'/'parameters.npz',allow_pickle=False) as d:
            params={k:d[k].copy() for k in d.files}
        raw=np.array([canonical[r['battery']]['context'][int(r['history_end_exclusive'])-1] for r in train],dtype=np.float64)
        mu,std=np.nanmean(raw,axis=0),np.nanstd(raw,axis=0)
        std=np.where(std>1e-6,std,1.)
        near(params['context_mean'],mu);near(params['context_std'],std)
        x=(np.where(np.isfinite(raw),raw,mu)-mu)/std
        target=[]
        for row in train:
            inner=json.loads((root/'folds'/f"test_{outer}__val_{row['battery']}"/'split.json').read_text())
            assert inner['validation_batteries']==[row['battery']] and inner['test_batteries']==[outer]
            assert set(inner['train_batteries'])==set(batteries)-{outer,row['battery']}
            assert row['sample_id'] in inner['validation'] and row['sample_id'] not in inner['train']
            errors=[]
            for length in lengths:
                values=[]
                for seed in seeds:
                    r=oof[outer,row['sample_id'],seed,length]
                    assert r['heldout_battery']==row['battery'] and r['fold']==f"test_{outer}__val_{row['battery']}"
                    near(float(r['soh_true']),float(row['soh']),atol=1e-6)
                    value=abs(float(r['soh_pred'])-float(r['soh_true']))
                    near(value,float(r['absolute_error']))
                    values.append(value)
                errors.append(np.mean(values))
            target.append(errors)
        target=np.array(target)
        with np.load(labels/'selector_data'/f'test_{outer}'/'dataset.npz',allow_pickle=False) as d:
            idx={str(s):j for j,s in enumerate(d['sample_id'])}
            near(target,d['mean_errors'][[idx[r['sample_id']] for r in train]])
        # Solve an augmented least-squares system independently of the normal-equation fitter.
        design=np.column_stack((x,np.ones(len(x))))
        penalty=np.diag(np.r_[np.full(15,np.sqrt(len(x)*p['selector']['l2'])),0.])
        solved=np.linalg.lstsq(np.vstack((design,penalty)),np.vstack((target,np.zeros((16,3)))),rcond=None)[0]
        near(solved[:-1],params['error_weight']);near(solved[-1],params['error_bias'])
        residual=x@params['error_weight']+params['error_bias']-target
        near(x.T@residual/len(x)+p['selector']['l2']*params['error_weight'],0.)
        near(residual.mean(axis=0),0.)
        max_residual=max(max_residual,float(np.abs(solved-np.vstack((params['error_weight'],params['error_bias']))).max()))
        for row in test:
            sid,n=row['sample_id'],int(row['history_end_exclusive'])
            raw=canonical[outer]['context'][n-1].astype(np.float64)
            context=(np.where(np.isfinite(raw),raw,mu)-mu)/std
            estimated=context@params['error_weight']+params['error_bias']
            selection=selected[outer,sid]
            near(estimated,[float(selection[f'estimated_error_lb{lb}']) for lb in lengths])
            chosen=lengths[int(np.argmin(estimated))]
            assert int(selection['selected_lookback'])==chosen
            assert int(selection['negative_score_count'])==int((estimated<0).sum())
        for seed in seeds:
            for length in lengths:
                checkpoint=torch.load(baseline/'models'/f'test_{outer}'/f'seed_{seed}'/f'lb_{length}'/'checkpoint.pt',map_location='cpu',weights_only=True)
                assert checkpoint['train_sample_ids']==split['train']
                model=CyclePredictor();model.load_state_dict(checkpoint['state_dict']);model.eval()
                mean,sd=np.array(scale['channel_mean'],dtype=np.float32),np.array(scale['channel_std'],dtype=np.float32)
                for row in test:
                    sid,n=row['sample_id'],int(row['history_end_exclusive'])
                    window=canonical[outer]['x'][n-length:n]
                    normalized=((np.where(np.isfinite(window),window,mean)-mean)/sd).astype(np.float32)
                    with torch.inference_mode(): value=float(model(torch.from_numpy(normalized[None])).item())
                    near(value,float(pred[outer,seed,sid,f'Fixed{length}']['soh_pred']),atol=1e-7)
                checked_models+=1
    stats={}
    for r in predictions:
        k=key(r);truth=float(samples[r['sample_id']]['soh'])
        near(float(r['soh_true']),truth)
        near(float(r['absolute_error']),abs(float(r['soh_pred'])-truth))
        if r['method']=='ErrorAware':
            lb=int(selected[r['outer_test_battery'],r['sample_id']]['selected_lookback'])
            assert int(r['lookback'])==lb
            near(float(r['soh_pred']),float(pred[k[:3]+(f'Fixed{lb}',)]['soh_pred']),atol=1e-7)
        else:
            near(float(r['soh_pred']),float(old[k]['soh_pred']),atol=1e-7)
            assert r['lookback']==old[k]['lookback']
        if r['method']=='Oracle':
            errors=[float(pred[k[:3]+(f'Fixed{lb}',)]['absolute_error']) for lb in lengths]
            near(float(r['absolute_error']),min(errors))
            assert int(r['lookback'])==lengths[int(np.argmin(errors))]
    for b in batteries:
        for s in seeds:
            for m in METHODS:
                rows=[r for r in predictions if (r['outer_test_battery'],int(r['seed']),r['method'])==(b,s,m)]
                error=np.array([float(r['soh_pred'])-float(r['soh_true']) for r in rows])
                stats[b,s,m]=(float(np.abs(error).mean()),float(np.sqrt(np.mean(error**2))))
    seed_rows=read_csv(output/'metrics_per_seed.csv')
    assert len(seed_rows)==len(batteries)*len(seeds)*len(METHODS)
    for r in seed_rows:
        near([float(r['mae']),float(r['rmse'])],stats[r['outer_test_battery'],int(r['seed']),r['method']])
    for r in read_csv(output/'metrics_by_battery.csv'):
        b,m=r['outer_test_battery'],r['method']
        a=np.array([stats[b,s,m] for s in seeds])
        near([float(r['mae_mean']),float(r['rmse_mean'])],a.mean(axis=0))
        near(float(r['mae_seed_sd']),a[:,0].std(ddof=1))
        near(float(r['excess_mae_vs_oracle']),np.mean([stats[b,s,m][0]-stats[b,s,'Oracle'][0] for s in seeds]))
    for r in read_csv(output/'paired_comparisons.csv'):
        bs=batteries if r['scope']=='MACRO' else [r['scope']]
        delta=[np.mean([stats[b,s,'ErrorAware'][0]-stats[b,s,r['baseline']][0] for b in bs]) for s in seeds]
        near(float(r['error_aware_minus_baseline_mae']),np.mean(delta))
        near(float(r['paired_delta_seed_sd']),np.std(delta,ddof=1))
    for r in read_csv(output/'selection_diagnostics.csv'):
        b,sid=r['outer_test_battery'],r['sample_id']
        errors=[np.mean([float(pred[b,s,sid,f'Fixed{lb}']['absolute_error']) for s in seeds]) for lb in lengths]
        lb=int(r['selected_lookback'])
        near(errors,[float(r[f'actual_seed_mean_error_lb{lb}']) for lb in lengths])
        assert int(r['posthoc_seed_mean_best_lb'])==lengths[int(np.argmin(errors))]
        assert int(r['matches_posthoc_best'])==int(lb==int(r['posthoc_seed_mean_best_lb']))
        near(float(r['excess_mae_vs_seed_mean_oracle']),errors[lengths.index(lb)]-min(errors))
    timing=read_csv(output/'latency_samples.csv')
    timing_keys=set()
    for r in timing:
        k=key(r)
        assert r['method']!='Oracle' and int(r['latency_ns'])>0 and r['lookback']==pred[k]['lookback']
        timing_keys.add((*k,int(r['repetition'])))
    want_timing={(*k,i) for k in expected if k[-1]!='Oracle' for i in range(p['benchmark']['rounds'])}
    assert timing_keys==want_timing and len(timing_keys)==len(timing)==run['timing_rows']
    for r in read_csv(output/'summary.csv'):
        m=r['method'];a=np.array([[stats[b,s,m][0] for b in batteries] for s in seeds]).mean(axis=1)
        near(float(r['battery_macro_mae']),a.mean());near(float(r['macro_mae_seed_sd']),a.std(ddof=1))
        near(float(r['battery_macro_rmse']),np.mean([stats[b,s,m][1] for b in batteries for s in seeds]))
        near(float(r['excess_mae_vs_oracle']),np.mean([stats[b,s,m][0]-stats[b,s,'Oracle'][0] for b in batteries for s in seeds]))
        if m!='ContextOnlyRidge':
            near(float(r['mean_lookback']),np.mean([int(v['lookback']) for v in predictions if v['method']==m]))
        if m!='Oracle':
            cost=[int(v['latency_ns'])/1e6 for v in timing if v['method']==m]
            near(float(r['latency_median_ms']),np.median(cost));near(float(r['latency_p95_ms']),np.percentile(cost,95))
    result=dict(status='PASS',protocol_sha256=sig,checked_predictors=checked_models,
        checked_selectors=len(batteries),unique_test_samples=len(samples),prediction_rows=len(predictions),
        timing_rows=len(timing),independent_fit_max_abs_difference=max_residual,
        checks=['unchanged original inputs/checkpoints/source hashes','cell-disjoint OOF error targets reconstructed from original seed predictions',
                'train-only context scaling and independent augmented least-squares fit','single selected fixed-predictor routing',
                'all original predictions reproduced','all four selectors fitted before evaluation',
                'independent per-cell/seed, paired and macro metric calculation','full same-run timing grid'],
        evidence='Exploratory: four previously inspected cells. Seed SD is not population uncertainty.')
    write_json(output/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    verify(parser.parse_args().output.resolve())
