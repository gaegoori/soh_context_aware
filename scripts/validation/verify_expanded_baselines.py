"""Independent reconstruction of development features, Ridge fits and metrics."""
from scripts.paths import PROJECT_ROOT, SCRIPTS, resolve_file, hash_matches
import argparse
import csv
import hashlib
import json
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np

LENGTHS=(10,20,50)
METHODS=('FixedContext10','FixedContext20','FixedContext50','GlobalBestOOF','Oracle')


def read(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(resolve_file(path).read_bytes()).hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def near(a,b):np.testing.assert_allclose(a,b,rtol=1e-8,atol=1e-10,equal_nan=True)
def transform(raw,m):
    return np.where(m['active'],(np.where(np.isfinite(raw),raw,m['mean'])-m['mean'])/m['std'],0.)


def audit_fit(folder,raw,y):
    raw=np.asarray(raw,dtype=np.float64)
    m=load(folder/'parameters.npz');active=np.isfinite(raw).any(axis=0)
    mu=np.zeros(raw.shape[1]);sd=np.ones(raw.shape[1])
    mu[active]=np.nanmean(raw[:,active],axis=0);sd[active]=np.nanstd(raw[:,active],axis=0)
    sd=np.where(sd>1e-6,sd,1.)
    near(mu,m['mean']);near(sd,m['std']);np.testing.assert_array_equal(active,m['active'])
    x=np.column_stack((transform(raw,m),np.ones(len(raw))))
    penalty=np.diag(np.r_[np.full(raw.shape[1],np.sqrt(.1*len(raw))),0.])
    w=np.linalg.lstsq(np.vstack((x,penalty)),np.r_[y,np.zeros(raw.shape[1]+1)],rcond=None)[0]
    near(w[:-1],m['weight']);near(w[-1],m['bias'])
    return m


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'outputs/expanded_predictor_baseline_v1')
    args=parser.parse_args();out=args.output.resolve()
    p=json.loads((out/'protocol.json').read_text())
    signature=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    root=Path(p['source_data_root']);cells=p['development_batteries'];reserved=set(p['reserved_final_test_batteries'])
    samples=read(root/'samples_development.csv');by_id={r['sample_id']:r for r in samples}
    assert len(by_id)==len(samples)==155 and {r['battery'] for r in samples}==set(cells)
    assert not reserved&set(cells) and len(cells)==8
    for filename,digest in p['input_sha256'].items():
        assert hash_matches(filename, digest)
        assert Path(filename).name not in ('samples.csv','samples_final_test.csv')
        assert not any(Path(filename).name.startswith(cell+'.') or Path(filename).name.startswith(cell+'_') for cell in reserved)
    for filename,digest in p['code_sha256'].items():assert hash_matches(SCRIPTS/filename, digest)
    frozen=json.loads((out/'protocol_frozen.json').read_text())
    fitted=json.loads((out/'fitting_complete.json').read_text())
    run=json.loads((out/'run_summary.json').read_text())
    artifacts=json.loads((out/'artifact_manifest.json').read_text())
    assert all(v['protocol_sha256']==signature for v in (frozen,fitted,run,artifacts))
    assert datetime.fromisoformat(frozen['frozen_utc'])<=datetime.fromisoformat(fitted['completed_utc'])<=datetime.fromisoformat(run['completed_utc'])
    assert not run['selector_trained'] and not run['final_test_payload_loaded'] and not run['final_test_evaluated']
    for mapping in (fitted['files_sha256'],artifacts['files_sha256']):
        for name,digest in mapping.items():assert hash_matches(out/name, digest)
    f=load(out/'features.npz')
    assert len(f['sample_id'])==len(set(f['sample_id']))==len(samples)
    assert set(f['sample_id'])==set(by_id)
    feature={str(sid):v for sid,v in zip(f['sample_id'],f['context_by_length'])}
    for cell in cells:
        d=load(root/f'{cell}.npz');meta=read(root/f'{cell}_cycles.csv')
        duration=np.array([float(r['duration_s']) for r in meta]);ah=np.cumsum([float(r['discharge_Ah']) for r in meta])
        for sid,row in by_id.items():
            if row['battery']!=cell:continue
            n=int(row['history_end_exclusive']);last=meta[n-1]
            assert n>=50 and d['cycle_valid'][n-50:n].all()
            assert d['cycle_end_time_s'][n-1]<float(row['target_start_time_s'])
            assert float(row['reference_delay_s'])<=120*3600
            common=[n,(float(last['end_time_s'])-float(meta[0]['start_time_s']))/86400,ah[n-1],
                    duration[n-1],float(last['discharge_Ah']),float(last['temperature_missing_fraction'])]
            for j,L in enumerate(LENGTHS):
                durations=duration[n-L:n];t=np.arange(L,dtype=float)
                slope=np.dot(t-t.mean(),durations-durations.mean())/np.square(t-t.mean()).sum()
                window=d['x'][n-L:n]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    moments=[float(fn(window[:,:,ch])) for ch in (0,1,3) for fn in (np.nanmean,np.nanstd)]
                expected=np.array([*common,durations.mean(),durations.std(),slope,*moments],dtype=np.float32)
                np.testing.assert_allclose(feature[sid][j],expected,rtol=2e-6,atol=1e-5,equal_nan=True)
            np.testing.assert_array_equal(feature[sid][0],d['context'][n-1])
    X=lambda ids,j:np.array([feature[sid][j] for sid in ids])
    Y=lambda ids:np.array([float(by_id[sid]['soh']) for sid in ids])
    ids=lambda cellset:[r['sample_id'] for r in samples if r['battery'] in cellset]
    oof_rows=read(out/'oof_predictions.csv')
    oof={(r['validation_battery'],r['sample_id'],int(r['lookback'])):r for r in oof_rows}
    label_rows=read(out/'oof_labels.csv');labels={(r['validation_battery'],r['sample_id']):r for r in label_rows}
    assert len(oof)==len(oof_rows)==155*7*3 and len(labels)==len(label_rows)==155*7
    expected_oof={(outer,sid,L) for outer in cells for sid,r in by_id.items() if r['battery']!=outer for L in LENGTHS}
    assert set(oof)==expected_oof
    choices={r['validation_battery']:r for r in read(out/'global_choices.csv')}
    predictions=read(out/'predictions.csv');pred={(r['battery'],r['sample_id'],r['method']):r for r in predictions}
    assert len(pred)==len(predictions)==155*5
    assert set(pred)=={(r['battery'],r['sample_id'],m) for r in samples for m in METHODS}
    count=0
    for outer in cells:
        fit_cells=[c for c in cells if c!=outer];fit_ids=ids(fit_cells);val_ids=ids([outer])
        split=json.loads((out/f'split_validation_{outer}.json').read_text())
        assert split['train_batteries']==fit_cells and split['validation_batteries']==[outer]
        assert split['train_sample_ids']==fit_ids and split['validation_sample_ids']==val_ids
        folder=out/'models'/f'validation_{outer}';means=[]
        for label_cell in fit_cells:
            train_cells=[c for c in fit_cells if c!=label_cell];train_ids=ids(train_cells);label_ids=ids([label_cell]);values=[]
            for j,L in enumerate(LENGTHS):
                path=folder/f'oof_{label_cell}'/f'lb_{L}'
                meta=json.loads((path/'fit.json').read_text())
                assert meta['train_batteries']==train_cells and meta['train_sample_ids']==train_ids
                assert meta['label_sample_ids']==label_ids and outer not in train_cells and label_cell not in train_cells
                model=audit_fit(path,X(train_ids,j),Y(train_ids));count+=1
                vv=transform(X(label_ids,j),model)@model['weight']+model['bias'];values.append(vv)
                for sid,value in zip(label_ids,vv):
                    row=oof[outer,sid,L]
                    assert row['label_battery']==label_cell
                    near(float(row['soh_pred']),value);near(float(row['absolute_error']),abs(value-float(by_id[sid]['soh'])))
            errors=np.abs(np.array(values).T-Y(label_ids)[:,None]);means.append(errors.mean(axis=0))
            for sid,err in zip(label_ids,errors):
                row=labels[outer,sid];near([float(row[f'error_lb{L}']) for L in LENGTHS],err)
                assert int(row['best_lookback'])==LENGTHS[int(np.argmin(err))]
                near(float(row['error_gap_to_second_best']),np.sort(err)[1]-min(err))
        means=np.mean(means,axis=0);chosen=LENGTHS[int(np.argmin(means))]
        assert int(choices[outer]['selected_lookback'])==chosen
        near(means,[float(choices[outer][f'inner_oof_mae_lb{L}']) for L in LENGTHS])
        vals=[]
        for j,L in enumerate(LENGTHS):
            path=folder/'candidates'/f'lb_{L}';meta=json.loads((path/'fit.json').read_text())
            assert meta['train_batteries']==fit_cells and meta['train_sample_ids']==fit_ids
            model=audit_fit(path,X(fit_ids,j),Y(fit_ids));count+=1
            vals.append(transform(X(val_ids,j),model)@model['weight']+model['bias'])
        for sid,values in zip(val_ids,np.array(vals).T):
            truth=float(by_id[sid]['soh']);errors=np.abs(values-truth)
            for method in METHODS:
                L=int(method.replace('FixedContext','')) if method.startswith('FixedContext') else chosen if method=='GlobalBestOOF' else LENGTHS[int(np.argmin(errors))]
                row=pred[outer,sid,method];value=values[LENGTHS.index(L)]
                assert int(row['selected_lookback'])==L
                near(float(row['soh_true']),truth);near(float(row['soh_pred']),value)
                near(float(row['absolute_error']),abs(value-truth));near(float(row['squared_error']),(value-truth)**2)
        print(outer,'fit/OOF/prediction PASS',flush=True)
    assert count==192
    metrics=read(out/'metrics_by_battery.csv');summary=read(out/'summary.csv')
    for method in METHODS:
        maes=[];rmses=[]
        for cell in cells:
            subset=[r for r in predictions if r['battery']==cell and r['method']==method]
            err=np.array([float(r['soh_pred'])-float(r['soh_true']) for r in subset])
            mae=np.abs(err).mean();rmse=np.sqrt(np.square(err).mean());maes.append(mae);rmses.append(rmse)
            row=next(r for r in metrics if r['battery']==cell and r['method']==method)
            near(float(row['mae']),mae);near(float(row['rmse']),rmse)
        row=next(r for r in summary if r['method']==method)
        near(float(row['battery_macro_mae']),np.mean(maes));near(float(row['battery_macro_rmse']),np.mean(rmses))
    result=dict(status='PASS',protocol_sha256=signature,models_verified=count,
        development_samples=155,inner_oof_predictions=len(oof_rows),prediction_rows=len(predictions),
        final_test_evaluated=False,selector_trained=False,
        checks=['Frozen code and data hashes','No final-test payload in experiment inputs',
            'Independent feature reconstruction','Training-only scalers',
            'All 192 Ridge solutions independently solved by augmented least squares',
            'Six/one/one battery roles for OOF labels, seven/one for validation',
            'Inner-only fixed-length selection','All development predictions and equal-cell aggregation'])
    (out/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
