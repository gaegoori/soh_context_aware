"""Independent causal feature, fit, exported-tree and result audit."""
from scripts.paths import PROJECT_ROOT, SCRIPTS, hash_matches
import argparse
from datetime import datetime
import json
from pathlib import Path
from importlib.metadata import version
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scripts.experiments.train_expanded_baselines import read,load,sha,signature,write_json

LENGTHS=(10,20,50)
SELECTORS=('Selector_Base_Linear','Selector_Change_Linear','Selector_Base_Forest','Selector_Change_Forest')
DIRECT=tuple(s.replace('Selector','Direct') for s in SELECTORS)
RF=dict(n_estimators=200,criterion='squared_error',max_depth=3,min_samples_split=2,
        min_samples_leaf=8,max_features=1.0,bootstrap=True,max_samples=None,random_state=42,n_jobs=1)


def near(a,b):np.testing.assert_allclose(a,b,rtol=1e-8,atol=1e-10,equal_nan=True)
def transform(raw,m):
    raw=np.asarray(raw,dtype=float)
    return np.where(m['active'],(np.where(np.isfinite(raw),raw,m['mean'])-m['mean'])/m['std'],0.)


def manual_predict(raw,m):
    x=transform(raw,m)
    if str(m['kind'])=='Linear':return x@m['weight']+m['bias']
    x=x.astype(np.float32);result=[]
    for row in x:
        values=[]
        for t in range(200):
            at=0;depth=0
            while m['left'][t,at]!=-1:
                at=int(m['left'][t,at] if row[int(m['feature'][t,at])]<=m['threshold'][t,at] else m['right'][t,at])
                depth+=1;assert depth<=3
            values.append(m['value'][t,at])
        result.append(np.mean(values,axis=0))
    return np.asarray(result)


def audit_model(folder,raw,y,kind):
    raw=np.asarray(raw,dtype=float);y=np.asarray(y,dtype=float)
    if y.ndim==1:y=y[:,None]
    m=load(folder/'parameters.npz');d=raw.shape[1]
    assert str(m['kind'])==kind
    active=np.isfinite(raw).any(axis=0);mu=np.zeros(d);sd=np.ones(d)
    mu[active]=np.nanmean(raw[:,active],axis=0);sd[active]=np.nanstd(raw[:,active],axis=0)
    sd=np.where(sd>1e-6,sd,1.)
    near(m['mean'],mu);near(m['std'],sd);np.testing.assert_array_equal(m['active'],active)
    x=transform(raw,m)
    if kind=='Linear':
        design=np.column_stack((x,np.ones(len(x))))
        penalty=np.diag(np.r_[np.full(d,np.sqrt(.1*len(x))),0.])
        w=np.linalg.lstsq(np.vstack((design,penalty)),np.vstack((y,np.zeros((d+1,y.shape[1])))),rcond=None)[0]
        near(w[:-1],m['weight']);near(w[-1],m['bias'])
    else:
        replay=RandomForestRegressor(**RF).fit(x,y[:,0] if y.shape[1]==1 else y)
        assert len(m['node_count'])==200
        for t,estimator in enumerate(replay.estimators_):
            tree=estimator.tree_;n=tree.node_count
            assert n==int(m['node_count'][t]) and tree.max_depth<=3
            assert estimator.random_state==int(m['tree_seed'][t])
            np.testing.assert_array_equal(tree.children_left,m['left'][t,:n])
            np.testing.assert_array_equal(tree.children_right,m['right'][t,:n])
            np.testing.assert_array_equal(tree.feature,m['feature'][t,:n])
            near(tree.threshold,m['threshold'][t,:n]);near(tree.value[:,:,0],m['value'][t,:n])
            # Independent bootstrap leaf means, rather than just library replay.
            draw=np.random.RandomState(int(m['tree_seed'][t])).randint(0,len(x),len(x),dtype=np.int32)
            routed={}
            for idx in draw:
                row=x[idx].astype(np.float32);at=0
                while m['left'][t,at]!=-1:
                    at=int(m['left'][t,at] if row[m['feature'][t,at]]<=m['threshold'][t,at] else m['right'][t,at])
                routed.setdefault(at,[]).append(int(idx))
            for node,indices in routed.items():
                near(np.mean(y[indices],axis=0),m['value'][t,node])
                assert len(set(indices))==int(m['node_samples'][t,node])>=8
                assert len(indices)==int(m['node_weight'][t,node])
        expected=replay.predict(x)
        if expected.ndim==1:expected=expected[:,None]
        near(manual_predict(raw,m),expected)
    return m


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'outputs/change_selector_v1')
    out=parser.parse_args().output.resolve();p=json.loads((out/'protocol.json').read_text())
    cells=p['development_batteries'];reserved=p['reserved_final_test_batteries']
    source=Path(p['source_baseline']);previous=Path(p['source_previous_selector']);root=Path(p['source_data_root'])
    assert len(cells)==8 and len(reserved)==4 and not set(cells)&set(reserved)
    assert p['models']['forest']==RF
    for name,digest in p['input_sha256'].items():
        assert hash_matches(name, digest),name
        n=Path(name).name
        assert n!='samples_final_test.csv'
        assert not any(n.startswith(c+'.') or n.startswith(c+'_') for c in reserved)
    for name,digest in p['code_sha256'].items():assert hash_matches(SCRIPTS/name, digest),name
    assert hash_matches(PROJECT_ROOT/'requirements-selector-extra.txt', p['requirements_sha256'])
    for name in ('numpy','scipy','scikit-learn','joblib','threadpoolctl'):assert version(name)==p['runtime'][name]
    frozen=json.loads((out/'protocol_frozen.json').read_text());fitted=json.loads((out/'fitting_complete.json').read_text())
    run=json.loads((out/'run_summary.json').read_text());manifest=json.loads((out/'artifact_manifest.json').read_text())
    for d in (frozen,fitted,run,manifest):assert d['protocol_sha256']==signature(p)
    assert datetime.fromisoformat(frozen['frozen_utc'])<=datetime.fromisoformat(fitted['completed_utc'])<=datetime.fromisoformat(run['completed_utc'])
    for d in (fitted,manifest):
        for name,digest in d['files_sha256'].items():assert hash_matches(out/name, digest),name
    assert not run['final_test_payload_loaded'] and not run['final_test_evaluated']
    samples=read(root/'samples_development.csv');lookup={r['sample_id']:r for r in samples}
    f=load(out/'features.npz');cached=load(source/'features.npz')
    assert len(lookup)==len(samples)==len(f['sample_id'])==155
    assert {r['battery'] for r in samples}==set(cells)
    candidate={str(s):v for s,v in zip(cached['sample_id'],cached['context_by_length'])}
    base={str(s):v for s,v in zip(f['sample_id'],f['base'])}
    change={str(s):v for s,v in zip(f['sample_id'],f['changes'])}
    aug={str(s):v for s,v in zip(f['sample_id'],f['augmented'])}
    assert set(base)==set(change)==set(aug)==set(lookup)
    for cell in cells:
        meta=read(root/f'{cell}_cycles.csv')
        with np.load(root/f'{cell}.npz',allow_pickle=False) as d:x=d['x'].copy();valid=d['cycle_valid'].copy()
        for sid,row in lookup.items():
            if row['battery']!=cell:continue
            n=int(row['history_end_exclusive']);assert n>=50 and valid[n-50:n].all()
            assert float(meta[n-1]['end_time_s'])<float(row['target_start_time_s'])
            expected=[]
            for ch in (0,1,3):
                a=x[n-10:n-5,:,ch].ravel();b=x[n-5:n,:,ch].ravel()
                av=a[np.isfinite(a)];bv=b[np.isfinite(b)]
                expected.append(float(np.mean(bv,dtype=np.float64)-np.mean(av,dtype=np.float64)) if len(av) and len(bv) else np.nan)
            for key in ('duration_s','discharge_Ah','temperature_missing_fraction'):
                before=[float(r[key]) for r in meta[n-10:n-5]];recent=[float(r[key]) for r in meta[n-5:n]]
                expected.append(np.mean(recent)-np.mean(before))
            np.testing.assert_allclose(change[sid],np.array(expected,dtype=np.float32),rtol=1e-4,atol=1e-5,equal_nan=True)
            np.testing.assert_array_equal(base[sid],candidate[sid][0])
            np.testing.assert_array_equal(aug[sid],np.r_[base[sid],change[sid]])
    features={'Base':base,'Change':aug}
    ids=lambda cs:[r['sample_id'] for r in samples if r['battery'] in cs]
    labels={(r['validation_battery'],r['sample_id']):r for r in read(source/'oof_labels.csv')}
    defaults={r['validation_battery']:int(r['selected_lookback']) for r in read(source/'global_choices.csv')}
    prediction_rows=read(out/'predictions.csv')
    predictions={(r['battery'],r['sample_id'],r['method']):r for r in prediction_rows}
    decisions={(r['sample_id'],r['method']):r for r in read(out/'decisions.csv')}
    assert len(predictions)==len(prediction_rows)==155*len(p['methods'])==2015
    assert len(decisions)==620
    model_count=0
    for outer in cells:
        fit_cells=[c for c in cells if c!=outer];ss=ids(fit_cells);val=ids([outer])
        y=np.array([float(lookup[s]['soh']) for s in ss])
        errors=np.array([[float(labels[outer,s][f'error_lb{L}']) for L in LENGTHS] for s in ss])
        bank=[load(source/'models'/f'validation_{outer}'/'candidates'/f'lb_{L}'/'parameters.npz') for L in LENGTHS]
        values=np.column_stack([transform(np.array([candidate[s][j] for s in val]),m)@m['weight']+m['bias'] for j,m in enumerate(bank)])
        for method in (*SELECTORS,*DIRECT):
            role,fs,kind=method.split('_');folder=out/'models'/f'validation_{outer}'/method
            meta=json.loads((folder/'fit.json').read_text())
            assert meta['train_batteries']==fit_cells and meta['train_sample_ids']==ss
            assert meta['validation_battery']==outer and meta['role']==role
            assert meta['feature_set']==fs and meta['kind']==kind
            assert meta['oof_validation_filter']==(outer if role=='Selector' else None)
            m=audit_model(folder,np.array([features[fs][s] for s in ss]),errors if role=='Selector' else y,kind);model_count+=1
            outputs=manual_predict(np.array([features[fs][s] for s in val]),m)
            for i,s in enumerate(val):
                r=predictions[outer,s,method]
                if role=='Selector':
                    selected=int(np.argmin(outputs[i]));pred=values[i,selected]
                    assert int(r['selected_lookback'])==LENGTHS[selected]
                    d=decisions[s,method];assert int(d['selected_lookback'])==LENGTHS[selected]
                    near([float(d[f'estimated_error_lb{L}']) for L in LENGTHS],outputs[i])
                else:pred=outputs[i,0];assert r['selected_lookback']==''
                near(float(r['soh_pred']),pred)
        for i,s in enumerate(val):
            truth=float(lookup[s]['soh']);default=defaults[outer];err=np.abs(values[i]-truth)
            for method in p['methods']:
                r=predictions[outer,s,method]
                if method not in (*SELECTORS,*DIRECT):
                    L=int(method.replace('FixedContext','')) if method.startswith('FixedContext') else default if method=='GlobalBestOOF' else LENGTHS[int(np.argmin(err))]
                    assert int(r['selected_lookback'])==L;near(float(r['soh_pred']),values[i,LENGTHS.index(L)])
                near(float(r['soh_true']),truth);actual=abs(float(r['soh_pred'])-truth)
                near(float(r['absolute_error']),actual);near(float(r['squared_error']),actual**2)
                near(float(r['improvement_over_default']),err[LENGTHS.index(default)]-actual)
        print(outer,'features / eight model fits / tree traversal / predictions PASS',flush=True)
    assert model_count==64
    # Reproduce both fixed candidates and the previous linear error Selector.
    for r in read(source/'predictions.csv'):
        near(float(predictions[r['battery'],r['sample_id'],r['method']]['soh_pred']),float(r['soh_pred']))
        if r['method']=='FixedContext10':
            near(float(predictions[r['battery'],r['sample_id'],'Direct_Base_Linear']['soh_pred']),float(r['soh_pred']))
    for r in read(previous/'predictions.csv'):
        if r['method']=='ErrorAware':
            current=predictions[r['battery'],r['sample_id'],'Selector_Base_Linear']
            assert current['selected_lookback']==r['selected_lookback']
            near(float(current['soh_pred']),float(r['soh_pred']))
    metrics=read(out/'metrics_by_battery.csv');summary=read(out/'summary.csv')
    for method in p['methods']:
        mm=[];total_help=0;total_harm=0;total_switches=0
        for cell in cells:
            rr=[predictions[cell,s,method] for s in ids([cell])]
            e=np.array([float(r['absolute_error']) for r in rr]);gain=np.array([float(r['improvement_over_default']) for r in rr])
            row=next(r for r in metrics if r['method']==method and r['battery']==cell)
            near(float(row['mae']),e.mean());near(float(row['rmse']),np.sqrt(np.mean(e**2)))
            near(float(row['mean_improvement_over_default']),gain.mean())
            assert int(row['improved_samples'])==int((gain>1e-12).sum())
            assert int(row['worsened_samples'])==int((gain < -1e-12).sum())
            total_help+=int(row['improved_samples']);total_harm+=int(row['worsened_samples'])
            if method not in DIRECT:
                near(float(row['mean_lookback']),np.mean([int(r['selected_lookback']) for r in rr]))
                assert int(row['switches'])==sum(r['selected_lookback']!=r['default_lookback'] for r in rr)
                total_switches+=int(row['switches'])
            else:assert row['mean_lookback']==row['switches']==''
            mm.append(row)
        row=next(r for r in summary if r['method']==method)
        near(float(row['battery_macro_mae']),np.mean([float(r['mae']) for r in mm]))
        near(float(row['battery_macro_rmse']),np.mean([float(r['rmse']) for r in mm]))
        assert int(row['cells_improved'])==sum(float(r['mean_improvement_over_default'])>1e-12 for r in mm)
        assert int(row['cells_worsened'])==sum(float(r['mean_improvement_over_default']) < -1e-12 for r in mm)
        assert int(row['improved_samples'])==total_help and int(row['worsened_samples'])==total_harm
        if method not in DIRECT:
            assert int(row['switches'])==total_switches
            near(float(row['mean_lookback']),np.mean([float(r['mean_lookback']) for r in mm]))
    for r in read(out/'metrics_by_condition.csv'):
        sub=[m for m in metrics if m['method']==r['method'] and m['condition']==r['condition']]
        near(float(r['battery_macro_mae']),np.mean([float(m['mae']) for m in sub]))
    contrasts=read(out/'contrasts_by_battery.csv')
    for r in contrasts:
        m={s['method']:float(s['mae']) for s in metrics if s['battery']==r['battery']}
        near(float(r['mae_difference']),m[r['method_a']]-m[r['method_b']])
    for r in read(out/'contrasts_summary.csv'):
        values=[float(c['mae_difference']) for c in contrasts if c['method_a']==r['method_a'] and c['method_b']==r['method_b']]
        assert len(values)==8;near(float(r['macro_mae_difference']),np.mean(values))
        assert int(r['cells_a_better'])==sum(v < -1e-12 for v in values)
        assert int(r['cells_a_worse'])==sum(v>1e-12 for v in values)
    result=dict(status='PASS',protocol_sha256=signature(p),models_verified=64,
        ridge_models=32,forest_models=32,trees_verified=6400,development_samples=155,prediction_rows=2015,
        final_test_evaluated=False,
        checks=['Source artifacts and frozen code unchanged','Past-only signed 5-versus-5 changes independently reconstructed',
            'Training-only scaling, inactive missing columns, and seven/one battery roles',
            'Independent augmented least-squares Ridge solutions',
            'Fixed RF settings replay and independent bootstrap leaf means',
            'Independent exported-tree traversal and all predictions',
            'Original Predictor and prior linear Selector predictions reproduced',
            'Cell-level metrics, operating groups, and predeclared contrasts',
            'No final-test payload or final-test evaluation'])
    write_json(out/'verification.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
