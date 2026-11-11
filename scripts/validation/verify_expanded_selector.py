"""Independent split, least-squares, classifier stationarity and decision audit."""
from scripts.paths import PROJECT_ROOT, SCRIPTS, hash_matches
import argparse
from datetime import datetime
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax

from scripts.experiments.train_expanded_baselines import read, load, sha, signature, write_json
from scripts.validation.verify_expanded_baselines import audit_fit, near, transform

LENGTHS=(10,20,50)
THRESHOLDS=(0.,.0005,.001,.002,.005,.01,.02,None)
METHODS=('FixedContext10','FixedContext20','FixedContext50','GlobalBestOOF',
         'Classifier','ErrorAware','GatedError','Oracle')


def audit_selector(folder, raw, errors, batteries, classifier=False):
    raw=np.asarray(raw,dtype=float); m=load(folder/'parameters.npz')
    active=np.isfinite(raw).any(axis=0);mean=np.zeros(15);std=np.ones(15)
    mean[active]=np.nanmean(raw[:,active],axis=0)
    std[active]=np.nanstd(raw[:,active],axis=0);std=np.where(std>1e-6,std,1.)
    near(m['mean'],mean);near(m['std'],std)
    np.testing.assert_array_equal(m['active'],active)
    x=transform(raw,m); design=np.column_stack((x,np.ones(len(x))))
    penalty=np.diag(np.r_[np.full(15,np.sqrt(.1*len(x))),0.])
    target=np.vstack((errors,np.zeros((16,3))))
    w=np.linalg.lstsq(np.vstack((design,penalty)),target,rcond=None)[0]
    near(w[:-1],m['error_weight']);near(w[-1],m['error_bias'])
    batteries=np.array(batteries)
    means=np.mean([errors[batteries==c].mean(axis=0) for c in np.unique(batteries)],axis=0)
    assert int(m['default_index'])==int(np.argmin(means))
    if classifier:
        observed,targets=np.unique(np.argmin(errors,axis=1),return_inverse=True)
        np.testing.assert_array_equal(m['class_labels'],observed)
        k=len(observed); logits=x@m['class_weight']+m['class_bias']
        residual=softmax(logits,axis=1)-np.eye(k)[targets]
        grad=np.r_[(x.T@residual/len(x)+.1*m['class_weight']).ravel(),residual.mean(axis=0)]
        assert np.abs(grad).max()<5e-6, np.abs(grad).max()
    return m


def independent_gate(scores, default, threshold):
    selected=[]
    for score in scores:
        best=min(range(3),key=lambda j:score[j])
        selected.append(best if threshold is not None and score[default]-score[best]>threshold else default)
    return np.array(selected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'outputs/expanded_selector_v1')
    out=parser.parse_args().output.resolve();p=json.loads((out/'protocol.json').read_text())
    source=Path(p['source_baseline']);root=Path(p['source_data_root'])
    cells=p['development_batteries'];reserved=set(p['reserved_final_test_batteries'])
    assert len(cells)==8 and len(reserved)==4 and not set(cells)&reserved
    assert p['thresholds_fraction_soh']==list(THRESHOLDS) and p['methods']==list(METHODS)
    assert json.loads((source/'verification.json').read_text())['status']=='PASS'
    for filename,digest in p['inputs_sha256'].items():
        assert hash_matches(filename, digest),filename
        name=Path(filename).name
        assert name!='samples_final_test.csv'
        assert not any(name.startswith(c+'.') or name.startswith(c+'_') for c in reserved)
    for name,digest in p['code_sha256'].items():assert hash_matches(SCRIPTS/name, digest),name
    frozen=json.loads((out/'protocol_frozen.json').read_text())
    fitted=json.loads((out/'fitting_complete.json').read_text())
    run=json.loads((out/'run_summary.json').read_text())
    manifest=json.loads((out/'artifact_manifest.json').read_text())
    for d in (frozen,fitted,run,manifest):assert d['protocol_sha256']==signature(p)
    assert datetime.fromisoformat(frozen['frozen_utc'])<=datetime.fromisoformat(fitted['completed_utc'])<=datetime.fromisoformat(run['completed_utc'])
    for d in (fitted,manifest):
        for name,digest in d['files_sha256'].items():assert hash_matches(out/name, digest),name
    assert not run['final_test_evaluated'] and not run['final_test_payload_loaded']
    samples=read(root/'samples_development.csv');lookup={r['sample_id']:r for r in samples}
    f=load(source/'features.npz')
    feature={str(s):v for s,v in zip(f['sample_id'],f['context_by_length'])}
    assert len(lookup)==len(samples)==len(feature)==155
    assert {r['battery'] for r in samples}==set(cells)
    ids=lambda cs:[r['sample_id'] for r in samples if r['battery'] in cs]
    X=lambda ss,j:np.array([feature[s][j] for s in ss])
    Y=lambda ss:np.array([float(lookup[s]['soh']) for s in ss])
    teachers={};teacher_count=0
    for train_cells in combinations(cells,5):
        key='_'.join(train_cells);ss=ids(train_cells);teachers[key]=[]
        for j,L in enumerate(LENGTHS):
            folder=out/'teachers'/key/f'lb_{L}'
            meta=json.loads((folder/'fit.json').read_text())
            assert meta['train_batteries']==list(train_cells) and meta['train_sample_ids']==ss
            teachers[key].append(audit_fit(folder,X(ss,j),Y(ss)));teacher_count+=1
    labels_rows=read(out/'nested_oof_labels.csv')
    labels={(r['validation_battery'],r['calibration_battery'],r['sample_id']):r for r in labels_rows}
    assert len(labels)==len(labels_rows)==155*7*6
    cal_rows=read(out/'calibration_predictions.csv')
    cal={(r['validation_battery'],r['sample_id']):r for r in cal_rows}
    assert len(cal)==len(cal_rows)==155*7
    base_oof={(r['validation_battery'],r['sample_id'],int(r['lookback'])):float(r['soh_pred'])
              for r in read(source/'oof_predictions.csv')}
    base_labels={(r['validation_battery'],r['sample_id']):r for r in read(source/'oof_labels.csv')}
    base_choices={r['validation_battery']:int(r['selected_lookback']) for r in read(source/'global_choices.csv')}
    choices={r['validation_battery']:r for r in read(out/'choices.csv')}
    tuning={(r['validation_battery'],int(r['threshold_index'])):r for r in read(out/'threshold_validation.csv')}
    assert len(tuning)==64
    predictions=read(out/'predictions.csv')
    pred={(r['battery'],r['sample_id'],r['method']):r for r in predictions}
    assert len(pred)==len(predictions)==155*len(METHODS)
    decisions={r['sample_id']:r for r in read(out/'decisions.csv')}
    assert len(decisions)==155
    selector_count=0
    for outer in cells:
        fit_cells=[c for c in cells if c!=outer];cv={}
        for c in fit_cells:
            selector_cells=[d for d in fit_cells if d!=c];by_id={};refs={}
            for label_cell in selector_cells:
                train_cells=[d for d in selector_cells if d!=label_cell];key='_'.join(train_cells)
                assert not {outer,c,label_cell}&set(train_cells)
                refs[label_cell]=key;ss=ids([label_cell])
                values=np.column_stack([transform(X(ss,j),m)@m['weight']+m['bias'] for j,m in enumerate(teachers[key])])
                errors=np.abs(values-Y(ss)[:,None])
                for s,e in zip(ss,errors):
                    by_id[s]=e;r=labels[outer,c,s]
                    assert r['teacher_fit_key']==key and r['label_battery']==label_cell
                    near([float(r[f'error_lb{L}']) for L in LENGTHS],e)
            ss=ids(selector_cells);folder=out/'selectors'/f'validation_{outer}'/f'calibration_{c}'
            meta=json.loads((folder/'fit.json').read_text())
            assert meta['train_batteries']==selector_cells and meta['train_sample_ids']==ss
            assert meta['teacher_references']==refs
            m=audit_selector(folder,X(ss,0),np.array([by_id[s] for s in ss]),[lookup[s]['battery'] for s in ss])
            selector_count+=1;ss=ids([c]);default=int(m['default_index'])
            scores=transform(X(ss,0),m)@m['error_weight']+m['error_bias']
            errors=np.array([[abs(base_oof[outer,s,L]-float(lookup[s]['soh'])) for L in LENGTHS] for s in ss])
            cv[c]=(scores,errors,default)
            for s,score,e in zip(ss,scores,errors):
                r=cal[outer,s]
                assert r['calibration_battery']==c and int(r['default_lookback'])==LENGTHS[default]
                near(score,[float(r[f'estimated_error_lb{L}']) for L in LENGTHS])
                near(e,[float(r[f'actual_error_lb{L}']) for L in LENGTHS])
        losses=[];counts=[]
        for ti,threshold in enumerate(THRESHOLDS):
            maes=[];count=0
            for scores,errors,default in cv.values():
                chosen=independent_gate(scores,default,threshold)
                maes.append(errors[np.arange(len(chosen)),chosen].mean());count+=int((chosen!=default).sum())
            losses.append(float(np.mean(maes)));counts.append(count)
            r=tuning[outer,ti]
            near(float(r['calibration_macro_mae']),losses[-1]);assert int(r['calibration_switches'])==count
        eligible=[j for j,loss in enumerate(losses) if loss<=min(losses)+1e-12]
        ti=min(eligible,key=lambda j:(counts[j],-j))
        assert int(choices[outer]['threshold_index'])==ti
        ss=ids(fit_cells);errors=np.array([[float(base_labels[outer,s][f'error_lb{L}']) for L in LENGTHS] for s in ss])
        folder=out/'selectors'/f'validation_{outer}'/'final';meta=json.loads((folder/'fit.json').read_text())
        assert meta['train_batteries']==fit_cells and meta['train_sample_ids']==ss and meta['threshold_index']==ti
        assert meta['labels_validation_filter']==outer and meta['threshold_fraction']==THRESHOLDS[ti]
        m=audit_selector(folder,X(ss,0),errors,[lookup[s]['battery'] for s in ss],classifier=True);selector_count+=1
        default=int(m['default_index'])
        assert int(choices[outer]['default_lookback'])==LENGTHS[default]==base_choices[outer]
        ss=ids([outer]);x=transform(X(ss,0),m);scores=x@m['error_weight']+m['error_bias']
        chosen={'Classifier':m['class_labels'][np.argmax(x@m['class_weight']+m['class_bias'],axis=1)],
                'ErrorAware':np.argmin(scores,axis=1),'GatedError':independent_gate(scores,default,THRESHOLDS[ti])}
        bank=[load(source/'models'/f'validation_{outer}'/'candidates'/f'lb_{L}'/'parameters.npz') for L in LENGTHS]
        values=np.column_stack([transform(X(ss,j),b)@b['weight']+b['bias'] for j,b in enumerate(bank)])
        errors=np.abs(values-Y(ss)[:,None]);oracle=np.argmin(errors,axis=1)
        for i,s in enumerate(ss):
            d=decisions[s]
            near([float(d[f'estimated_error_lb{L}']) for L in LENGTHS],scores[i])
            near(float(d['predicted_gain']),scores[i,default]-min(scores[i]))
            assert int(d['oracle_lookback'])==LENGTHS[int(oracle[i])]
            for method in METHODS:
                j=LENGTHS.index(int(method.replace('FixedContext',''))) if method.startswith('FixedContext') else default if method=='GlobalBestOOF' else int(oracle[i]) if method=='Oracle' else int(chosen[method][i])
                r=pred[outer,s,method]
                assert int(r['selected_lookback'])==LENGTHS[j]
                if method in chosen:assert int(d[f'{method}_lookback'])==LENGTHS[j]
                near(float(r['soh_true']),float(lookup[s]['soh']));near(float(r['soh_pred']),values[i,j])
                near(float(r['absolute_error']),errors[i,j]);near(float(r['squared_error']),errors[i,j]**2)
                near(float(r['improvement_over_default']),errors[i,default]-errors[i,j])
                near(float(r['selection_regret']),errors[i,j]-errors[i,oracle[i]])
        print(outer,'nested roles / threshold / selectors / predictions PASS',flush=True)
    for r in read(source/'predictions.csv'):
        now=pred[r['battery'],r['sample_id'],r['method']]
        near(float(now['soh_pred']),float(r['soh_pred']))
    metrics=read(out/'metrics_by_battery.csv');summary=read(out/'summary.csv')
    for method in METHODS:
        maes=[];rmses=[];lengths=[];gains=[];counts=np.zeros(3,dtype=int)
        for cell in cells:
            rr=[pred[cell,s,method] for s in ids([cell])]
            e=np.array([float(r['absolute_error']) for r in rr])
            gain=np.array([float(r['improvement_over_default']) for r in rr])
            maes.append(e.mean());rmses.append(np.sqrt(np.square(e).mean()))
            lengths.append(np.mean([int(r['selected_lookback']) for r in rr]));gains.append(gain.mean())
            cc=[sum(r['selected_lookback']!=r['default_lookback'] for r in rr),int((gain>1e-12).sum()),int((gain < -1e-12).sum())]
            counts+=cc
            row=next(r for r in metrics if r['method']==method and r['battery']==cell)
            near(float(row['mae']),maes[-1]);near(float(row['rmse']),rmses[-1]);near(float(row['mean_lookback']),lengths[-1])
            near(float(row['mean_improvement_over_default']),gains[-1])
            near([int(row[n]) for n in ('switches','helpful_switches','harmful_switches')],cc)
        row=next(r for r in summary if r['method']==method)
        near(float(row['battery_macro_mae']),np.mean(maes));near(float(row['battery_macro_rmse']),np.mean(rmses))
        near(float(row['battery_macro_mean_lookback']),np.mean(lengths))
        near(float(row['battery_macro_improvement_over_default']),np.mean(gains))
        near([int(row[n]) for n in ('switches','helpful_switches','harmful_switches')],counts)
        assert int(row['cells_better_than_default'])==sum(g>1e-12 for g in gains)
        assert int(row['cells_worse_than_default'])==sum(g < -1e-12 for g in gains)
    for row in read(out/'metrics_by_condition.csv'):
        mm=[float(r['mae']) for r in metrics if r['method']==row['method'] and r['condition']==row['condition']]
        near(float(row['battery_macro_mae']),np.mean(mm));assert int(row['cells'])==len(mm)
    assert teacher_count==168 and selector_count==64
    result=dict(status='PASS',protocol_sha256=signature(p),new_teacher_models_verified=teacher_count,
        error_selectors_verified=selector_count,classifiers_verified=8,development_samples=155,
        prediction_rows=len(predictions),final_test_evaluated=False,
        checks=['Unchanged baseline predictions, source files and frozen code hashes',
                'All five-cell teachers reconstructed by independent augmented least squares',
                'Outer validation and inner calibration cells absent from every label teacher',
                'All 64 error Selector fits reconstructed with train-only scaling',
                'Eight convex classifier solutions pass independent stationarity check',
                'All thresholds independently selected using inner cell-macro SOH errors only',
                'Target-free choices and all prediction rows independently recomputed',
                'Equal-cell aggregate metrics and helpful/harmful switches',
                'Final-test payloads absent from inputs; no final-test evaluation'])
    write_json(out/'verification.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
