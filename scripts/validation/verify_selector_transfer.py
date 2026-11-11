"""Independently reproduce selector fits and paired bank-transfer diagnostics."""
from scripts.paths import hash_matches
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from scripts.experiments.train_lookback_oof import read_csv,sha256,write_json
from scripts.experiments.train_selector_transfer import DEFAULT_OUTPUT,BANKS,METHODS,check_inputs
from scripts.experiments.train_temporal_context import load_npz,signature
from scripts.validation.verify_temporal_context import audit_ridge,near,scaled,scale_fit

LENGTHS=(10,20,50)


def verify(output):
    p=json.loads((output/'protocol.json').read_text());check_inputs(p);sig=signature(p)
    run=json.loads((output/'run_summary.json').read_text())
    manifest=json.loads((output/'artifact_manifest.json').read_text())
    frozen=json.loads((output/'protocol_frozen.json').read_text())
    fitted=json.loads((output/'fitting_complete.json').read_text())
    assert run['status']=='EVALUATION_COMPLETE' and not run['current_run_test_tuning']
    assert all(x['protocol_sha256']==sig for x in (run,manifest,frozen,fitted))
    assert datetime.fromisoformat(frozen['frozen_utc'])<=datetime.fromisoformat(fitted['completed_utc'])<=datetime.fromisoformat(run['completed_utc'])
    for mapping in (manifest['files_sha256'],fitted['files_sha256']):
        for name,digest in mapping.items():assert hash_matches(output/name, digest),name
    source,root=Path(p['source_root']),Path(p['source_data_root'])
    samples={r['sample_id']:r for r in read_csv(root/'samples.csv')}
    feature_file=load_npz(source/'features.npz')
    features={str(sid):x for sid,x in zip(feature_file['sample_id'],feature_file['context_by_length'])}
    canonical={b:load_npz(root/f'{b}.npz')['context'] for b in p['batteries']}
    for sid,row in samples.items():
        np.testing.assert_array_equal(features[sid][0],canonical[row['battery']][int(row['history_end_exclusive'])-1])
    splits=json.loads((output/'splits.json').read_text());assert len(splits)==12
    splitkeys={(s['outer_test_battery'],s['calibration_battery']) for s in splits}
    assert splitkeys=={(a,b) for a in p['batteries'] for b in p['batteries'] if a!=b}
    pred_rows=read_csv(output/'predictions.csv')
    key=lambda r:(r['outer_test_battery'],r['calibration_battery'],r['sample_id'],r['bank'],r['method'])
    pred={key(r):r for r in pred_rows}
    expected={(s['outer_test_battery'],s['calibration_battery'],sid,bank,m)
              for s in splits for sid in s['test_sample_ids'] for bank in BANKS for m in METHODS}
    assert set(pred)==expected and len(pred)==len(pred_rows)==run['prediction_rows']
    selection_rows=read_csv(output/'selections.csv')
    skey=lambda r:(r['outer_test_battery'],r['calibration_battery'],r['sample_id'],r['method'])
    selection={skey(r):r for r in selection_rows}
    assert len(selection)==len(selection_rows)==len(samples)*3*2
    shift_rows=read_csv(output/'label_shift.csv')
    shift={(r['outer_test_battery'],r['calibration_battery'],r['sample_id']):r for r in shift_rows}
    assert len(shift)==len(shift_rows)==run['test_sample_rotations']==len(samples)*3
    diag_rows=read_csv(output/'selector_diagnostics.csv');diag={skey(r):r for r in diag_rows}
    assert set(diag)==set(selection) and len(diag_rows)==len(selection)
    previous={(r['outer_test_battery'],r['sample_id'],r['method']):r for r in read_csv(source/'predictions.csv')}
    raw=lambda ids,j:np.array([features[sid][j] for sid in ids])
    targets=lambda ids:np.array([float(samples[sid]['soh']) for sid in ids])
    verified_models={};all_stats={}
    for s in splits:
        outer,cal=s['outer_test_battery'],s['calibration_battery']
        split=json.loads((root/'folds'/f'test_{outer}__val_{cal}'/'split.json').read_text())
        final=json.loads((root/'folds'/f'test_{outer}__final'/'split.json').read_text())
        assert s['predictor_fit_sample_ids']==split['train'] and s['selector_fit_sample_ids']==split['validation']
        assert s['test_sample_ids']==split['test'] and s['predictor_fit_batteries']==split['train_batteries']
        roles=[set(split[k]) for k in ('train','validation','test')]
        assert not roles[0]&roles[1] and not roles[0]&roles[2] and not roles[1]&roles[2]
        assert set(split['train_batteries'])==set(p['batteries'])-{outer,cal}
        assert all(samples[sid]['battery']==cal for sid in split['validation'])
        assert all(samples[sid]['battery']==outer for sid in split['test'])
        assert set(final['train_batteries'])==set(p['batteries'])-{outer}
        assert set(final['train'])==set(split['train'])|set(split['validation'])
        banks={}
        for bank in BANKS:
            fit_ids=split['train'] if bank=='FrozenPair' else final['train']
            sub=f'oof_{cal}' if bank=='FrozenPair' else 'final';banks[bank]={}
            for j,lb in enumerate(LENGTHS):
                path=source/'models'/f'test_{outer}'/sub/f'lb_{lb}'
                fit=json.loads((path/'fit.json').read_text());assert fit['train_sample_ids']==fit_ids
                if path not in verified_models:
                    verified_models[path]=audit_ridge(path,raw(fit_ids,j),targets(fit_ids),p['selectors']['l2'])
                banks[bank][lb]=verified_models[path]
        calibration_preds=np.column_stack([scaled(raw(split['validation'],j),banks['FrozenPair'][lb])@
            banks['FrozenPair'][lb]['weight']+banks['FrozenPair'][lb]['bias'] for j,lb in enumerate(LENGTHS)])
        error=np.abs(calibration_preds-targets(split['validation'])[:,None])
        folder=output/'models'/f'test_{outer}'/f'calibration_{cal}'
        selector=load_npz(folder/'parameters.npz');training=load_npz(folder/'training_data.npz')
        meta=json.loads((folder/'fit.json').read_text());assert meta==s
        assert list(training['sample_id'])==split['validation']
        np.testing.assert_array_equal(training['context_raw'],raw(split['validation'],0));near(training['errors'],error)
        labels=np.argmin(error,axis=1);np.testing.assert_array_equal(training['class_label'],labels)
        mu,sd,active=scale_fit(training['context_raw'])
        near(selector['mean'],mu);near(selector['std'],sd);np.testing.assert_array_equal(selector['active'],active)
        x=scaled(training['context_raw'],selector);classes,yi=np.unique(labels,return_inverse=True)
        np.testing.assert_array_equal(classes,selector['class_labels'])
        if len(classes)>1:
            logits=x@selector['class_weight']+selector['class_bias']
            prob=np.exp(logits-logits.max(axis=1,keepdims=True));prob/=prob.sum(axis=1,keepdims=True)
            residual=prob-np.eye(len(classes))[yi]
            assert np.abs(x.T@residual/len(x)+p['selectors']['l2']*selector['class_weight']).max()<1e-5
            assert np.abs(residual.mean(axis=0)).max()<1e-5
        else:near(selector['class_weight'],0.);near(selector['class_bias'],0.)
        design=np.column_stack((x,np.ones(len(x))))
        penalty=np.diag(np.r_[np.full(15,np.sqrt(len(x)*p['selectors']['l2'])),0.])
        solved=np.linalg.lstsq(np.vstack((design,penalty)),np.vstack((error,np.zeros((16,3)))),rcond=None)[0]
        near(solved[:-1],selector['error_weight']);near(solved[-1],selector['error_bias'])
        assert int(selector['global_best_lb'])==LENGTHS[np.argmin(error.mean(axis=0))]
        assert int(selector['majority_lb'])==LENGTHS[np.argmax(np.bincount(labels,minlength=3))]
        for sid in split['test']:
            z=scaled(features[sid][0],selector)
            logits=z@selector['class_weight']+selector['class_bias']
            prob=np.exp(logits-logits.max());prob/=prob.sum();full=np.zeros(3);full[classes]=prob
            scores=z@selector['error_weight']+selector['error_bias']
            decisions={'Adaptive':LENGTHS[np.argmax(full)],'ErrorAware':LENGTHS[np.argmin(scores)]}
            for method,score in (('Adaptive',full),('ErrorAware',scores)):
                row=selection[outer,cal,sid,method]
                near(score,[float(row[f'score_lb{lb}']) for lb in LENGTHS])
                assert int(row['selected_lookback'])==decisions[method]
                assert int(row['negative_score_count'])==int((score<0).sum())
            err={};best={}
            truth=float(samples[sid]['soh'])
            for bank in BANKS:
                values=np.array([float(scaled(features[sid][j],banks[bank][lb])@banks[bank][lb]['weight']+banks[bank][lb]['bias']) for j,lb in enumerate(LENGTHS)])
                err[bank]=np.abs(values-truth);best[bank]=LENGTHS[np.argmin(err[bank])]
                lengths={**{f'FixedContext{lb}':lb for lb in LENGTHS},**decisions,
                    'CalibrationBest':int(selector['global_best_lb']),'LabelMajority':int(selector['majority_lb']),
                    'Oracle':best[bank]}
                for method,lb in lengths.items():
                    row=pred[outer,cal,sid,bank,method]
                    assert int(row['lookback'])==lb;near(float(row['soh_pred']),values[LENGTHS.index(lb)])
                    near(float(row['soh_true']),truth);near(float(row['absolute_error']),abs(float(row['soh_pred'])-truth))
                    if bank=='RefitThree' and method.startswith('FixedContext'):
                        near(float(row['soh_pred']),float(previous[outer,sid,method]['soh_pred']))
            row=shift[outer,cal,sid];pair,refit=err['FrozenPair'],err['RefitThree']
            assert int(row['frozen_best_lookback'])==best['FrozenPair'] and int(row['refit_best_lookback'])==best['RefitThree']
            assert int(row['best_length_changed'])==int(best['FrozenPair']!=best['RefitThree'])
            near([float(row[f'frozen_error_lb{lb}']) for lb in LENGTHS],pair)
            near([float(row[f'refit_error_lb{lb}']) for lb in LENGTHS],refit)
            near(float(row['frozen_gap_to_second']),np.sort(pair)[1]-min(pair))
            near(float(row['refit_gap_to_second']),np.sort(refit)[1]-min(refit))
            near(float(row['frozen_oracle_mae']),min(pair));near(float(row['refit_oracle_mae']),min(refit))
            transferred=refit[LENGTHS.index(best['FrozenPair'])]
            near(float(row['frozen_oracle_choice_on_refit_error']),transferred)
            near(float(row['frozen_oracle_choice_on_refit_regret']),transferred-min(refit))
            for method,lb in decisions.items():
                d=diag[outer,cal,sid,method];j=LENGTHS.index(lb)
                assert int(d['selected_lookback'])==lb
                assert int(d['matches_frozen_best'])==int(lb==best['FrozenPair'])
                assert int(d['matches_refit_best'])==int(lb==best['RefitThree'])
                r2,r3=pair[j]-min(pair),refit[j]-min(refit)
                near(float(d['frozen_selection_regret']),r2);near(float(d['refit_selection_regret']),r3)
                near(float(d['regret_change_after_refit']),r3-r2)
        for bank in BANKS:
            oracle=np.mean([float(pred[outer,cal,sid,bank,'Oracle']['absolute_error']) for sid in split['test']])
            for method in METHODS:
                rows=[pred[outer,cal,sid,bank,method] for sid in split['test']]
                errors=np.array([float(r['absolute_error']) for r in rows])
                all_stats[outer,cal,bank,method]=dict(mae=float(errors.mean()),rmse=float(np.sqrt(np.mean(errors**2))),
                    regret=float(errors.mean()-oracle),mean_lookback=float(np.mean([int(r['lookback']) for r in rows])))
    assert len(verified_models)==48
    split_rows=read_csv(output/'metrics_by_split.csv');assert len(split_rows)==12*2*len(METHODS)
    for row in split_rows:
        k=(row['outer_test_battery'],row['calibration_battery'],row['bank'],row['method'])
        for metric,value in all_stats[k].items():near(float(row[metric]),value)
    cell_stats={}
    for b in p['batteries']:
        for bank in BANKS:
            for method in METHODS:
                stats=[all_stats[b,cal,bank,method] for cal in p['batteries'] if cal!=b]
                cell_stats[b,bank,method]={metric:float(np.mean([r[metric] for r in stats])) for metric in ('mae','rmse','regret','mean_lookback')}
    for row in read_csv(output/'metrics_by_battery.csv'):
        for metric,value in cell_stats[row['outer_test_battery'],row['bank'],row['method']].items():near(float(row[metric]),value)
    for row in read_csv(output/'summary.csv'):
        for metric in ('mae','rmse','regret','mean_lookback'):
            near(float(row[f'battery_macro_{metric}']),np.mean([cell_stats[b,row['bank'],row['method']][metric] for b in p['batteries']]))
    for row in read_csv(output/'paired_comparisons.csv'):
        bs=p['batteries'] if row['scope']=='MACRO' else [row['scope']]
        near(float(row['mae_difference']),np.mean([cell_stats[b,row['bank'],row['method']]['mae']-cell_stats[b,row['bank'],row['baseline']]['mae'] for b in bs]))
    for row in read_csv(output/'bank_effects.csv'):
        bs=p['batteries'] if row['scope']=='MACRO' else [row['scope']]
        for metric in ('mae','regret'):
            near(float(row[f'refit_minus_frozen_{metric}']),np.mean([cell_stats[b,'RefitThree',row['method']][metric]-cell_stats[b,'FrozenPair',row['method']][metric] for b in bs]))
    for row in read_csv(output/'label_shift_by_battery.csv'):
        rows=[r for r in shift_rows if r['outer_test_battery']==row['outer_test_battery']]
        assert len(rows)==int(row['target_rotations'])==3*int(row['unique_targets'])
        near(float(row['label_flip_rate']),np.mean([int(r['best_length_changed']) for r in rows]))
        near(float(row['transferred_oracle_regret']),np.mean([float(r['frozen_oracle_choice_on_refit_regret']) for r in rows]))
        near(float(row['frozen_gap_median']),np.median([float(r['frozen_gap_to_second']) for r in rows]))
        near(float(row['refit_gap_median']),np.median([float(r['refit_gap_to_second']) for r in rows]))
    for row in read_csv(output/'selector_diagnostics_by_battery.csv'):
        rows=[r for r in diag_rows if (r['outer_test_battery'],r['method'])==(row['outer_test_battery'],row['method'])]
        for metric in ('matches_frozen_best','matches_refit_best','frozen_selection_regret','refit_selection_regret','regret_change_after_refit'):
            near(float(row[metric]),np.mean([float(r[metric]) for r in rows]))
    result=dict(status='PASS',protocol_sha256=sig,checked_predictor_fits=len(verified_models),
        checked_selector_pairs=12,unique_test_samples=len(samples),test_sample_rotations=len(shift_rows),
        prediction_rows=len(pred_rows),checks=['unchanged verified parent and data hashes',
        '2 predictor cells / 1 selector cell / 1 test cell; no test fit',
        '48 reused predictor fits by independent least-squares reconstruction',
        'calibration errors regenerated from their exact frozen bank',
        'calibration-only scalers and 24 independently checked selectors',
        'identical selector decisions across both bank conditions',
        'independent target-free routing and posthoc label-change diagnostics',
        'original three-cell fixed predictions reproduced',
        'all rotations/cells included with declared macro aggregation'],
        limitations='Posthoc diagnostic plus one-cell-calibrated controls; cannot isolate all causes or compare directly to previous three-cell-selector training.')
    write_json(output/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    verify(parser.parse_args().output.resolve())
