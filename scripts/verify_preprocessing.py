"""Independent invariants for the canonical data, measured targets, folds and loader."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from battery_data import BatteryDataset, DEFAULT_ROOT


def main():
    root=DEFAULT_ROOT
    samples=list(csv.DictReader((root/'samples.csv').open()))
    by_id={r['sample_id']:r for r in samples}
    assert len(samples)==len(by_id)>0
    arrays={};raw={};audit=json.loads((root/'audit.json').read_text())
    for b in audit:
        with np.load(root/f'{b}.npz',allow_pickle=False) as f:arrays[b]={k:f[k] for k in f.files}
        raw_path=root.parents[1]/'raw'/f'{b}.mat'
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest()==audit[b]['raw_sha256']
        raw[b]=loadmat(raw_path,simplify_cells=True)['data']['step']
        d=arrays[b];assert d['x'].shape[1:]==(128,4)
        assert np.array_equal(d['observed_mask'],np.isfinite(d['x']))
        assert np.array_equal(d['context_observed_mask'],np.isfinite(d['context']))
        assert np.all(np.diff(d['cycle_end_time_s'])>0)
        assert np.isnan(d['x'][:,:,3][~d['observed_mask'][:,:,3]]).all()
        assert np.all(d['x'][:,:,3][d['observed_mask'][:,:,3]]>=-40)
        assert np.all(d['x'][:,:,3][d['observed_mask'][:,:,3]]<=100)
        np.testing.assert_allclose(d['x'][:,1:,2],np.diff(d['x'][:,:,0],axis=1),atol=5e-7)
        assert np.all(d['x'][:,0,2]==0)
        # All raw RW discharge steps must belong to exactly one recorded cycle.
        cycles=list(csv.DictReader((root/f'{b}_cycles.csv').open()))
        covered=[]
        for c in cycles:
            first,last=int(c['first_raw_step']),int(c['last_raw_step'])
            covered.extend(i for i in range(first,last+1) if raw[b][i]['comment']=='discharge (random walk)')
        expected=[i for i,r in enumerate(raw[b]) if r['comment']=='discharge (random walk)']
        assert covered==expected
    for r in samples:
        b=r['battery'];d=arrays[b];n=int(r['history_end_exclusive'])
        assert n>=50 and np.all(d['cycle_valid'][n-50:n])
        assert d['cycle_end_time_s'][n-1]<float(r['target_start_time_s'])
        assert n==len(d['x']) or d['cycle_end_time_s'][n]>=float(r['target_start_time_s'])
        refs=list(csv.DictReader((root/f'{b}_references.csv').open()))
        ref=refs[int(r['reference_index'])];rr=raw[b][int(ref['raw_step'])]
        q=np.trapezoid(np.atleast_1d(rr['current']),np.atleast_1d(rr['relativeTime']))/3600
        np.testing.assert_allclose(float(r['soh']),q/audit[b]['initial_reference_capacity_Ah'],rtol=1e-12)
        assert np.isclose(d['context'][n-1,0],n)
        assert d['context'].shape[1]==15
    folds=list((root/'folds').iterdir());assert len(folds)==16
    oof={b:[] for b in audit}
    for folder in folds:
        split=json.loads((folder/'split.json').read_text());scale=json.loads((folder/'scalers.json').read_text())
        tr,va,te=[set(split[k]) for k in ['train','validation','test']]
        assert not(tr&va or tr&te or va&te)
        for role in ['train','validation','test']:
            assert {by_id[s]['battery'] for s in split[role]}==set(split[role+'_batteries'])
        train_values=[];train_context=[]
        for b in split['train_batteries']:
            used=set()
            for sid in split['train']:
                r=by_id[sid]
                if r['battery']!=b:continue
                n=int(r['history_end_exclusive']);used.update(range(n-50,n));train_context.append(arrays[b]['context'][n-1])
            assert sorted(used)==scale['fit_cycle_indices'][b]
            train_values.append(arrays[b]['x'][sorted(used)].reshape(-1,4))
        np.testing.assert_allclose(np.nanmean(np.concatenate(train_values),axis=0),scale['channel_mean'],rtol=1e-6)
        np.testing.assert_allclose(np.nanmean(train_context,axis=0),scale['context_mean'],rtol=1e-6)
        for b in audit:
            with np.load(folder/f'{b}.npz',allow_pickle=False) as f:
                assert np.isfinite(f['x']).all() and np.isfinite(f['context']).all()
                # Missing values become exactly zero in normalized coordinates.
                assert np.all(f['x'][~arrays[b]['observed_mask']]==0)
        for role in ['train','validation','test']:
            datasets=[BatteryDataset(folder.name,role,lb) for lb in [10,20,50]]
            assert all(len(ds)==len(split[role]) for ds in datasets)
            for j in range(len(datasets[0])):
                items=[ds[j] for ds in datasets]
                assert len({item['sample_id'] for item in items})==1
                assert len({float(item['soh']) for item in items})==1
                for lb,item in zip([10,20,50],items):
                    assert item['x'].shape==(lb,128,4)
                    np.testing.assert_array_equal(item['x'],items[-1]['x'][-lb:])
        if split['validation']:
            oof[split['test_batteries'][0]].extend(split['validation'])
    for test,ids in oof.items():
        expected={r['sample_id'] for r in samples if r['battery']!=test}
        assert len(ids)==len(set(ids)) and set(ids)==expected
    result=dict(status='PASS',primary_samples=len(samples),folds=len(folds),
        checks=['raw SHA256','all RW steps assigned exactly once','cycle and timestamp validity',
                'independent reference capacity integration','no future cycle input',
                'same targets and suffix windows across lookbacks','train-only scaler fitting',
                'battery-disjoint splits','finite normalized tensors','missing masks',
                'out-of-fold validation covers each outer-training sample once','dataset loader'])
    (root/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
