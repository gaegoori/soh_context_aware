"""Verify the QC version from raw temperature traces and prior measured targets."""
from scripts.paths import PROJECT_ROOT, resolve_file, hash_matches
import argparse
import csv
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
from scipy.io import loadmat


def read(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def sha(path):return hashlib.sha256(resolve_file(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=PROJECT_ROOT)
    args=parser.parse_args();project=args.project_root.resolve()
    source=project/'data/processed/expanded_v1';root=project/'data/processed/expanded_qc_v1'
    policy=json.loads((root/'quality_policy.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    audit=json.loads((root/'audit.json').read_text())
    assert policy['temperature_valid_range_C']==[0.,100.] and policy['maximum_reference_delay_hours']==120.
    assert manifest['quality_policy_sha256']==sha(root/'quality_policy.json')
    for path,digest in policy['source_input_sha256'].items():assert hash_matches(path, digest)
    for name,digest in policy['code_sha256'].items():assert hash_matches(project/'scripts'/name, digest)
    assert hash_matches(source/'split_plan.json', sha(root/'split_plan.json'))
    split=json.loads((root/'split_plan.json').read_text())
    before=read(source/'samples.csv');after=read(root/'samples.csv')
    assert after==[r for r in before if float(r['reference_delay_s'])<=120*3600]
    excluded=read(root/'excluded_samples.csv')
    assert {r['sample_id'] for r in excluded}=={r['sample_id'] for r in before if float(r['reference_delay_s'])>120*3600}
    for role,key in [('development','development_batteries'),('final_test','final_test_batteries')]:
        assert read(root/f'samples_{role}.csv')==[r for r in after if r['battery'] in split[key]]
    for cell in manifest['batteries']:
        with np.load(root/f'{cell}.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
        with np.load(source/f'{cell}.npz',allow_pickle=False) as f:old={k:f[k] for k in f.files}
        for key in ('cycle_valid','cycle_end_time_s','secondary_interpolated_soh'):np.testing.assert_array_equal(d[key],old[key])
        np.testing.assert_array_equal(d['x'][:,:,:3],old['x'][:,:,:3])
        unchanged=[i for i in range(15) if i not in (5,13,14)]
        np.testing.assert_array_equal(d['context'][:,unchanged],old['context'][:,unchanged])
        raw_path=project/f'data/raw/{cell}.mat'
        assert hash_matches(raw_path, audit[cell]['raw_sha256'])
        steps=loadmat(raw_path,simplify_cells=True)['data']['step']
        meta=read(root/f'{cell}_cycles.csv')
        for i,c in enumerate(meta):
            records=steps[int(c['first_raw_step']):int(c['last_raw_step'])+1]
            t=np.concatenate([np.atleast_1d(r['time']) for r in records]).astype(float)
            temp=np.concatenate([np.atleast_1d(r['temperature']) for r in records]).astype(float)
            t,index=np.unique(t,return_index=True);temp=temp[index]
            temp[(temp<0)|(temp>100)|(~np.isfinite(temp))]=np.nan
            expected=np.interp(np.linspace(t[0],t[-1],128),t,temp).astype(np.float32)
            np.testing.assert_array_equal(d['x'][i,:,3],expected)
            missing=float(np.mean(~np.isfinite(expected)))
            np.testing.assert_allclose(missing,float(c['temperature_missing_fraction']),rtol=1e-7)
            np.testing.assert_allclose(d['context'][i,5],missing,rtol=1e-6,atol=1e-8)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                recent=d['x'][max(0,i-9):i+1,:,3]
                np.testing.assert_allclose(d['context'][i,13:],[np.nanmean(recent),np.nanstd(recent)],rtol=1e-6,atol=1e-6,equal_nan=True)
        np.testing.assert_array_equal(d['observed_mask'],np.isfinite(d['x']))
        np.testing.assert_array_equal(d['context_observed_mask'],np.isfinite(d['context']))
        cell_rows=[r for r in after if r['battery']==cell]
        assert audit[cell]['primary_samples']==len(cell_rows)
        for r in cell_rows:
            n=int(r['history_end_exclusive'])
            assert n>=50 and d['cycle_valid'][n-50:n].all()
            assert d['cycle_end_time_s'][n-1]<float(r['target_start_time_s'])
            assert float(r['reference_delay_s'])<=120*3600
        print(cell,'quality PASS',len(cell_rows),flush=True)
    result=dict(status='PASS',primary_samples=len(after),excluded_for_horizon=len(excluded),
        development_samples=len(read(root/'samples_development.csv')),
        reserved_final_test_samples=len(read(root/'samples_final_test.csv')),
        checks=['Raw temperature mask and interpolation independently reconstructed',
            'Voltage, current, dV and cycle structure unchanged','Context changed only in temperature fields',
            'Measured targets exactly equal prior targets within predeclared 120h horizon',
            'Cell reservation unchanged','Source files unchanged','No model performance used for QC'])
    (root/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
