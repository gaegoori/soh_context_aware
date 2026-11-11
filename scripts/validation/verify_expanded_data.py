"""Independently check sources, measured targets, causal histories and cell roles."""
from scripts.paths import PROJECT_ROOT, resolve_file, hash_matches
import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

EXPECTED_MD5 = {'high':'9a0fabc687ae3a8d990c9d6f2f0412d4',
                'low':'14130c74dd2caa3c6fbda83df26eb55d'}


def rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def digest(path, algorithm='sha256'):
    h=hashlib.new(algorithm)
    with resolve_file(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def vector(record, key):
    return np.atleast_1d(record[key]).astype(np.float64)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=PROJECT_ROOT)
    parser.add_argument('--original-root',type=Path)
    parser.add_argument('--version',default='expanded_v1')
    args=parser.parse_args()
    project=args.project_root.resolve()
    root=project/'data/processed'/args.version
    original=args.original_root or project/'data/processed/main_v1'
    manifest=json.loads((root/'manifest.json').read_text())
    split=json.loads((root/'split_plan.json').read_text())
    audit=json.loads((root/'audit.json').read_text())
    sources=json.loads((root/'sources.json').read_text())
    samples=rows(root/'samples.csv')
    by_id={r['sample_id']:r for r in samples}
    assert len(samples)==len(by_id)>0
    dev=set(split['development_batteries']); test=set(split['final_test_batteries'])
    assert len(dev)==8 and len(test)==4 and not dev&test
    assert dev|test==set(manifest['batteries'])==set(audit)
    assert {'RW3','RW4','RW5','RW6'}<=dev
    assert manifest['quality_current_range_A']==[-0.2,5.25]
    for group, source in sources.items():
        archive=project/'data/raw/source_archives'/source['title']
        assert digest(archive,'md5')==EXPECTED_MD5[group]==source['md5']
        assert hash_matches(archive, source['archive_sha256'])
        assert len(dev&set(source['cells']))==len(test&set(source['cells']))==2
        with zipfile.ZipFile(archive) as z:
            for cell in source['cells']:
                names=[n for n in z.namelist() if '__MACOSX' not in n and Path(n).name==cell+'.mat']
                assert len(names)==1
                assert hashlib.sha256(z.read(names[0])).hexdigest()==digest(project/f'data/raw/{cell}.mat')
    for role,cells in (('development',dev),('final_test',test)):
        assert rows(root/f'samples_{role}.csv')==[r for r in samples if r['battery'] in cells]
    totals={'cycles':0,'valid_cycles':0,'references':0}
    original_rows=rows(original/'samples.csv')
    for cell in manifest['batteries']:
        raw_path=project/f'data/raw/{cell}.mat'
        assert hash_matches(raw_path, audit[cell]['raw_sha256'])
        steps=loadmat(raw_path,simplify_cells=True)['data']['step']
        with np.load(root/f'{cell}.npz',allow_pickle=False) as f:
            d={k:f[k] for k in f.files}
        cycles=rows(root/f'{cell}_cycles.csv'); refs=rows(root/f'{cell}_references.csv')
        assert d['x'].shape==(len(cycles),128,4)
        assert d['context'].shape==(len(cycles),15)
        np.testing.assert_array_equal(d['observed_mask'],np.isfinite(d['x']))
        np.testing.assert_array_equal(d['context_observed_mask'],np.isfinite(d['context']))
        np.testing.assert_array_equal(d['cycle_valid'],[r['valid']=='True' for r in cycles])
        assert np.all(np.diff(d['cycle_end_time_s'])>0)
        np.testing.assert_allclose(d['x'][:,1:,2],np.diff(d['x'][:,:,0],axis=1),atol=5e-7)
        assert np.all(d['x'][:,0,2]==0)
        finite_temp=d['x'][:,:,3][d['observed_mask'][:,:,3]]
        assert np.all((finite_temp>=-40)&(finite_temp<=100))
        covered=[]
        for c in cycles:
            first,last=int(c['first_raw_step']),int(c['last_raw_step'])
            ids=[i for i in range(first,last+1) if steps[i]['comment']=='discharge (random walk)']
            covered.extend(ids)
            q=sum(np.trapezoid(vector(steps[i],'current'),vector(steps[i],'relativeTime')) for i in ids)/3600
            np.testing.assert_allclose(q,float(c['discharge_Ah']),rtol=1e-12,atol=1e-12)
            if c['valid']=='True':
                current=np.concatenate([vector(steps[i],'current') for i in range(first,last+1)])
                voltage=np.concatenate([vector(steps[i],'voltage') for i in range(first,last+1)])
                assert np.all(np.isfinite(current)) and np.all(np.isfinite(voltage))
                assert np.all((current>=-0.2)&(current<=5.25))
                assert np.all((voltage>=2.5)&(voltage<=4.5))
                assert float(c['preceding_charge_end_V'])>=4.15 and float(c['end_voltage_V'])<=3.25
        assert covered==[i for i,r in enumerate(steps) if r['comment']=='discharge (random walk)']
        expected=[]; seen=set(); q0=None
        for reference in refs:
            rr=steps[int(reference['raw_step'])]
            assert rr['comment']=='reference discharge'
            t,current,v=vector(rr,'relativeTime'),vector(rr,'current'),vector(rr,'voltage')
            good=len(t)>1 and np.isfinite(t).all() and np.isfinite(current).all() and np.all(np.diff(t)>=0)
            q=float(np.trapezoid(current,t)/3600) if good else 0.
            valid=bool(good and q>0 and np.isfinite(v).all() and v[-1]<=3.25)
            assert valid==(reference['valid']=='True')
            np.testing.assert_allclose(q,float(reference['capacity_Ah']),rtol=1e-12,atol=1e-12)
            if valid and q0 is None:q0=q
            start=vector(rr,'time')[0]
            n=int(np.count_nonzero(d['cycle_end_time_s']<start))
            assert n==int(reference['history_end_exclusive'])
            if not valid or n<50 or not d['cycle_valid'][n-50:n].all() or n in seen:continue
            seen.add(n)
            sid=f"{cell}_ref{int(reference['reference_index']):03d}"
            expected.append(sid)
            r=by_id[sid]
            assert n==int(r['history_end_exclusive'])
            assert float(r['input_end_time_s'])==d['cycle_end_time_s'][n-1]<float(r['target_start_time_s'])
            np.testing.assert_allclose(float(r['soh']),q/q0,rtol=1e-12)
            np.testing.assert_allclose(float(r['reference_delay_s']),start-d['cycle_end_time_s'][n-1],atol=1e-8)
            assert d['context'][n-1,0]==n
            assert np.isclose(d['context'][n-1,2],sum(float(c['discharge_Ah']) for c in cycles[:n]),rtol=1e-6)
            for length in (10,20,50):
                assert d['x'][n-length:n].shape==(length,128,4)
                np.testing.assert_array_equal(d['x'][n-length:n],d['x'][n-50:n][-length:])
        assert expected==[r['sample_id'] for r in samples if r['battery']==cell]
        assert len(expected)==audit[cell]['primary_samples']
        np.testing.assert_allclose(q0,audit[cell]['initial_reference_capacity_Ah'],rtol=1e-12)
        if cell in ('RW3','RW4','RW5','RW6'):
            with np.load(original/f'{cell}.npz',allow_pickle=False) as old:
                assert set(d)==set(old.files)
                for key in d:np.testing.assert_array_equal(d[key],old[key])
            assert [r for r in samples if r['battery']==cell]==[r for r in original_rows if r['battery']==cell]
        totals['cycles']+=len(cycles)
        totals['valid_cycles']+=int(d['cycle_valid'].sum())
        totals['references']+=len(refs)
        print(cell,'PASS',len(expected),'measured targets',flush=True)
    assert not (root/'models').exists() and not (root/'folds').exists()
    result=dict(status='PASS',batteries=len(manifest['batteries']),primary_samples=len(samples),
        development_samples=sum(r['battery'] in dev for r in samples),
        reserved_final_test_samples=sum(r['battery'] in test for r in samples),**totals,
        checks=['Published archive MD5 and MAT provenance','Raw discharge step coverage',
            'Cycle discharge amount and reference capacity independently reintegrated',
            'Measured eligible targets complete, unique and common to all three lookbacks',
            'Future cycles and reference discharge excluded from input',
            'Temperature masks, dV and tensor shapes','Original four cells exactly reproduce main_v1',
            'Disjoint 8-cell development / 4-cell final-test reservation',
            'No fitted models, scalers or final-test performance evaluation'],
        scope='Data/protocol verification only. Not a model performance or sensor calibration claim.')
    (root/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
