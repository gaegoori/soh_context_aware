"""Build audited cycle-level, measured-target NASA RW data and nested LOBO folds.

Run: .venv-repro/bin/python -m scripts.preprocessing.preprocess_main
No model training and no use of the reproduction caches.
"""
from scripts.paths import PROJECT_ROOT
import csv
import hashlib
import json
import platform
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from scipy.io import loadmat

ROOT = PROJECT_ROOT
OUT = ROOT / 'data/processed/main_v1'
CELLS = ['RW3', 'RW4', 'RW5', 'RW6']
LBS = [10, 20, 50]
RW = {'discharge (random walk)', 'rest (random walk)', 'rest post random walk discharge'}
CHANNELS = ['voltage_V', 'current_A', 'voltage_difference_V', 'temperature_C']
CONTEXT_NAMES = ['completed_rw_count', 'elapsed_days', 'cumulative_rw_discharge_Ah',
                 'last_duration_s', 'last_discharge_Ah', 'last_temperature_missing_fraction',
                 'last10_duration_mean_s', 'last10_duration_std_s', 'last10_duration_slope',
                 'last10_voltage_mean_V', 'last10_voltage_std_V',
                 'last10_current_mean_A', 'last10_current_std_A',
                 'last10_temperature_mean_C', 'last10_temperature_std_C']

def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))

def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def arr(r, key):
    return np.atleast_1d(r[key]).astype(np.float64)

def groups(steps):
    group = []
    for i, r in enumerate(steps):
        if r['comment'] in RW:
            group.append(i)
        elif group:
            if any(steps[j]['comment'] == 'discharge (random walk)' for j in group):
                yield group
            group = []
    if group and any(steps[j]['comment'] == 'discharge (random walk)' for j in group):
        yield group

def build_cell(cell, *, maximum_current_A=5.0, minimum_temperature_C=-40.0):
    if not np.isfinite(maximum_current_A) or maximum_current_A <= 0:
        raise ValueError('maximum_current_A must be positive and finite')
    if not np.isfinite(minimum_temperature_C) or minimum_temperature_C >= 100:
        raise ValueError('minimum_temperature_C must be finite and below 100 C')
    path = ROOT / f'data/raw/{cell}.mat'
    steps = loadmat(path, simplify_cells=True)['data']['step']
    cycle_rows, features, refs = [], [], []
    for ci, group in enumerate(groups(steps)):
        ds = [j for j in group if steps[j]['comment'] == 'discharge (random walk)']
        # Include internal brief rests, exclude rest before first/after last discharge.
        ids = [j for j in group if ds[0] <= j <= ds[-1]]
        t = np.concatenate([arr(steps[j], 'time') for j in ids])
        vals = np.concatenate([np.stack([arr(steps[j], k) for k in ['voltage','current','temperature']], axis=1) for j in ids])
        reasons = []
        if not np.isfinite(t).all() or np.any(np.diff(t) < 0):
            raise ValueError(f'{cell} cycle {ci}: nonfinite or reversed timestamps')
        t, keep = np.unique(t, return_index=True); vals = vals[keep]
        vals[:, 2][(vals[:, 2] < minimum_temperature_C) | (vals[:, 2] > 100) | (~np.isfinite(vals[:, 2]))] = np.nan
        if len(t) < 2 or t[-1] <= t[0]:
            reasons.append('insufficient_time_samples')
        if not np.isfinite(vals[:, :2]).all():
            reasons.append('invalid_voltage_or_current')
        previous = group[0] - 1
        charge_end = float(arr(steps[previous], 'voltage')[-1]) if previous >= 0 else None
        if previous < 0 or steps[previous]['comment'] != 'charge (after random walk discharge)':
            reasons.append('no_preceding_rw_charge')
        elif charge_end < 4.15:
            reasons.append('preceding_charge_below_4.15V')
        if vals[-1, 0] > 3.25:
            reasons.append('discharge_did_not_reach_3.25V')
        if np.any((vals[:, 0] < 2.5) | (vals[:, 0] > 4.5)):
            reasons.append('voltage_outside_2.5_to_4.5V')
        if np.any((vals[:, 1] < -0.2) | (vals[:, 1] > maximum_current_A)):
            reasons.append('current_outside_expected_range')
        grid = np.linspace(t[0], t[-1], 128)
        v, current, temp = [np.interp(grid, t, vals[:, k]) for k in range(3)]
        x = np.stack([v, current, np.diff(v, prepend=v[0]), temp], axis=1).astype(np.float32)
        features.append(x)
        # Integrate each discharge step separately; do not bridge logging gaps.
        ah = float(sum(np.trapezoid(arr(steps[j], 'current'), arr(steps[j], 'relativeTime')) for j in ds) / 3600)
        cycle_rows.append(dict(cycle_index=ci, first_raw_step=ds[0], last_raw_step=ds[-1],
            start_time_s=float(t[0]), end_time_s=float(t[-1]), duration_s=float(t[-1]-t[0]),
            discharge_Ah=ah, discharge_steps=len(ds), preceding_charge_end_V=charge_end,
            end_voltage_V=float(vals[-1, 0]), valid=not reasons, exclusion_reason=';'.join(reasons),
            temperature_missing_fraction=float(np.mean(~np.isfinite(x[:, 3]))),
            maximum_record_gap_s=float(np.max(np.diff(t))) if len(t)>1 else 0))
    for i, r in enumerate(steps):
        if r['comment'] != 'reference discharge':
            continue
        t, current, v = arr(r, 'relativeTime'), arr(r, 'current'), arr(r, 'voltage')
        good = len(t)>1 and np.isfinite(t).all() and np.isfinite(current).all() and np.all(np.diff(t)>=0)
        q = float(np.trapezoid(current, t)/3600) if good else 0.0
        valid = bool(good and q>0 and np.isfinite(v).all() and v[-1]<=3.25)
        refs.append(dict(reference_index=len(refs),raw_step=i,start_time_s=float(arr(r,'time')[0]),
                         end_time_s=float(arr(r,'time')[-1]),capacity_Ah=q,valid=valid))
    valid_refs = [r for r in refs if r['valid']]
    assert valid_refs and all(a['start_time_s']<b['start_time_s'] for a,b in zip(valid_refs,valid_refs[1:]))
    q0 = valid_refs[0]['capacity_Ah']
    x = np.array(features); ends = np.array([r['end_time_s'] for r in cycle_rows])
    assert np.all(np.diff(ends)>0)
    valid_cycles = np.array([r['valid'] for r in cycle_rows])
    duration = np.array([r['duration_s'] for r in cycle_rows]); ah = np.array([r['discharge_Ah'] for r in cycle_rows])
    contexts = []
    for i in range(len(x)):
        lo = max(0, i-9); win = x[lo:i+1]; dur = duration[lo:i+1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            moments = [float(fn(win[:, :, c])) for c in [0,1,3] for fn in [np.nanmean,np.nanstd]]
        contexts.append([i+1,(ends[i]-cycle_rows[0]['start_time_s'])/86400,float(ah[:i+1].sum()),
            duration[i],ah[i],cycle_rows[i]['temperature_missing_fraction'],float(dur.mean()),float(dur.std()),
            float(np.polyfit(np.arange(len(dur)),dur,1)[0]) if len(dur)>1 else 0.,*moments])
    contexts = np.array(contexts, dtype=np.float32)
    samples, seen = [], set()
    for r in refs:
        # Only operational cycles completed before the reference discharge are visible.
        n = int(np.searchsorted(ends,r['start_time_s'],side='left'))
        r['history_end_exclusive'] = n
        r['soh'] = r['capacity_Ah']/q0
        reason = ''
        if not r['valid']: reason = 'invalid_reference'
        elif n<50: reason = 'less_than_50_cycles'
        elif not np.all(valid_cycles[n-50:n]): reason = 'invalid_cycle_in_50_cycle_window'
        elif n in seen: reason = 'repeated_reference_same_rw_history'
        r['primary_exclusion_reason'] = reason
        if reason: continue
        seen.add(n)
        samples.append(dict(sample_id=f'{cell}_ref{r["reference_index"]:03d}',battery=cell,
            reference_index=r['reference_index'],history_end_exclusive=n,
            input_end_time_s=float(ends[n-1]),target_start_time_s=r['start_time_s'],
            reference_delay_s=float(r['start_time_s']-ends[n-1]),soh=r['soh'],
            capacity_Ah=r['capacity_Ah'],stage='early' if r['soh']>=0.9 else 'middle' if r['soh']>=0.8 else 'late'))
    rt=np.array([r['end_time_s'] for r in valid_refs]); ry=np.array([r['capacity_Ah']/q0 for r in valid_refs])
    # Secondary label only, no constant extrapolation. Never used in primary samples/context.
    dense=np.interp(ends,rt,ry,left=np.nan,right=np.nan).astype(np.float32)
    np.savez_compressed(OUT/f'{cell}.npz',x=x,observed_mask=np.isfinite(x),context=contexts,
        context_observed_mask=np.isfinite(contexts),cycle_valid=valid_cycles,cycle_end_time_s=ends,
        secondary_interpolated_soh=dense)
    save_csv(OUT/f'{cell}_cycles.csv',cycle_rows);save_csv(OUT/f'{cell}_references.csv',refs)
    stats=dict(raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),raw_steps=len(steps),
        rw_cycles=len(x),valid_rw_cycles=int(valid_cycles.sum()),reference_measurements=len(refs),
        primary_samples=len(samples),temperature_missing_fraction=float(np.mean(~np.isfinite(x[:,:,3]))),
        excluded_cycle_reasons=dict(Counter(r['exclusion_reason'] for r in cycle_rows if not r['valid'])),
        excluded_reference_reasons=dict(Counter(r['primary_exclusion_reason'] for r in refs if r['primary_exclusion_reason'])),
        initial_reference_capacity_Ah=q0,max_record_gap_s=max(r['maximum_record_gap_s'] for r in cycle_rows),
        primary_stage_counts=dict(Counter(r['stage'] for r in samples)),
        maximum_primary_reference_delay_hours=(max(r['reference_delay_s'] for r in samples)/3600 if samples else None))
    print(cell,json.dumps(stats),flush=True)
    return dict(x=x,context=contexts,rows=samples,valid=valid_cycles,ends=ends),stats

def fit_stats(values):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        mean=np.nanmean(values,axis=0);std=np.nanstd(values,axis=0)
    if not np.isfinite(mean).all():
        raise ValueError('A training feature is entirely missing; explicit feature removal is required')
    std=np.where(np.isfinite(std)&(std>1e-6),std,1.)
    return mean.astype(np.float32),std.astype(np.float32)

def create_fold(name,train,val,test,data):
    folder=OUT/'folds'/name;folder.mkdir(parents=True,exist_ok=True)
    signal_fit=[];context_fit=[];fit_indices={}
    for b in train:
        used=np.zeros(len(data[b]['x']),bool)
        for r in data[b]['rows']:
            n=r['history_end_exclusive'];used[n-50:n]=True
        fit_indices[b]=np.flatnonzero(used).tolist()
        signal_fit.append(data[b]['x'][used].reshape(-1,4))
        context_fit.append(data[b]['context'][[r['history_end_exclusive']-1 for r in data[b]['rows']]])
    mean,std=fit_stats(np.concatenate(signal_fit));cm,cs=fit_stats(np.concatenate(context_fit))
    scalers=dict(channel_mean=mean.tolist(),channel_std=std.tolist(),context_mean=cm.tolist(),context_std=cs.tolist(),
                 fit_cycle_indices=fit_indices,fit_sample_ids=[r['sample_id'] for b in train for r in data[b]['rows']])
    save_json(folder/'scalers.json',scalers)
    split=dict(train=[r['sample_id'] for b in train for r in data[b]['rows']],
               validation=[r['sample_id'] for b in val for r in data[b]['rows']],
               test=[r['sample_id'] for b in test for r in data[b]['rows']],
               train_batteries=train,validation_batteries=val,test_batteries=test)
    assert not(set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    save_json(folder/'split.json',split)
    for b,d in data.items():
        x=((np.where(np.isfinite(d['x']),d['x'],mean)-mean)/std).astype(np.float32)
        c=((np.where(np.isfinite(d['context']),d['context'],cm)-cm)/cs).astype(np.float32)
        assert np.isfinite(x).all() and np.isfinite(c).all()
        np.savez_compressed(folder/f'{b}.npz',x=x,context=c)
    return dict(name=name,train=len(split['train']),validation=len(split['validation']),test=len(split['test']))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data={};audit={}
    for b in CELLS:data[b],audit[b]=build_cell(b)
    rows=[r for b in CELLS for r in data[b]['rows']]
    save_csv(OUT/'samples.csv',rows)
    save_csv(OUT/'stage_summary.csv',[dict(battery=b,stage=stage,samples=sum(r['stage']==stage for r in data[b]['rows']))
             for b in CELLS for stage in ['early','middle','late']])
    folds=[]
    for test in CELLS:
        remaining=[b for b in CELLS if b!=test]
        for val in remaining:
            folds.append(create_fold(f'test_{test}__val_{val}',[b for b in remaining if b!=val],[val],[test],data))
        folds.append(create_fold(f'test_{test}__final',remaining,[],[test],data))
    save_json(OUT/'audit.json',audit);save_csv(OUT/'fold_summary.csv',folds)
    save_json(OUT/'manifest.json',dict(version='main_v1',batteries=CELLS,lookbacks=LBS,
        channels=CHANNELS,context_names=CONTEXT_NAMES,primary_target='measured reference discharge capacity / first valid reference capacity',
        input='x[n-lookback:n]; n=history_end_exclusive; all input cycles finish before target reference discharge',
        temperature_valid_range_C=[-40,100],resampling_points=128,
        resampling='linear in elapsed time; internal short rests included, post-discharge rest excluded; NaNs propagate',
        cycle_quality='preceding RW charge >=4.15V; discharge end <=3.25V; finite V/I; V within [2.5,4.5], I within [-0.2,5]',
        duplicate_timestamps='keep first observation',duplicate_reference='first eligible measurement per identical RW history',
        stage='analysis only: early SOH>=0.9, middle 0.8<=SOH<0.9, late SOH<0.8',
        missing='NaN and observed masks in canonical files; training means impute fold arrays',
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    report=['# 본 실험 전처리 — main_v1','',
        '원본 MAT에서 재생성했다. 기존 step 단위 재현 실험과 다른, 완전한 RW 방전 cycle 단위 데이터다.',
        '', '| Battery | RW cycles | Valid cycles | Reference measurements | Primary samples | Temperature missing |',
        '|---|---:|---:|---:|---:|---:|']
    for b,a in audit.items():
        report.append(f"| {b} | {a['rw_cycles']} | {a['valid_rw_cycles']} | {a['reference_measurements']} | {a['primary_samples']} | {a['temperature_missing_fraction']:.1%} |")
    report += ['', '## 기본 학습 데이터','',
        '- samples.csv의 각 행이 실제 reference discharge 용량을 타깃으로 하는 한 샘플이다. 같은 입력 이력의 반복 reference는 첫 유효 측정만 사용한다.',
        '- history_end_exclusive=n이면 입력은 x[n-L:n]이다. L=10/20/50 모두 같은 샘플을 사용한다. 기준 방전 자체는 입력에 포함하지 않는다.',
        '- 대표 reference 직전까지 완료된 RW cycles만 입력한다. reference 준비 충전/휴지/진단에 따른 시간 간격은 reference_delay_s에 남겼다.',
        '- 이 간격은 일부 셀에서 약 60시간에 이른다. 따라서 타깃은 마지막 RW cycle 직후의 직접 측정 SOH가 아니라, 이후 기준 진단에서 관측한 SOH다. 추가 민감도 분석 시 이 간격으로 필터링할 수 있다.',
        '- 불완전/비정상 cycle은 원래 인덱스에 보존하되, 최근 50회에 포함되면 그 샘플을 제외한다. 삭제 후 떨어진 이력을 연속 cycle처럼 붙이지 않는다.',
        '- 한 cycle은 128×4(V/I/dV/T). dV는 resampling 후 전압 차분으로 단위는 V이며 dV/dt가 아니다. 전체 방전 종료 후 사용할 수 있는 입력이다.',
        '- 온도 -4094°C 같은 값은 [-40,100]°C 기준으로 NaN 처리한다. 마스크를 보존하고 폴드별 학습 평균으로만 대체한다. 이 범위는 명시적인 전처리 기준이며 센서 교정 결과가 아니다.',
        '- Q0는 각 셀의 첫 유효 기준 용량이다. 테스트 셀도 초기 용량 측정이 제공되는 설정이며, 완전 무보정 신규 셀 설정은 아니다.',
        '', '## 분할과 Selector용 Context','',
        '- 외부 LOBO 4개, 각각 내부 validation 배터리 3가지: 총 12개 내부 fold. 학습 2개 / 검증 1개 / 테스트 1개 셀로 분리했다.',
        '- final fold 4개는 내부 검증으로 설정을 정한 뒤 3개 셀로 재학습할 때 사용한다. final에는 validation이 없으므로 외부 테스트로 epoch를 선택하면 안 된다.',
        '- 채널 통계는 학습 샘플의 최대 50-cycle 입력에 실제 등장하는 cycle의 합집합으로만 계산한다. Context 통계도 학습 샘플에서만 계산한다.',
        '- Context는 입력 마지막 cycle까지의 cycle age, 경과 시간, 누적 방전량 및 최근 10회 운용 통계다. 실제/보간 SOH, 타깃 용량, 미래 수명, stage, battery ID는 Context에 넣지 않았다.',
        '- 동일 외부 fold의 내부 validation 예측을 모으면 Selector용 out-of-fold label을 만들 수 있다. 외부 테스트 오차는 Selector 학습에 사용하지 않는다.',
        '- Stage는 SOH 0.9/0.8 임계값의 사후 분석용 구분이다. 각 stage에 샘플이 있는지는 별도로 확인해야 한다.',
        '- stage_summary.csv에 셀별 Stage 개수를 저장했다. Early 표본이 특히 적어 Stage별 최적 Lookback을 강하게 결론 내리기 어렵다.',
        '', '## 보조 라벨과 한계','',
        '- secondary_interpolated_soh는 기준 측정 완료 시각 사이에서만 선형 보간했다. 범위 밖은 NaN이다. 미래 기준 측정을 쓰는 보조 라벨이며 주 평가 및 Context에는 사용하지 않는다.',
        '- 실제 용량 라벨은 희소하다. 데이터 수를 늘리려고 보간 라벨을 실제 독립 측정값으로 보고하면 안 된다.',
        '- 원본 기록 사이 간격은 최대 약 225초다. 현재는 이를 포함해 선형 resampling하며 maximum_record_gap_s를 cycle 메타데이터에 저장했다. 촘촘한 원신호를 완전히 보존하는 표현은 아니다.',
        '- RW3 온도 결측이 높아 온도 제외 ablation이 필요하다. 결측 마스크 역시 셀을 구별하는 단서가 될 수 있다.',
        '- 같은 배터리의 인접 샘플은 독립이 아니며, 셀 4개 결과만으로 넓은 일반화를 주장할 수 없다.',
        '- Context 생성 비용 및 cycle embedding 캐시 비용은 이후 효율성 실험에 포함해야 한다.',
        '', '## 사용','',
        '```python',
        'from scripts.preprocessing.battery_data import BatteryDataset',
        'ds = BatteryDataset("test_RW3__val_RW4", "train", lookback=20)',
        'item = ds[0]',
        '# item["x"]: (20,128,4), item["observed_mask"]: same shape',
        '# item["context"]: (15,), item["soh"]: scalar',
        '```', '', '재생성: `.venv-repro/bin/python -m scripts.preprocessing.preprocess_main`',
        '검증: `.venv-repro/bin/python -m scripts.validation.verify_preprocessing`', '']
    (OUT/'README.md').write_text('\n'.join(report))
    print('DONE',len(rows),'primary samples;',len(folds),'folds',flush=True)

if __name__=='__main__':main()
