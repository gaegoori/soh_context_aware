"""Freeze room-temperature data rules before development baseline evaluation."""
from scripts.paths import PROJECT_ROOT, SCRIPTS, resolve_file
import argparse
import collections
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scripts.preprocessing.preprocess_main as canonical

ROOT=PROJECT_ROOT
MIN_TEMPERATURE_C=0.
MAX_REFERENCE_DELAY_HOURS=120.


def sha(path):
    h=hashlib.sha256()
    with resolve_file(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    args=parser.parse_args();project=args.project_root.resolve()
    source=project/'data/processed/expanded_v1';out=project/'data/processed/expanded_qc_v1'
    if out.exists():raise FileExistsError('Refusing to replace existing quality dataset')
    check=json.loads((source/'verification.json').read_text())
    assert check['status']=='PASS'
    previous=json.loads((source/'manifest.json').read_text())
    split=json.loads((source/'split_plan.json').read_text())
    out.mkdir(parents=True)
    policy=dict(version='expanded_qc_v1',frozen_utc=datetime.now(timezone.utc).isoformat(),
        source_version='expanded_v1',temperature_valid_range_C=[0.,100.],
        temperature_rule='Mask raw temperatures outside [0,100] C before interpolation; '
                         'NaNs propagate. Room-temperature plausibility screen, not sensor calibration.',
        maximum_reference_delay_hours=120.,
        horizon_rule='All cells use measured references starting <=120 hours after the last input cycle. '
                     'A longer gap is excluded from this short-horizon task, not declared an invalid measurement.',
        rationale='The previously quality-flagged 38-day idle interval is outside a five-day prediction '
                  'horizon. Five days is a predeclared operational limit, not selected by predictive error. '
                  'Negative temperatures are implausible for the source room-temperature protocol.',
        current_valid_range_A=[-0.2,5.25],lookbacks=[10,20,50],
        all_cells_same_rules=True,model_performance_used_for_policy=False,
        reserved_final_test_policy='Quality preprocessing only; no model fitting or predictions.',
        source_input_sha256={str(p.resolve()):sha(p) for p in sorted(source.glob('*')) if p.is_file()},
        code_sha256={str(Path(__file__).name):sha(__file__),
                     'preprocess_main.py':sha(canonical.__file__),
                     'paths.py':sha(SCRIPTS/'paths.py'),
                     'source_layout.json':sha(SCRIPTS/'source_layout.json')})
    write_json(out/'quality_policy.json',policy)
    for name in ('split_plan.json','sources.json','raw_quality.json'):
        shutil.copyfile(source/name,out/name)
    canonical.ROOT=project;canonical.OUT=out
    samples=[];excluded=[];audits={};changes=[]
    for cell in previous['batteries']:
        data,audit=canonical.build_cell(cell,maximum_current_A=5.25,minimum_temperature_C=MIN_TEMPERATURE_C)
        eligible=[r for r in data['rows'] if r['reference_delay_s']<=MAX_REFERENCE_DELAY_HOURS*3600]
        dropped=[r for r in data['rows'] if r['reference_delay_s']>MAX_REFERENCE_DELAY_HOURS*3600]
        for r in dropped:excluded.append(dict(**r,exclusion_reason='reference_delay_over_120h'))
        references=read_csv(out/f'{cell}_references.csv')
        bad_refs={r['reference_index'] for r in dropped}
        for r in references:
            if int(r['reference_index']) in bad_refs:r['primary_exclusion_reason']='reference_delay_over_120h'
        canonical.save_csv(out/f'{cell}_references.csv',references)
        audit['primary_samples']=len(eligible)
        audit['primary_stage_counts']=dict(collections.Counter(r['stage'] for r in eligible))
        audit['excluded_reference_reasons']=dict(collections.Counter(r['primary_exclusion_reason'] for r in references if r['primary_exclusion_reason']))
        audit['maximum_primary_reference_delay_hours']=max((r['reference_delay_s']/3600 for r in eligible),default=None)
        audits[cell]=audit;samples.extend(eligible)
        with np.load(source/f'{cell}.npz',allow_pickle=False) as old, np.load(out/f'{cell}.npz',allow_pickle=False) as new:
            for key in ('cycle_valid','cycle_end_time_s','secondary_interpolated_soh'):
                np.testing.assert_array_equal(old[key],new[key])
            np.testing.assert_array_equal(old['x'][:,:,:3],new['x'][:,:,:3])
            same=(old['x'][:,:,3]==new['x'][:,:,3]) | (np.isnan(old['x'][:,:,3])&np.isnan(new['x'][:,:,3]))
            changes.append(dict(battery=cell,changed_resampled_temperature_points=int((~same).sum()),
                temperature_missing_fraction=audit['temperature_missing_fraction'],
                eligible_samples=len(eligible),excluded_for_horizon=len(dropped)))
    canonical.save_csv(out/'samples.csv',samples)
    for role,key in [('development','development_batteries'),('final_test','final_test_batteries')]:
        canonical.save_csv(out/f'samples_{role}.csv',[r for r in samples if r['battery'] in split[key]])
    canonical.save_csv(out/'excluded_samples.csv',excluded)
    canonical.save_csv(out/'quality_changes.csv',changes)
    write_json(out/'audit.json',audits)
    manifest={**previous,'version':'expanded_qc_v1','source_version':'expanded_v1',
        'canonical_parser_sha256':sha(canonical.__file__),'preparation_script_sha256':sha(__file__),
        'quality_temperature_range_C':[0.,100.],'maximum_reference_delay_hours':120.,
        'quality_policy_sha256':sha(out/'quality_policy.json'),
        'inherited_assumptions':['Initial reference Q0 available for each cell, including reserved final test.',
            'No current/future SOH, target capacity, stage, battery ID or reference delay in Context.',
            'Cumulative features include history older than L; L controls nine recent statistics.',
            'Five-day horizon limits target timing; it is not a fixed forecasting horizon.',
            'Nonnegative plausible temperatures can still contain sensor error.']}
    write_json(out/'manifest.json',manifest)
    report=['# 확장 데이터 품질 기준', '',
        '- 상온 자료의 0°C 미만/100°C 초과 온도는 원신호 단계에서 결측 처리하고, 시간 보간에서 결측을 전파했습니다.',
        '- 입력 종료 후 120시간 이내 시작한 기준 방전만 주 예측 대상으로 정했습니다. 더 긴 간격은 별도 보존했습니다.',
        '- 두 규칙은 모든 셀에 동일하게 적용하며, 예측 오차를 확인하기 전에 고정했습니다.',
        '- 5일은 이 실험의 예측 범위입니다. 장기 간격의 측정이 잘못되었다는 뜻은 아닙니다.',
        '- 기존 main_v1, expanded_v1, 최종 평가 셀 지정은 유지했습니다.', '',
        '| 셀 | 온도 변경 지점 | 측정 SOH 샘플 | 기간 초과 제외 |','|---|---:|---:|---:|']
    for r in changes:report.append(f"| {r['battery']} | {r['changed_resampled_temperature_points']} | {r['eligible_samples']} | {r['excluded_for_horizon']} |")
    report+=['','실제 사용 가능 샘플 수: '+str(len(samples)),
             '온도 범위 검사는 센서 교정이 아니며, 남은 온도 값의 정확성을 보장하지 않습니다.','']
    (out/'QUALITY_REPORT.md').write_text('\n'.join(report))
    print(json.dumps(dict(output=str(out),samples=len(samples),excluded=len(excluded)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
