"""Import verified NASA room-temperature archives and prepare a separate dataset.

No Predictor/Selector training, model selection, or test performance evaluation.
The original main_v1 artifacts and original four-cell training entry points stay
intact. The canonical preprocessing implementation is reused in this process.
"""
from scripts.paths import PROJECT_ROOT, resolve_file, hash_matches
import argparse
import collections
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import scripts.preprocessing.preprocess_main as canonical

ROOT = PROJECT_ROOT
OLD_CELLS = ['RW3', 'RW4', 'RW5', 'RW6']
SOURCES = {
    'high': dict(md5='9a0fabc687ae3a8d990c9d6f2f0412d4',
        title='5. RW_Skewed_High_Room_Temp_DataSet_2Post.zip',
        nasa_url='https://data.nasa.gov/docs/legacy/ames/5.RW_Skewed_High_Room_Temp_DataSet_2Post.zip'),
    'low': dict(md5='14130c74dd2caa3c6fbda83df26eb55d',
        title='7. RW_Skewed_Low_Room_Temp_DataSet_2Post.zip',
        nasa_url='https://data.nasa.gov/docs/legacy/ames/7.RW_Skewed_Low_Room_Temp_DataSet_2Post.zip'),
}


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with resolve_file(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def copy_exact(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not hash_matches(source, digest(target)):
            raise ValueError(f'Refusing to overwrite different existing file: {target}')
    else:
        shutil.copyfile(source, target)


def import_archive(group, source, project):
    info = SOURCES[group]
    if digest(source, 'md5') != info['md5']:
        raise ValueError(f'{group}: incomplete download or checksum mismatch')
    raw = project / 'data/raw'
    cells = []
    with zipfile.ZipFile(source) as archive:
        if archive.testzip() is not None:
            raise ValueError(f'{group}: ZIP integrity check failed')
        for entry in archive.infolist():
            if '__MACOSX' in entry.filename or entry.is_dir():
                continue
            name = Path(entry.filename).name
            if re.fullmatch(r'RW\d+\.mat', name):
                destination = raw / name
                cells.append(Path(name).stem)
            elif name.lower().endswith('.rmd'):
                destination = raw / 'source_docs' / group / name
            else:
                continue
            content = archive.read(entry)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != content:
                raise ValueError(f'Refusing to overwrite different file: {destination}')
            if not destination.exists():
                destination.write_bytes(content)
    cells.sort(key=lambda name: int(name[2:]))
    if len(cells) != 4 or len(set(cells)) != 4 or set(cells) & set(OLD_CELLS):
        raise ValueError(f'Unexpected cells in {group}: {cells}')
    copy_exact(source, raw / 'source_archives' / info['title'])
    return dict(**info, record_url='https://zenodo.org/records/15277374',
                archive_sha256=digest(source), cells=cells)


def raw_quality(path):
    data = loadmat(path, simplify_cells=True)['data']
    steps = data['step']
    rw = [r for r in steps if r['comment'] == 'discharge (random walk)']
    currents = np.concatenate([np.atleast_1d(r['current']) for r in rw])
    temperatures = np.concatenate([np.atleast_1d(r['temperature']) for r in rw])
    valid = np.isfinite(temperatures) & (temperatures >= -40) & (temperatures <= 100)
    return dict(step_comments=dict(collections.Counter(str(r['comment']) for r in steps)),
        rw_current_min_A=float(np.nanmin(currents)), rw_current_max_A=float(np.nanmax(currents)),
        valid_rw_temperature_min_C=float(temperatures[valid].min()) if valid.any() else None,
        valid_rw_temperature_max_C=float(temperatures[valid].max()) if valid.any() else None,
        raw_rw_temperature_missing_fraction=float(1-valid.mean()),
        retained_below_zero_temperature_observations=int(np.count_nonzero(valid & (temperatures<0))),
        source_metadata={str(k): str(v)[:5000] for k,v in data.items() if k != 'step'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--high-archive', type=Path, required=True)
    parser.add_argument('--low-archive', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--version', default='expanded_v1')
    args = parser.parse_args()
    if not re.fullmatch(r'expanded_[A-Za-z0-9_]+', args.version):
        raise ValueError('Use an explicit expanded_* version, never main_v1')
    project = args.project_root.resolve()
    out = project / 'data/processed' / args.version
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite dataset version: {out}')
    provenance = {group: import_archive(group, getattr(args, group+'_archive'), project)
                  for group in SOURCES}
    cells = OLD_CELLS + sorted([c for p in provenance.values() for c in p['cells']],
                              key=lambda name: int(name[2:]))
    if len(set(cells)) != 12:
        raise ValueError('Expected 12 distinct cells across original and added data')
    # Assign roles before any model fitting or predictive performance evaluation.
    development = OLD_CELLS + sorted([c for p in provenance.values() for c in p['cells'][:2]],
                                    key=lambda name: int(name[2:]))
    final_test = sorted([c for p in provenance.values() for c in p['cells'][2:]],
                        key=lambda name: int(name[2:]))
    out.mkdir(parents=True)
    split = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        development_batteries=development, final_test_batteries=final_test,
        assignment='All previously inspected cells are development cells. Within each added '
                   'archive, the two lowest numeric cell IDs are development and the two '
                   'highest IDs are reserved final test. No performance-based assignment.',
        final_test_policy='No fitting, hyperparameter/threshold selection or SOH performance '
                          'evaluation. Raw quality checks and target construction only.',
        development_validation='Battery-disjoint validation. Within each development '
            'validation fold, regenerate all selector OOF targets using only its fitting '
            'cells; no Predictor producing training labels may use the validation cell.')
    json_write(out/'split_plan.json', split)
    json_write(out/'sources.json', provenance)
    canonical.ROOT = project
    canonical.OUT = out
    audit, quality, all_samples = {}, {}, []
    for cell in cells:
        quality[cell] = raw_quality(project/f'data/raw/{cell}.mat')
        data, audit[cell] = canonical.build_cell(cell, maximum_current_A=5.25)
        if not data['rows']:
            raise ValueError(f'{cell}: no eligible measured targets under unchanged criteria')
        all_samples.extend(data['rows'])
    canonical.save_csv(out/'samples.csv', all_samples)
    canonical.save_csv(out/'samples_development.csv', [r for r in all_samples if r['battery'] in development])
    canonical.save_csv(out/'samples_final_test.csv', [r for r in all_samples if r['battery'] in final_test])
    canonical.save_csv(out/'cell_inventory.csv', [dict(
        battery=c, source_group='original_uniform' if c in OLD_CELLS else next(g for g,p in provenance.items() if c in p['cells']),
        role='development' if c in development else 'reserved_final_test',
        rw_cycles=audit[c]['rw_cycles'], valid_rw_cycles=audit[c]['valid_rw_cycles'],
        reference_measurements=audit[c]['reference_measurements'], primary_samples=audit[c]['primary_samples'],
        temperature_missing_fraction=audit[c]['temperature_missing_fraction']) for c in cells])
    json_write(out/'audit.json', audit)
    json_write(out/'raw_quality.json', quality)
    flags=[]
    for cell,q in quality.items():
        if q['retained_below_zero_temperature_observations']:
            flags.append(dict(battery=cell,sample_id='',kind='retained_subzero_temperature',
                value=q['retained_below_zero_temperature_observations'],unit='raw_observations',
                action='Flag only: below-zero temperatures at room conditions may indicate sensor issues; '
                       'the inherited [-40,100] C rule has not been silently tightened.'))
    for r in all_samples:
        if r['reference_delay_s']>120*3600:
            flags.append(dict(battery=r['battery'],sample_id=r['sample_id'],kind='reference_delay_over_120h',
                value=r['reference_delay_s']/3600,unit='hours',
                action='Flag only: 120 h is a reporting threshold, not sample exclusion or model tuning. '
                       'The prediction horizon differs greatly from immediate SOH estimation.'))
    canonical.save_csv(out/'quality_flags.csv',flags)
    json_write(out/'manifest.json', dict(version=args.version, batteries=cells,
        lookbacks=canonical.LBS, channels=canonical.CHANNELS, context_names=canonical.CONTEXT_NAMES,
        resampling_points=128, primary_target='Measured reference capacity / first valid reference capacity',
        input='x[n-L:n], all cycles completed before reference discharge; common valid 50-cycle window',
        canonical_parser_sha256=digest(Path(canonical.__file__)),
        preparation_script_sha256=digest(__file__),
        quality_current_range_A=[-0.2, 5.25],
        current_bound_note='Source README specifies nominal loads up to 5 A; the expanded '
            'version allows a documented 5% measurement/overshoot margin (5.25 A) for '
            'every cell. This is a preprocessing allowance, not a sensor calibration. '
            'No current values are clipped or rescaled. main_v1 retains its original 5.0 A rule.',
        scalers='Not fitted. Fit within the Predictor/Selector training cell split only.',
        training_status='Data preparation only; no model trained or evaluated on this version.',
        holdout_status='Reserved final test; quality audited, not performance evaluated.',
        inherited_assumptions=['Q0 is available for every cell, including final-test cells.',
            'Secondary interpolated SOH is not a primary target or model input.',
            'Context excludes current/future reference capacity, SOH, stage and battery ID.',
            'Resampling, missing temperature and charge/end-voltage criteria match main_v1; '
            'the upper current validity bound is explicitly 5.25 A for all expanded-version cells.',
            'Original training entry points default to main_v1; they do not automatically consume this version.']))
    report = ['# 추가 배터리 데이터 준비', '',
        'NASA 공개 상온 전류 편향 자료를 추가하고 기존과 동일한 전처리로 별도 버전을 생성했습니다.',
        '개발/최종 평가 역할은 모델 학습 및 예측 성능 확인 전에 셀 ID로 확정했습니다.', '',
        '| 셀 | 역할 | 유효 RW cycles | 실제 측정 기반 학습·평가 샘플 | 온도 결측 |',
        '|---|---|---:|---:|---:|']
    for c in cells:
        a=audit[c]
        role='개발' if c in development else '최종 평가 예약'
        report.append(f"| {c} | {role} | {a['valid_rw_cycles']} | {a['primary_samples']} | {a['temperature_missing_fraction']:.1%} |")
    report += ['', f"- 전체 {len(cells)}개 셀, 공통 50-cycle 이력이 있는 측정 기반 샘플 {len(all_samples)}개입니다.",
        '- 같은 셀의 샘플은 독립 배터리가 아닙니다. 보간 SOH는 정답 개수에 포함하지 않았습니다.',
        '- 상온은 시험 환경 분류입니다. 기록된 셀 온도는 운용 중 상승할 수 있습니다.',
        '- 원본 README와 raw_quality.json에 기록 방식·온도 품질을 보존했습니다.',
        '- 명목 최대 전류 5A의 측정 편차를 고려해 확장판의 전류 상한은 모든 셀에 5.25A를 적용했습니다. 센서 교정값은 아니며 전류를 잘라내거나 보정하지 않았습니다.',
        '- 기존 main_v1은 5.0A 상한을 유지합니다. 공통 셀의 샘플 및 신호 일치 여부는 별도 검증합니다.',
        '- 모든 배터리의 초기 기준 용량 Q0를 아는 설정입니다.',
        '- quality_flags.csv의 온도·측정 지연 표시는 품질 점검 항목입니다. 샘플을 임의 제거하거나 원인을 확정하지 않았습니다.',
        '- 상온 자료에서도 기존 [-40,100]°C 범위 안에 음수 온도가 남을 수 있습니다. 마스크 검증 통과가 센서 정확성을 보장하지 않습니다.',
        '- 마지막 입력 cycle과 기준 용량 측정 사이 120시간 초과 간격을 별도 표시했습니다. 120시간은 보고용 기준이며 제외 기준이 아닙니다.',
        '- 기존 네 셀의 main_v1 결과는 유지합니다. 새 버전의 학습/검증 코드는 split_plan.json을 명시적으로 사용해야 합니다.',
        '- 최종 평가용 셀은 Predictor·Selector 학습, 임계값·기본 길이 선택에서 제외합니다.',
        '- 이번 작업은 데이터 준비입니다. SOH 예측 성능 개선을 검증한 결과가 아닙니다.', '']
    (out/'DATA_REPORT.md').write_text('\n'.join(report))
    print(json.dumps(dict(output=str(out), cells=len(cells), primary_samples=len(all_samples),
        development=development, final_test=final_test), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
