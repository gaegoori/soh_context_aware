"""Create concise figures and a Korean findings note from verified saved results.

Requires Matplotlib and NumPy; no model training or result-file rewriting.
"""
from scripts.paths import PROJECT_ROOT, resolve_file, hash_matches
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT = PROJECT_ROOT
LABELS = dict(Fixed10='Fixed LB10', Fixed20='Fixed LB20', Fixed50='Fixed LB50',
    GlobalBestOOF='OOF-selected fixed', LabelMajority='Majority-label fixed',
    Adaptive='Original selector', ErrorAware='Error-aware selector',
    ContextOnlyRidge='Context-only ridge', Oracle='Oracle (post hoc)')
COLORS = dict(Fixed10='#B5BDC3', Fixed20='#94A2AC', Fixed50='#516D80',
    GlobalBestOOF='#3F5F74', LabelMajority='#ACB5BD', Adaptive='#9278AB',
    ErrorAware='#197E79', ContextOnlyRidge='#86A09C', Oracle='#CED0D5')
ORDER = ('Fixed10','Fixed20','Fixed50','GlobalBestOOF','LabelMajority',
         'Adaptive','ErrorAware','ContextOnlyRidge','Oracle')


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(resolve_file(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT/'outputs/error_selector_v1')
    root=parser.parse_args().output.resolve()
    verification=json.loads((root/'verification.json').read_text())
    artifacts=json.loads((root/'artifact_manifest.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    sig=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    assert verification['status']=='PASS' and verification['protocol_sha256']==artifacts['protocol_sha256']==sig
    sources=('summary.csv','metrics_by_battery.csv','paired_comparisons.csv',
             'selections.csv','selection_diagnostics.csv','metrics_per_seed.csv')
    for name in sources:
        assert hash_matches(root/name, artifacts['files_sha256'][name])
    summary={r['method']:r for r in read(root/'summary.csv')}
    battery={(r['outer_test_battery'],r['method']):r for r in read(root/'metrics_by_battery.csv')}
    paired={(r['scope'],r['baseline']):r for r in read(root/'paired_comparisons.csv')}
    selections=read(root/'selections.csv')
    diagnostics=read(root/'selection_diagnostics.csv')
    seed_stats={(r['outer_test_battery'],int(r['seed']),r['method']):float(r['mae'])
                for r in read(root/'metrics_per_seed.csv')}
    cells=protocol['batteries']
    figures=root/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#263442',
        'axes.labelcolor':'#263442','xtick.color':'#536371','ytick.color':'#536371',
        'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,
        'grid.color':'#E5E9EC','figure.facecolor':'white','savefig.facecolor':'white',
        'svg.fonttype':'none'})
    paths=[]
    def save(fig,name):
        for ext in ('png','svg'):
            file=figures/f'{name}.{ext}'
            fig.savefig(file,dpi=200)
            paths.append(file)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(10.8,6.8))
    fig.subplots_adjust(left=.25,right=.96,top=.86,bottom=.17)
    values=np.array([float(summary[m]['battery_macro_mae'])*100 for m in ORDER])
    sd=np.array([float(summary[m]['macro_mae_seed_sd'])*100 for m in ORDER])
    ax.barh(np.arange(len(ORDER)),values,color=[COLORS[m] for m in ORDER],height=.62,
            xerr=sd,error_kw=dict(ecolor='#596773',capsize=3,lw=1))
    for i,(v,e) in enumerate(zip(values,sd)):
        ax.text(v+e+.13,i,f'{v:.2f}',va='center',weight='bold' if ORDER[i]=='ErrorAware' else 'normal')
    ax.set_yticks(np.arange(len(ORDER)),[LABELS[m] for m in ORDER])
    ax.invert_yaxis();ax.set_xlim(0,float(np.max(values+sd))+1.25)
    ax.set_xlabel('Battery-macro SOH MAE (percentage points; lower is better)')
    ax.grid(axis='x');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.suptitle('Selector comparison with the same frozen predictors',x=.05,ha='left',fontsize=16,weight='bold')
    fig.text(.05,.04,'Exploratory: 4 previously inspected cells, 62 SOH targets.\nError bars: SD across 5 predictor seeds, not a confidence interval. Oracle uses test truth after prediction.',fontsize=9,color='#586775')
    save(fig,'selector_mae_comparison')
    fig,ax=plt.subplots(figsize=(10.5,5.9))
    fig.subplots_adjust(left=.10,right=.96,top=.84,bottom=.23)
    positions=np.arange(len(cells));width=.30
    for i,(baseline,color) in enumerate((('Adaptive','#9278AB'),('GlobalBestOOF','#3F5F74'))):
        values=np.array([float(paired[b,baseline]['error_aware_minus_baseline_mae'])*100 for b in cells])
        x=positions+(i-.5)*width
        ax.bar(x,values,width,color=color,label=f'vs {LABELS[baseline]}')
        for px,v in zip(x,values):
            ax.annotate(f'{v:+.3f}',(px,v),xytext=(0,5 if v>=0 else -7),
                        textcoords='offset points',ha='center',va='bottom' if v>=0 else 'top',fontsize=9)
    ax.axhline(0,color='#596773',lw=1);ax.grid(axis='y')
    ax.set_xticks(positions,cells);ax.set_ylim(-1.4,1.0)
    ax.set_ylabel('MAE difference (percentage points)')
    ax.legend(loc='upper left',frameon=False,ncols=2,fontsize=10)
    fig.suptitle('Where did the new selector help?',x=.05,ha='left',fontsize=16,weight='bold')
    fig.text(.10,.07,'Difference = error-aware selector minus the comparator; negative means improvement.\nEach bar averages the same 5 predictor seeds. All four cells are shown.',fontsize=10,color='#586775')
    save(fig,'selector_paired_cell_changes')

    mae=lambda m:float(summary[m]['battery_macro_mae'])*100
    time_ms=lambda m:float(summary[m]['latency_median_ms'])
    new=mae('ErrorAware')
    wins={m:sum(float(paired[b,m]['error_aware_minus_baseline_mae'])<0 for b in cells)
          for m in ('Adaptive','GlobalBestOOF')}
    neg=sum(int(r['negative_score_count']) for r in selections)
    changed=sum(int(r['changed_from_old']) for r in diagnostics)
    matches=sum(int(r['matches_posthoc_best']) for r in diagnostics)
    counts={b:Counter(int(r['selected_lookback']) for r in selections if r['outer_test_battery']==b) for b in cells}
    lines=['# 오차 예측형 Selector — 첫 탐색 실험 결과','',
        '기존 신호 기반 Predictor를 고정하고, 최적 길이를 분류하던 Selector를 길이별 OOF 오차를 예측하는 Ridge 회귀로 바꾸었습니다.',
        '이는 신호 기반 Predictor에서 Selector만 비교한 실험입니다. 길이별 Context Ridge Predictor 실험은 아직 실행하지 않았습니다.','',
        '## 실행 조건','',
        '- RW3~RW6, 동일한 SOH 기준 측정 62개를 평가했습니다. 기존 Predictor 60개를 그대로 사용하고 새 Selector 4개만 학습했습니다.',
        '- 학습 정답은 해당 셀을 학습하지 않은 Predictor의 길이별 절대오차를 5개 seed에 걸쳐 평균한 값입니다.',
        '- Context 15개와 기존 학습 전용 정규화를 유지했습니다. 규제값 0.1을 고정했으며 이 실행에서 튜닝하지 않았습니다.',
        '- 모든 새 Selector를 학습한 뒤 평가했습니다. 기존 예측 재현, OOF 정답 재구성, 독립 회귀 계산 및 집계 검증을 통과했습니다.','',
        '## 전체 결과','',
        '| 방법 | SOH MAE (%p) | Predictor seed SD (%p) | 평균 선택 길이 | 추론 중앙값 (ms) |',
        '|---|---:|---:|---:|---:|']
    names={'Fixed50':'고정 LB50','GlobalBestOOF':'OOF 고정 선택','Adaptive':'기존 분류형 Selector',
           'ErrorAware':'새 오차 예측형 Selector','ContextOnlyRidge':'Context 직접 회귀','Oracle':'Oracle (사후 비교)'}
    for m in names:
        r=summary[m]
        length='—' if r['mean_lookback']=='' else f"{float(r['mean_lookback']):.2f}"
        latency='—' if r['latency_median_ms']=='' else f"{float(r['latency_median_ms']):.4f}"
        lines.append(f"| {names[m]} | {mae(m):.3f} | {float(r['macro_mae_seed_sd'])*100:.3f} | {length} | {latency} |")
    lines += ['', 'SOH MAE는 배터리별 MAE를 동일하게 평균한 값입니다. SD는 초기값이 다른 Predictor 5개의 변동이며, 통계적 유의성이나 모집단 신뢰구간을 나타내지 않습니다.',
        '추론 시간은 이번 실행에서 모든 방법을 함께 다시 측정한 값입니다. Context 산출·선택·정규화·선택 Predictor 실행을 포함하며 원시 전처리와 모델 로딩은 제외합니다.','',
        '## 확인한 점','',
        f'- 기존 Selector 대비 MAE가 {mae("Adaptive"):.3f} → {new:.3f}%p로 {mae("Adaptive")-new:.3f}%p 감소했습니다. 네 셀 중 {wins["Adaptive"]}개에서 seed 평균 오차가 감소했습니다.',
        f'- OOF 고정 선택 대비 MAE가 {mae("GlobalBestOOF"):.3f} → {new:.3f}%p로 {mae("GlobalBestOOF")-new:.3f}%p 감소했습니다. 네 셀 중 {wins["GlobalBestOOF"]}개에서 개선되었고 RW4에서는 악화되었습니다.',
        f'- 고정 LB50({mae("Fixed50"):.3f}%p)과 Context 직접 회귀({mae("ContextOnlyRidge"):.3f}%p)보다 오차가 큽니다. 기존 Selector 개선은 관측했지만 전체 방법의 우수성을 확립한 결과는 아닙니다.',
        f'- 추론 중앙값은 새 Selector {time_ms("ErrorAware"):.4f} ms, OOF 고정 선택 {time_ms("GlobalBestOOF"):.4f} ms입니다. 이 측정에서는 고정 선택보다 {time_ms("ErrorAware")/time_ms("GlobalBestOOF"):.2f}배 느렸습니다.','',
        '## 배터리별 비교','',
        '| 셀 | 기존 Selector MAE (%p) | 새 Selector MAE (%p) | OOF 고정 대비 차이 (%p) | 새 선택 LB10 / LB20 / LB50 |',
        '|---|---:|---:|---:|---|']
    for b in cells:
        c=counts[b]
        lines.append(f'| {b} | {float(battery[b,"Adaptive"]["mae_mean"])*100:.3f} | {float(battery[b,"ErrorAware"]["mae_mean"])*100:.3f} | {float(paired[b,"GlobalBestOOF"]["error_aware_minus_baseline_mae"])*100:+.3f} | {c[10]} / {c[20]} / {c[50]} |')
    lines += ['', 'OOF 고정 대비 차이는 새 Selector − OOF 고정 선택입니다. 음수이면 새 Selector의 오차가 작습니다.','',
        '## 진단과 다음 단계','',
        f'- 62개 샘플 중 {changed}개에서 기존 Selector와 다른 길이를 선택했습니다. 사후 seed 평균 최적 길이와의 일치는 {matches}/62이며, 주 성능 지표는 위 SOH 오차입니다.',
        f'- 186개 예상 오차 점수 중 {neg}개가 음수였습니다. 선형 회귀의 외삽 점수를 자르지 않고 순위로만 사용했습니다. 예상 오차의 보정 성능을 입증한 것은 아닙니다.',
        '- 현재 네 셀은 이전 결과를 보고 방법을 설계하는 데 사용했으므로 결과는 탐색적입니다. 유의미한 일반화 개선을 주장하려면 새 셀에서 독립 확인이 필요합니다.',
        '- OOF 정답 생성 모델은 2개 셀, 평가 Predictor는 3개 셀로 학습된 차이가 남습니다.',
        '- 다음 실험은 동일한 변수 종류와 개수를 유지한 LB10·20·50 Context Ridge 후보에서 기존/새 Selector를 비교하는 단계입니다. 이후 설정을 확정하고 새로운 셀에서 평가합니다.','',
        '## 결과 파일','',
        '- `summary.csv`: 모든 비교군의 전체 결과',
        '- `metrics_by_battery.csv`, `paired_comparisons.csv`: 셀별 결과와 방법 간 차이',
        '- `predictions.csv`, `selections.csv`: 원본 예측 및 선택 점수',
        '- `verification.json`: 검증 PASS 기록',
        '- `figures/selector_mae_comparison.png`: 전체 오차 비교',
        '- `figures/selector_paired_cell_changes.png`: 셀별 오차 변화','']
    (root/'FINDINGS.md').write_text('\n'.join(lines),encoding='utf-8')
    (figures/'manifest.json').write_text(json.dumps(dict(protocol_sha256=sig,
        source_sha256={name:digest(root/name) for name in sources},
        verification_sha256=digest(root/'verification.json'),script_sha256=digest(Path(__file__)),
        files_sha256={str(f.relative_to(root)):digest(f) for f in [*paths,root/'FINDINGS.md']}),indent=2)+'\n')
    print(root/'FINDINGS.md')
    for file in paths:print(file)


if __name__=='__main__':
    main()
