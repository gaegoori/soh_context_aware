"""Plot verified temporal Context Ridge results and write a concise findings note."""
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

PROJECT=PROJECT_ROOT
ORDER=('Usage3','LastState6','FixedContext10','FixedContext20','FixedContext50',
       'GlobalBestOOF','LabelMajority','Adaptive','ErrorAware','Oracle')
LABEL=dict(Usage3='Cumulative use (3 features)',LastState6='Use + last cycle (6 features)',
    FixedContext10='Fixed context LB10',FixedContext20='Fixed context LB20',
    FixedContext50='Fixed context LB50',GlobalBestOOF='OOF-selected fixed',
    LabelMajority='Majority-label fixed',Adaptive='Classification selector',
    ErrorAware='Error-predicting selector',Oracle='Oracle (post hoc)')
COLOR=dict(Usage3='#BDC4CA',LastState6='#9DA9B1',FixedContext10='#2E6E68',
    FixedContext20='#729A95',FixedContext50='#A7BDBA',GlobalBestOOF='#3E617B',
    LabelMajority='#818C95',Adaptive='#8C6EAA',ErrorAware='#C07843',Oracle='#D0D2D7')


def read(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))


def digest(path):return hashlib.sha256(resolve_file(path).read_bytes()).hexdigest()


def write_table(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT/'outputs/temporal_context_ridge_v1')
    output=parser.parse_args().output.resolve()
    p=json.loads((output/'protocol.json').read_text());v=json.loads((output/'verification.json').read_text())
    artifacts=json.loads((output/'artifact_manifest.json').read_text())
    sig=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    assert v['status']=='PASS' and v['protocol_sha256']==artifacts['protocol_sha256']==sig
    sources=('summary.csv','metrics_by_battery.csv','predictions.csv','selections.csv',
             'selection_diagnostics.csv','paired_comparisons.csv','labels.csv')
    for name in sources:assert hash_matches(output/name, artifacts['files_sha256'][name])
    summary={r['method']:r for r in read(output/'summary.csv')}
    cells=p['batteries'];lengths=p['lookbacks']
    battery={(r['outer_test_battery'],r['method']):r for r in read(output/'metrics_by_battery.csv')}
    predictions=read(output/'predictions.csv');selections=read(output/'selections.csv')
    diagnostics=read(output/'selection_diagnostics.csv')
    samples={r['sample_id']:r for r in read(Path(p['source_data_root'])/'samples.csv')}
    figures=output/'figures';figures.mkdir(exist_ok=True);files=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#263441',
        'axes.labelcolor':'#263441','xtick.color':'#586775','ytick.color':'#586775',
        'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,
        'grid.color':'#E4E9EC','figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none'})
    def save(fig,name):
        for ext in ('png','svg'):
            path=figures/f'{name}.{ext}';fig.savefig(path,dpi=200);files.append(path)
        plt.close(fig)
    mae=lambda m:float(summary[m]['battery_macro_mae'])*100
    cell_mae=lambda b,m:float(battery[b,m]['mae'])*100
    latency=lambda m:float(summary[m]['latency_median_ms'])

    fig,ax=plt.subplots(figsize=(10.8,7.1));fig.subplots_adjust(left=.30,right=.96,top=.86,bottom=.16)
    values=[mae(m) for m in ORDER]
    ax.barh(np.arange(len(ORDER)),values,height=.62,color=[COLOR[m] for m in ORDER])
    for i,value in enumerate(values):ax.text(value+.025,i,f'{value:.3f}',va='center')
    ax.set_yticks(np.arange(len(ORDER)),[LABEL[m] for m in ORDER]);ax.invert_yaxis()
    ax.set_xlim(0,max(values)*1.16);ax.set_xlabel('Battery-macro SOH MAE (percentage points; lower is better)')
    ax.grid(axis='x');ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
    fig.suptitle('Selecting the window for Context Ridge predictors',x=.05,ha='left',fontsize=16,weight='bold')
    fig.text(.05,.04,'Exploratory: 4 previously inspected cells, 62 measured SOH targets.\nDeterministic fits: no artificial seed repeats or error bars. Oracle is a post-hoc reference.',fontsize=9,color='#586775')
    save(fig,'temporal_context_mae')

    fig,ax=plt.subplots(figsize=(10.2,5.8));fig.subplots_adjust(left=.11,right=.96,top=.85,bottom=.22)
    positions=np.arange(len(cells));width=.30;all_deltas=[]
    for j,method in enumerate(('Adaptive','ErrorAware')):
        delta=np.array([cell_mae(b,method)-cell_mae(b,'FixedContext10') for b in cells]);all_deltas.extend(delta)
        x=positions+(j-.5)*width
        ax.bar(x,delta,width,color=COLOR[method],label=LABEL[method])
        for xx,value in zip(x,delta):
            ax.annotate(f'{value:+.3f}',(xx,value),xytext=(0,5 if value>=0 else -6),textcoords='offset points',
                        ha='center',va='bottom' if value>=0 else 'top',fontsize=9)
    ax.axhline(0,color='#586775',lw=1);ax.grid(axis='y');ax.set_xticks(positions,cells)
    ax.set_ylim(min(all_deltas)-.12,max(all_deltas)+.18)
    ax.set_ylabel('MAE difference vs fixed LB10 (percentage points)');ax.legend(frameon=False,loc='upper left')
    fig.suptitle('Does selection improve on the strong LB10 baseline?',x=.05,ha='left',fontsize=16,weight='bold')
    fig.text(.11,.06,'Negative = lower error than fixed LB10. Positive = higher error.\nAll four cells are shown; results are not an independent confirmation.',fontsize=10,color='#586775')
    save(fig,'selector_vs_fixed10_by_cell')

    fig,axes=plt.subplots(2,2,figsize=(12,8.2),sharey=True)
    fig.subplots_adjust(left=.08,right=.97,bottom=.15,top=.83,hspace=.35,wspace=.15)
    for ax,b in zip(axes.ravel(),cells):
        truths=sorted([r for r in predictions if r['outer_test_battery']==b and r['method']=='FixedContext10'],
                      key=lambda r:int(samples[r['sample_id']]['history_end_exclusive']))
        n=[int(samples[r['sample_id']]['history_end_exclusive']) for r in truths]
        ax.scatter(n,[float(r['soh_true'])*100 for r in truths],s=24,c='#263441',label='Measured SOH',zorder=4)
        for method in ('FixedContext10','Adaptive','ErrorAware'):
            values={r['sample_id']:float(r['soh_pred'])*100 for r in predictions if r['outer_test_battery']==b and r['method']==method}
            ax.plot(n,[values[r['sample_id']] for r in truths],color=COLOR[method],lw=1.6,
                    label=LABEL[method],linestyle='--' if method=='ErrorAware' else '-')
        ax.set_title(f'{b} (n = {len(truths)})',loc='left',weight='bold');ax.grid(alpha=.8)
        ax.set_xlabel('Completed random-walk cycles');ax.set_ylabel('SOH (%)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc='upper center',bbox_to_anchor=(.55,.94),frameon=False,fontsize=10)
    fig.suptitle('Predictions at the same measured reference targets',x=.05,ha='left',fontsize=16,weight='bold')
    fig.text(.08,.04,'Dots are measured reference SOH. Lines connect predictions at those same targets.\nAll methods use only history available before each target; exploratory evaluation on four previously inspected cells.',fontsize=9,color='#586775')
    save(fig,'temporal_context_soh_predictions')

    fig,ax=plt.subplots(figsize=(10.5,5.8));fig.subplots_adjust(left=.10,right=.65,top=.85,bottom=.22)
    for method in ORDER[:-1]:
        ax.scatter(latency(method),mae(method),s=95,color=COLOR[method],
                   marker='D' if method in ('Adaptive','ErrorAware') else 'o',label=LABEL[method],edgecolor='white')
    ax.grid();ax.set_xlim(0,max(latency(m) for m in ORDER[:-1])*1.13)
    ys=[mae(m) for m in ORDER[:-1]];ax.set_ylim(min(ys)-.07,max(ys)+.07)
    ax.set_xlabel('Median inference latency (ms)');ax.set_ylabel('Battery-macro SOH MAE (percentage points)')
    ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),frameon=False,fontsize=9)
    fig.suptitle('Measured accuracy and inference cost',x=.05,ha='left',fontsize=16,weight='bold')
    fig.text(.10,.06,'Lower and further left is better. CPU 1 thread, batch 1; all methods timed in the same run.\nIncludes context extraction, selection and prediction; excludes raw preprocessing and model loading.',fontsize=9,color='#586775')
    save(fig,'temporal_context_accuracy_cost')

    structure=[]
    for b in cells:
        meta=json.loads((output/'models'/f'test_{b}'/'selectors'/'fit.json').read_text())
        assert hash_matches(output/'models'/f'test_{b}'/'selectors'/'fit.json', artifacts['files_sha256'][f'models/test_{b}/selectors/fit.json'])
        record=dict(outer_test_battery=b,
            global_best_oof_lookback=lengths[int(np.argmin(meta['oof_cell_macro_mae']))],
            posthoc_best_fixed_lookback=min(lengths,key=lambda lb:cell_mae(b,f'FixedContext{lb}')),
            **{f'oof_cell_macro_mae_lb{lb}':meta['oof_cell_macro_mae'][j] for j,lb in enumerate(lengths)})
        for method in ('Adaptive','ErrorAware'):
            count=Counter(int(r['selected_lookback']) for r in selections if r['outer_test_battery']==b and r['method']==method)
            record.update({f'{method}_selected_lb{lb}':count[lb] for lb in lengths})
        structure.append(record)
    write_table(output/'selection_structure.csv',structure);files.append(output/'selection_structure.csv')
    paired_wins=lambda m,ref:sum(cell_mae(b,m)<cell_mae(b,ref) for b in cells)
    negative=sum(int(r['negative_score_count']) for r in selections if r['method']=='ErrorAware')
    lines=['# 길이별 Context Ridge와 Selector 비교 결과','',
        '기존 Context Ridge를 LB10 기준선으로 재현하고, 최근 통계 산출 구간을 20·50 cycles로 확장했습니다.',
        '새 Predictor의 OOF 오차로 분류형 Selector와 오차 예측형 Selector를 각각 다시 학습했습니다.',
        '두 Selector의 명칭은 학습 방식이 같다는 뜻이며, 앞선 신호 기반 실험의 학습 가중치나 라벨을 재사용하지 않았습니다.','',
        '## 실험 조건','',
        '- RW3~RW6의 동일한 측정 SOH 62개를 평가했습니다. 배터리 단위 분할과 기존 SOH 정답을 유지했습니다.',
        '- 세 Predictor 모두 15개 입력입니다. 공통 이력·마지막 cycle 변수 6개는 유지하고, 최근 구간 통계 9개의 계산 범위만 바꾸었습니다.',
        '- 누적 변수에는 L 이전의 이력이 포함됩니다. L은 모든 과거 정보의 최대 범위가 아닌 최근 통계의 산출 범위입니다.',
        '- OOF Predictor 36개, 최종 Predictor·대조군 20개, Selector 8개를 학습했습니다. 모두 설정을 고정하고 튜닝 없이 평가했습니다.',
        '- 결정적인 모델이므로 seed 반복을 하지 않았습니다. 기존 Context Ridge의 변수·학습 계수·예측을 재현했고 독립 검증을 통과했습니다.','',
        '## 전체 결과','',
        '| 방법 | SOH MAE (%p) | 평균 선택 길이 | 추론 중앙값 (ms) |',
        '|---|---:|---:|---:|']
    names={'Usage3':'누적 사용 이력 3개','LastState6':'누적 이력·마지막 상태 6개',
        'FixedContext10':'고정 Context LB10 (기존 Context 회귀)','FixedContext20':'고정 Context LB20',
        'FixedContext50':'고정 Context LB50','GlobalBestOOF':'OOF 고정 선택','LabelMajority':'OOF 다수 라벨 고정',
        'Adaptive':'분류형 Selector','ErrorAware':'오차 예측형 Selector','Oracle':'Oracle (사후 비교)'}
    for method in ORDER:
        r=summary[method]
        length='—' if r['mean_lookback']=='' else f"{float(r['mean_lookback']):.2f}"
        time='—' if r['latency_median_ms']=='' else f"{float(r['latency_median_ms']):.4f}"
        lines.append(f'| {names[method]} | {mae(method):.3f} | {length} | {time} |')
    lines += ['', 'MAE는 각 배터리의 MAE를 동일 가중 평균한 값입니다. 샘플 전체 평균과 구분합니다.',
        '추론 시간은 이번 실행의 같은 환경에서 재측정했습니다. Context 계산·선택·정규화·해당 Ridge 계산을 포함하며, 원시 전처리와 초기 캐시 구축은 제외합니다.','',
        '## Selector의 추가 효과','']
    for method in ('Adaptive','ErrorAware'):
        d_fixed=mae(method)-mae('FixedContext10');d_global=mae(method)-mae('GlobalBestOOF')
        lines.append(f'- {names[method]}: 고정 LB10 대비 {d_fixed:+.3f}%p, OOF 고정 선택 대비 {d_global:+.3f}%p입니다. 고정 LB10보다 낮은 오차를 보인 셀은 {paired_wins(method,"FixedContext10")}/4개입니다.')
    if all(mae(m)>=mae('FixedContext10') for m in ('Adaptive','ErrorAware')):
        lines.append('- 이번 실행에서는 두 Selector 모두 고정 LB10보다 전체 MAE가 컸습니다. 강한 고정 기준선 대비 적응형 선택의 추가 이득은 확인하지 못했습니다.')
    lines += [f'- 고정 LB10과 Oracle의 차이는 {mae("FixedContext10")-mae("Oracle"):.3f}%p입니다. 후보 간 선택의 사후 개선 여지는 있지만, 관측 가능한 Context로 그 선택을 학습할 수 있음을 뜻하지는 않습니다.',
        f'- 추론 중앙값은 고정 LB10 {latency("FixedContext10"):.4f} ms, 분류형 {latency("Adaptive"):.4f} ms, 오차 예측형 {latency("ErrorAware"):.4f} ms입니다. 평균 L 감소만으로 효율 개선을 판단하지 않습니다.','',
        '## 배터리별 결과','',
        '| 셀 | 고정 LB10 | 고정 LB20 | 고정 LB50 | OOF 고정 | 분류형 Selector | 오차 예측형 Selector | Oracle |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for b in cells:
        ms=('FixedContext10','FixedContext20','FixedContext50','GlobalBestOOF','Adaptive','ErrorAware','Oracle')
        lines.append('| '+b+' | '+' | '.join(f'{cell_mae(b,m):.3f}' for m in ms)+' |')
    lines += ['', '단위는 SOH MAE (%p)입니다. 모든 셀을 표시하며, 좋은 셀만 선택해 집계하지 않았습니다.','',
        '## 선택 결과 진단','',
        '| 셀 | OOF로 정한 고정 L | 테스트 결과상 최저 MAE의 고정 L | 분류형 선택 10 / 20 / 50 | 오차 예측형 선택 10 / 20 / 50 |',
        '|---|---:|---:|---|---|']
    for r in structure:
        counts=[' / '.join(str(r[f'{m}_selected_lb{lb}']) for lb in lengths) for m in ('Adaptive','ErrorAware')]
        lines.append(f'| {r["outer_test_battery"]} | {r["global_best_oof_lookback"]} | {r["posthoc_best_fixed_lookback"]} | {counts[0]} | {counts[1]} |')
    lines += ['', '테스트 결과상 최저 고정 길이는 사후 진단값입니다. 모델 학습이나 길이 선택에는 사용하지 않았습니다.',
        f'오차 예측형 Selector의 186개 오차 점수 중 음수는 {negative}개입니다. 선형 외삽 점수를 자르지 않고 선택 순위로 사용했습니다.','',
        '## 해석 범위와 다음 판단','',
        '- 전체 SOH 오차가 앞선 신호 기반 모델보다 줄어든 변화와 Selector의 추가 효과를 구분해야 합니다. Selector의 효과는 같은 Ridge 후보를 사용한 고정 방식과 비교합니다.',
        '- 두 Selector는 새 OOF 정답으로 재학습했습니다. 앞선 Predictor에서 관측한 개선이 이번 Predictor에도 유지되는지는 위 짝지은 비교로 판단합니다.',
        '- 현재 네 셀은 이전 실험에서 이미 확인한 데이터이므로 탐색 결과입니다. 통계적으로 유의한 일반화 개선이나 연구 가설의 최종 검증으로 보고하지 않습니다.',
        '- OOF 모델은 2개 셀, 최종 모델은 3개 셀로 학습되어, 라벨 생성 때의 후보 우열과 실제 평가 때의 우열이 달라질 수 있습니다.',
        '- 이번 설정과 결과를 보존합니다. 추가 셀의 사용 가능성을 확인하고 개발·최종 평가 셀을 먼저 구분한 뒤, 길이 선택의 이득과 OOF→최종 모델 간 차이를 재검증하는 것이 다음 단계입니다.','',
        '## 파일','',
        '- `summary.csv`, `metrics_by_battery.csv`: 전체 및 셀별 정확도',
        '- `paired_comparisons.csv`, `selection_diagnostics.csv`, `selection_structure.csv`: 추가 효과와 선택 진단',
        '- `oof_predictions.csv`, `labels.csv`: 새 Ridge 후보의 OOF 오차와 정답',
        '- `verification.json`: 독립 검증 PASS 기록',
        '- `figures/temporal_context_mae.png`: 전체 오차 비교',
        '- `figures/selector_vs_fixed10_by_cell.png`: 고정 LB10 대비 셀별 차이',
        '- `figures/temporal_context_soh_predictions.png`: 실제 SOH와 예측 곡선',
        '- `figures/temporal_context_accuracy_cost.png`: 정확도와 실제 지연','']
    (output/'FINDINGS.md').write_text('\n'.join(lines),encoding='utf-8');files.append(output/'FINDINGS.md')
    (figures/'manifest.json').write_text(json.dumps(dict(protocol_sha256=sig,
        artifact_manifest_sha256=digest(output/'artifact_manifest.json'),verification_sha256=digest(output/'verification.json'),
        script_sha256=digest(Path(__file__)),source_sha256={name:digest(output/name) for name in sources},
        files_sha256={str(f.relative_to(output)):digest(f) for f in files}),indent=2)+'\n')
    print(output/'FINDINGS.md')
    for f in files:
        if f.suffix=='.png':print(f)


if __name__=='__main__':main()
