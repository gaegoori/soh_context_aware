"""Plot verified follow-up CSVs and write a Korean interpretation report.

Requires only NumPy and Matplotlib. No model training or result-based tuning.
All results and generated figures stay in the ignored local outputs directory.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
VARIANTS = ['WaveMean','WaveMetaMean','WaveMetaContextMean','WaveMetaContextTrend']
LABELS = ['Signals only','+ Duration / Ah','+ Context','+ Ordered trend']
GROUPS = ['ContextAll','ContextNoDurationAh','ContextUsageOnly','ContextNoTemperature']
GROUP_LABELS = ['All 15 features','Without duration / Ah','Usage history only (3)','Without temperature']
COLORS = ['#536E91','#1E8A83','#8662AE','#C16D48']
CELLS = ['RW3','RW4','RW5','RW6']


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, folder, name):
    for extension in ('png','pdf','svg'):
        fig.savefig(folder/f'{name}.{extension}',dpi=240,bbox_inches='tight')
    plt.close(fig)


def context_panel(ax, summary):
    vals = [float(summary[m,'']['macro_mae'])*100 for m in GROUPS]
    bars = ax.barh(np.arange(4),vals,color=['#1E8A83','#A4B3C5','#8662AE','#C2AD91'],height=.58)
    ax.set_yticks(np.arange(4),GROUP_LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0,max(vals)*1.27)
    for bar,value in zip(bars,vals):
        ax.text(value+.025,bar.get_y()+bar.get_height()/2,f'{value:.2f}',va='center',fontweight='bold')
    ax.set_xlabel('Battery-macro MAE (SOH percentage points)')
    ax.set_title('Context summaries remain a strong baseline',loc='left',pad=15)
    ax.grid(axis='x',alpha=.5)


def predictor_panel(ax, summary):
    for variant,label,color in zip(VARIANTS,LABELS,COLORS):
        vals = [float(summary[variant,str(lb)]['macro_mae'])*100 for lb in (10,20,50)]
        sd = [float(summary[variant,str(lb)]['seed_sd'])*100 for lb in (10,20,50)]
        ax.errorbar([0,1,2],vals,yerr=sd,fmt='o-',capsize=4,color=color,label=label,linewidth=2,markersize=6)
    ridge=float(summary['ContextAll','']['macro_mae'])*100
    ax.axhline(ridge,color='#303A44',ls='--',lw=1.5,label=f'Context ridge: {ridge:.2f}')
    ax.set_xticks([0,1,2],['10','20','50'])
    ax.set_xlim(-.17,2.18)
    ax.set_ylim(0,13)
    ax.set_xlabel('Lookback (completed cycles)')
    ax.set_ylabel('Battery-macro MAE (SOH percentage points)')
    ax.set_title('Additional inputs do not close the baseline gap',loc='left',pad=15)
    ax.grid(axis='y',alpha=.5)
    ax.legend(fontsize=9,loc='upper right',ncol=2)


def heatmap(ax, values, labels, title, vmax):
    im=ax.imshow(values,cmap='YlGnBu',vmin=0,vmax=vmax,aspect='auto')
    ax.set_xticks(range(4),CELLS)
    ax.set_yticks(range(len(labels)),labels)
    for i in range(len(labels)):
        for j in range(4):
            ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white' if values[i,j]>vmax*.6 else '#21313B')
    ax.set_title(title,loc='left',pad=15)
    ax.tick_params(length=0)
    return im


def write_report(out, summary, cellrows, fits):
    val=lambda m,lb='':float(summary[m,str(lb)]['macro_mae'])*100
    sd=lambda m,lb='':float(summary[m,str(lb)]['seed_sd'])*100
    cell=lambda m,b,lb='':np.mean([float(r['mae'])*100 for r in cellrows if r['method']==m and r['battery']==b and r['lookback']==str(lb)])
    lines=['# Context / 과거 이력 후속 실험 결과','',
        '## 결론','',
        '**현재 데이터에서는 길이 선택을 더 복잡하게 만들기보다, 강한 Context 회귀 기준선 위에서 과거 이력이 주는 추가 이득을 검증하는 것이 우선이다.**','',
        '- 방전 시간·방전량을 cycle별로 추가하면 Predictor 오차는 감소했다.',
        '- 단순히 Context를 신경망에 연결하거나 선형 순서 추세를 추가하는 방식은 외부 셀 오차를 개선하지 못했다.',
        '- 누적 사용 이력 3개 변수만으로도 강한 성능이 나왔다. 현재 셀들에서 사용량과 열화가 밀접한 가능성이 있으며, 이를 다양한 운용 조건으로 분리하여 검증해야 한다.',
        '- 신호 기반 Predictor에서는 긴 이력이 도움이 됐다. 따라서 “과거 이력은 필요 없다”는 결론은 낼 수 없다.','',
        '## 실험 범위','',
        '- RW3/RW4/RW5/RW6, 측정 SOH 목표 62개. 3개 셀 학습 / 1개 셀 평가를 4번 회전.',
        '- Context ridge 4설정 × 4분할 = 16개, 신경망 4설정 × 3길이 × 5seed × 4분할 = 240개.',
        '- 모든 신경망은 같은 37,633개 파라미터 텐서와 seed별 동일 초기값. 비활성 입력은 0으로 대체. 활성 정보와 유효 자유도는 다르다.',
        '- Adam lr=.001, full-batch MSE, 200 epochs 고정. 결과를 보고 설정이나 epoch를 다시 고르지 않았다.',
        '- MAE 단위는 SOH %p. 셀별 MAE를 동일 가중 평균하고, ±는 5개 seed의 표준편차이다.',
        '- 결정적 ridge는 분할당 1회 학습한다. 신뢰구간을 표현하는 반복 seed를 만들지 않았다.',
        '- **이전 테스트 결과를 본 뒤 설계한 탐색 실험이다. 새 독립 데이터에서 확인하기 전까지 확증 결과로 해석할 수 없다.**','',
        '## 1. 어떤 Context 묶음이 필요한가?','',
        '| Context 설정 | 변수 수 | MAE (%p) |', '|---|---:|---:|']
    for name,label,n in zip(GROUPS,['전체','방전 시간·방전량 관련 변수 제외','사용 이력만: cycle 수·경과 일수·누적 Ah','온도 통계·결측 비율 제외'],[15,9,3,12]):
        lines.append(f'| {label} | {n} | {val(name):.3f} |')
    lines += ['',f'전체에서 방전 시간·방전량 관련 변수를 제거해도 MAE는 {val("ContextNoDurationAh"):.2f}%p였다. 전체 {val("ContextAll"):.2f}%p 대비 증가는 {val("ContextNoDurationAh")-val("ContextAll"):.2f}%p이다. 따라서 기존 신경망과의 큰 성능 차이를 “방전량 정보가 빠졌기 때문” 하나로 설명할 수 없다.',
        '',f'사용 이력 3개만 사용한 회귀는 {val("ContextUsageOnly"):.2f}%p였다. 입력에 최근 신호 통계가 없어도 현재 네 셀에서는 상당한 SOH 정보를 제공한다. 다만 cycle 수·경과 시간·누적 Ah 각각의 기여는 이번 묶음 실험으로 구분할 수 없다.',
        '', 'Context 전체와 온도 제외 비교는 셀마다 방향이 다르다. 특히 RW3은 온도 제외 시 오차가 감소했지만 전체 셀 평균은 증가했다. 온도나 결측이 전체 실패의 단독 원인이라고 단정할 수 없다.',
        '', '| Context 설정 | RW3 | RW4 | RW5 | RW6 |','|---|---:|---:|---:|---:|']
    for m in GROUPS:
        lines.append('| '+m+' | '+' | '.join(f'{cell(m,b):.2f}' for b in CELLS)+' |')
    lines += ['','해석 한계: Context는 최근 10-cycle 요약과 누적 이력이다. History-free 모델이 아니다. 변수 묶음 제거는 상관된 다른 변수로의 대체를 허용하며 인과적 중요도 분석이 아니다. 과거 방전 Ah는 reference SOH와 별도 측정이지만 용량의 대리 정보일 수 있어 실제 배포 시점의 관측 가능성을 확인해야 한다.','',
        '## 2. Predictor 입력과 순서 표현을 바꾸면?','',
        '| Predictor | LB10 | LB20 | LB50 |','|---|---:|---:|---:|']
    korean=['신호 + 평균 집계','+ cycle별 방전 시간·방전량','+ 15개 Context','+ 순서 추세']
    for m,label in zip(VARIANTS,korean):
        lines.append('| '+label+' | '+' | '.join(f'{val(m,lb):.2f} ± {sd(m,lb):.2f}' for lb in (10,20,50))+' |')
    lines += ['',f'- LB50에서 메타데이터 추가: {val("WaveMean",50):.2f} → {val("WaveMetaMean",50):.2f}%p. LB10/20/50 각각 5개 seed 모두에서 셀 평균 오차가 감소했다.',
        f'- 동일 LB50에 Context 추가: {val("WaveMetaMean",50):.2f} → {val("WaveMetaContextMean",50):.2f}%p. 단순 연결로는 회귀 기준선의 장점을 재현하지 못했다.',
        f'- 순서 추세 추가: {val("WaveMetaContextMean",50):.2f} → {val("WaveMetaContextTrend",50):.2f}%p. 이번 선형 추세 표현으로는 개선되지 않았다.',
        '- 순서 추세는 cycle embedding을 상대 위치 [-1,1]에 회귀한 기울기이다. 모든 시계열 모델이나 모든 순서 정보가 불필요하다는 검증이 아니다.',
        '', '### 긴 기록 자체의 효과','',
        f'방전 정보를 추가한 Predictor에서는 LB10 {val("WaveMetaMean",10):.2f} → LB50 {val("WaveMetaMean",50):.2f}%p로 감소했다. 이 차이는 5개 seed 모두 같은 방향이다. 다만 전체 Context ridge {val("ContextAll"):.2f}%p를 넘지 못하므로, “긴 기록의 이득”과 “강한 요약 기준선 대비 추가 이득”을 구분해야 한다.',
        '', '### 남아 있는 길이 선택 여유','',
        '| 후보 Predictor | 사후 Oracle MAE (%p) |','|---|---:|']
    for m,label in zip(VARIANTS,korean):
        lines.append(f'| {label} | {val(m+"Oracle"):.2f} ± {sd(m+"Oracle"):.2f} |')
    lines += ['',f'가장 낮은 Oracle도 {min(val(m+"Oracle") for m in VARIANTS):.2f}%p로 Context ridge보다 높다. 이는 **이번에 만든 후보 Predictor들에서 세 길이 중 하나만 선택하는 방식**의 한계다. 다른 입력, 다른 모델, 예측 결합 방식까지 불가능하다는 의미는 아니다. Oracle은 테스트 정답을 사용한 사후 진단이며 실제 모델이 아니다.','',
        '## 3. 학습은 잘 되는데 새 셀에서 왜 나빠지는가?','',
        '| LB50 Predictor | 학습 RMSE 평균 (%p) | 외부 셀 RMSE 평균 (%p) |','|---|---:|---:|']
    for m,label in zip(VARIANTS,korean):
        train=float(np.mean([100*np.sqrt(f['training_mse']) for f in fits if f['method']==m]))
        test=100*float(summary[m,'50']['macro_rmse'])
        lines.append(f'| {label} | {train:.2f} | {test:.2f} |')
    lines += ['','Context·추세를 추가하면 학습 오차는 작아지지만 외부 셀 오차는 증가했다. 학습 목표 43–53개에 비해 모델 파라미터가 많다는 점과 함께 **과적합 및 셀 간 분포 차이 가능성**을 점검할 근거다. 이 비교만으로 원인을 확정할 수는 없다.',
        '', '## 4. 다음 연구 방향','',
        '**연구 질문을 “언제 긴 기록이 필요한가?”로 유지하되, 먼저 “사용 이력 요약만으로 설명되지 않는 정보가 긴 기록에 있는가?”를 검증한다.**','',
        '1. **강한 기준선부터 고정:** 전체 Context ridge와 사용 이력 3개 ridge를 기본 비교군으로 유지한다. cycle 수만 / 경과 시간만 / 누적 Ah만 사용한 기준선을 추가하면 누적 이력 내부의 기여를 더 구분할 수 있다.',
        '2. **작고 규제된 이력 모델:** LB10/20/50의 방전량·시간·전압 등의 통계/추세를 동일한 ridge 구조로 비교한다. 또는 Context 예측의 잔차를 이력으로 보정하되, 잔차 학습용 Context 예측은 내부 OOF로 만든다.',
        '3. **Selector로 돌아갈 조건:** 동일 Context 기준선 위에서 긴 이력이 추가 이득을 주고, 그 이득이 샘플 상태에 따라 달라질 때 새 Predictor로 OOF 라벨을 다시 만든다. 기존 라벨은 새 Predictor의 최적 길이를 나타내지 않는다.',
        '4. **독립 확인:** 새로운 셀/운용 조건을 확보한 뒤 설정을 잠그고 확인한다. 튜닝은 학습 셀 내부에서만 수행한다. 현재 네 셀 결과에 계속 맞추면 일반화 근거가 약해진다.',
        '', '이번 작업은 원인 진단용 ablation까지 완료했다. 위 다음 단계의 새 모델이나 Selector를 추가로 학습한 것은 아니다. 추론 비용은 이번에 재측정하지 않았다.','',
        '## 발표용 한 문단','',
        '> 이전 실험에서 Adaptive 방식이 단순 Context 회귀보다 좋지 않아, 입력 정보와 과거 기록의 역할을 분리하는 실험을 진행했습니다. 방전 시간과 방전량을 추가하면 Predictor는 개선됐지만, Context나 순서 추세를 신경망에 추가하는 것만으로는 성능 차이를 해소하지 못했습니다. 특히 세 가지 사용 이력만으로도 낮은 오차가 나와, 다음에는 이러한 강한 기준선에 긴 기록이 추가 정보를 제공하는지 작은 규제 모델로 확인할 계획입니다. 이후 이득이 확인되는 경우에 Adaptive Selector를 다시 검증하겠습니다.','',
        '## 검증과 파일','',
        '- verification.json: 240개 모델의 예측과 학습 오차 재현, 16개 ridge의 최적 조건, 학습 전용 정규화, 시점 경계, 지표·해시 확인 PASS.',
        '- 단위 검사: 비활성 입력 차단, 동일 초기값/파라미터 수, 평균 집계의 순열 불변성, 순서 추세 반응, 미래 데이터/정답 변조에 대한 입력 불변성.',
        '- ContextAll은 이전 ContextOnlyRidge의 예측을 재현했다. 기존 결과를 덮어쓰지 않았다.',
        '- predictions.csv / summary.csv / metrics_by_battery_seed.csv / paired_comparisons.csv: 원자료와 지표.',
        '- figures/figure_00_followup_overview.png: 발표용 요약. 동일 이름의 PDF/SVG 제공.',
        '- figures/figure_01_context_ablation.*, figure_02_predictor_inputs.*, figure_03_cell_breakdown.*, figure_04_train_test_gap.*: 개별 그림.',
        '- 코드/실행 문서 외 데이터·모델·결과·그림은 Git 제외, 로컬 저장.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=REPO/'outputs/context_history_v1')
    args=p.parse_args()
    out=args.results
    verification=json.loads((out/'verification.json').read_text())
    manifest=json.loads((out/'artifact_manifest.json').read_text())
    assert verification['status']=='PASS'
    assert verification['protocol_sha256']==manifest['protocol_sha256']
    for name in ('summary.csv','metrics_by_battery_seed.csv','paired_comparisons.csv'):
        assert sha(out/name)==manifest['files_sha256'][name]
    summary={(r['method'],r['lookback']):r for r in read(out/'summary.csv')}
    cellrows=read(out/'metrics_by_battery_seed.csv')
    fits=[]
    for b in CELLS:
        for m in VARIANTS:
            for s in (42,0,1,2,3):
                f=json.loads((out/'models'/f'test_{b}'/m/f'seed_{s}'/'lb_50'/'fit.json').read_text())
                fits.append(dict(f,method=m,battery=b,seed=s))
    folder=out/'figures'
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.titleweight':'bold',
        'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,'grid.color':'#D4DCE3',
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','legend.frameon':False,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(9,4.7),layout='constrained')
    context_panel(ax,summary)
    fig.suptitle('Exploratory follow-up | four held-out-cell rotations',fontsize=11,color='#657685')
    save(fig,folder,'figure_01_context_ablation')
    fig,ax=plt.subplots(figsize=(9,5.7),layout='constrained')
    predictor_panel(ax,summary)
    fig.suptitle('Mean ± SD across five seeds; four-cell macro MAE',fontsize=11,color='#657685')
    save(fig,folder,'figure_02_predictor_inputs')
    fig,axes=plt.subplots(1,2,figsize=(16,5.8),gridspec_kw={'width_ratios':[1,1.35]})
    context_panel(axes[0],summary)
    predictor_panel(axes[1],summary)
    fig.suptitle('What explains the SOH prediction gap?',x=.055,ha='left',fontsize=21,fontweight='bold')
    fig.text(.055,.90,'Exploratory follow-up on previously inspected cells | 62 measured targets | lower MAE is better',fontsize=11,color='#657685')
    fig.text(.055,.015,'Ridge: deterministic fits. Neural predictors: identical allocated parameters and seeded initial states; error bars show seed SD.',fontsize=10,color='#657685')
    fig.subplots_adjust(left=.17,right=.98,bottom=.17,top=.78,wspace=.45)
    save(fig,folder,'figure_00_followup_overview')
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    def cell_matrix(methods,lb):
        return np.array([[np.mean([float(r['mae'])*100 for r in cellrows if r['method']==m and r['battery']==b and r['lookback']==lb]) for b in CELLS] for m in methods])
    a=cell_matrix(GROUPS,''); b=cell_matrix(VARIANTS,'50')
    im=heatmap(axes[0],a,GROUP_LABELS,'Context ridge ablations',max(a.max(),.1))
    fig.colorbar(im,ax=axes[0],shrink=.75,label='MAE (%p)')
    im=heatmap(axes[1],b,LABELS,'Neural predictors at LB50',max(b.max(),.1))
    fig.colorbar(im,ax=axes[1],shrink=.75,label='MAE (%p)')
    fig.suptitle('Per-cell results (separate color scales)',fontsize=15,fontweight='bold')
    save(fig,folder,'figure_03_cell_breakdown')
    fig,ax=plt.subplots(figsize=(10,5.3),layout='constrained')
    train=[np.mean([100*np.sqrt(f['training_mse']) for f in fits if f['method']==m]) for m in VARIANTS]
    test=[100*float(summary[m,'50']['macro_rmse']) for m in VARIANTS]
    x=np.arange(4)
    bars1=ax.bar(x-.19,train,.36,label='Training',color='#92BDB5')
    bars2=ax.bar(x+.19,test,.36,label='Held-out cell',color='#536E91')
    ax.bar_label(bars1,fmt='%.2f',padding=3,fontsize=10)
    ax.bar_label(bars2,fmt='%.2f',padding=3,fontsize=10)
    ax.set_xticks(x,['Signals\nonly','+ Duration\n/ Ah','+ Context','+ Ordered\ntrend'])
    ax.set_ylabel('Mean RMSE (SOH percentage points)')
    ax.set_ylim(0,max(test)*1.22)
    ax.legend(ncol=2,loc='upper left')
    ax.grid(axis='y',alpha=.5)
    ax.set_title('Lower training error does not imply better cell generalization',loc='left',pad=17)
    fig.suptitle('LB50 | means across four rotations and five seeds',fontsize=11,color='#657685')
    save(fig,folder,'figure_04_train_test_gap')
    write_report(out,summary,cellrows,fits)
    (folder/'CAPTIONS.txt').write_text('모든 결과는 이전에 확인한 4개 셀에 대한 탐색 실험입니다.\n'
        'figure_00: Context 제거 실험과 Predictor 입력 추가 실험 요약.\n'
        'figure_01: ridge L2=.1 고정. 결정적 모델이므로 seed 오차 막대 없음.\n'
        'figure_02: 4셀 MAE 평균, ±5seed 표준편차. 독립 셀 신뢰구간 아님.\n'
        'figure_03: 셀별 MAE. 두 패널의 색상 범위가 다름. 신경망은 LB50, seed 평균.\n'
        'figure_04: LB50 학습/외부 셀 RMSE. 작은 학습 오차에도 외부 셀 오차가 높음을 보여 줌.\n'
        'Context에는 최근10 및 누적 이력이 포함됩니다. 이번에는 추론 시간을 재측정하지 않았습니다.\n')
    files=list(folder.glob('figure_*.*'))+[folder/'CAPTIONS.txt',out/'REPORT.md']
    record=dict(protocol_sha256=verification['protocol_sha256'],plot_script_sha256=sha(Path(__file__)),
        source_sha256={n:sha(out/n) for n in ('summary.csv','metrics_by_battery_seed.csv','verification.json')},
        generated_sha256={str(f.relative_to(out)):sha(f) for f in files})
    (folder/'figure_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Wrote Korean REPORT.md and 5 Matplotlib figures (PNG/PDF/SVG).')


if __name__=='__main__':
    main()
