"""Figures and a concise Korean note for the verified paired transfer study."""
from scripts.paths import PROJECT_ROOT, resolve_file, hash_matches
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT=PROJECT_ROOT
METHODS=('FixedContext10','FixedContext20','FixedContext50','CalibrationBest',
         'LabelMajority','Adaptive','ErrorAware','Oracle')
LABEL=dict(FixedContext10='Fixed context LB10',FixedContext20='Fixed context LB20',
    FixedContext50='Fixed context LB50',CalibrationBest='Calibration-selected fixed',
    LabelMajority='Majority-label fixed',Adaptive='Classification selector',
    ErrorAware='Error-predicting selector',Oracle='Oracle (post hoc)')
BANKS=('FrozenPair','RefitThree')
BANK_LABEL={'FrozenPair':'Keep the 2-cell predictors','RefitThree':'Use the 3-cell refit'}
COLOR={'FrozenPair':'#9BB1C1','RefitThree':'#41667E'}


def read(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))


def digest(path):return hashlib.sha256(resolve_file(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT/'outputs/selector_transfer_v1')
    root=parser.parse_args().output.resolve()
    p=json.loads((root/'protocol.json').read_text());v=json.loads((root/'verification.json').read_text())
    manifest=json.loads((root/'artifact_manifest.json').read_text())
    sig=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    assert v['status']=='PASS' and v['protocol_sha256']==manifest['protocol_sha256']==sig
    sources=('summary.csv','label_shift.csv','label_shift_by_battery.csv','bank_effects.csv',
             'paired_comparisons.csv','selector_diagnostics_by_battery.csv')
    for name in sources:assert hash_matches(root/name, manifest['files_sha256'][name])
    summary={(r['bank'],r['method']):r for r in read(root/'summary.csv')}
    shift_rows=read(root/'label_shift_by_battery.csv')
    shifts=read(root/'label_shift.csv');cells=p['batteries']
    macro_flip=float(np.mean([float(r['label_flip_rate']) for r in shift_rows]))
    transfer_penalty=float(np.mean([float(r['transferred_oracle_regret']) for r in shift_rows]))
    flips=sum(int(r['best_length_changed']) for r in shifts)
    mae=lambda bank,m:float(summary[bank,m]['battery_macro_mae'])*100
    regret=lambda bank,m:float(summary[bank,m]['battery_macro_regret'])*100
    figures=root/'figures';figures.mkdir(exist_ok=True);files=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#263442',
        'axes.labelcolor':'#263442','xtick.color':'#536371','ytick.color':'#536371',
        'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,
        'grid.color':'#E5E9EC','figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none'})
    def save(fig,name):
        for ext in ('png','svg'):
            path=figures/f'{name}.{ext}';fig.savefig(path,dpi=200);files.append(path)
        plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5.9));fig.subplots_adjust(left=.10,right=.96,top=.84,bottom=.23)
    values=[float(r['label_flip_rate'])*100 for r in shift_rows]
    positions=np.arange(len(cells));ax.bar(positions,values,.57,color='#8874A1')
    for j,value in enumerate(values):ax.text(j,value+1.7,f'{value:.1f}%',ha='center',weight='bold')
    ax.axhline(macro_flip*100,color='#3C5F78',ls='--',label=f'Equal-cell mean: {macro_flip*100:.1f}%')
    ax.set_xticks(positions,[f'{r["outer_test_battery"]}\n{r["unique_targets"]} targets x 3 rotations' for r in shift_rows],fontsize=10)
    ax.set_ylim(0,75);ax.set_ylabel('Post-hoc best length changed (%)');ax.grid(axis='y')
    ax.legend(loc='upper left',frameon=False)
    fig.suptitle('Does the best lookback change after predictor refitting?',x=.05,ha='left',fontsize=15,weight='bold')
    fig.text(.10,.055,'Same unseen targets; 2-cell candidates vs 3-cell candidates. True SOH is used only for this diagnosis.\n62 distinct targets, 186 target-rotation pairs; these are not 186 independent observations.',fontsize=9,color='#586775')
    save(fig,'best_length_shift')

    fig,ax=plt.subplots(figsize=(11.2,7.1));fig.subplots_adjust(left=.28,right=.96,top=.82,bottom=.17)
    y=np.arange(len(METHODS));height=.32
    for j,bank in enumerate(BANKS):
        values=[mae(bank,m) for m in METHODS];yy=y+(j-.5)*height
        ax.barh(yy,values,height,color=COLOR[bank],label=BANK_LABEL[bank])
        for pos,value in zip(yy,values):ax.text(value+.025,pos,f'{value:.3f}',va='center',fontsize=9)
    ax.set_yticks(y,[LABEL[m] for m in METHODS],fontsize=10);ax.invert_yaxis()
    ax.set_xlim(0,max(mae(b,m) for b in BANKS for m in METHODS)*1.16)
    ax.set_xlabel('Battery-macro SOH MAE (percentage points; lower is better)')
    ax.grid(axis='x');ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
    fig.suptitle('Same selectors, different predictor banks',x=.05,ha='left',fontsize=16,weight='bold')
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.60,.94),ncol=2,frameon=False,fontsize=10)
    fig.text(.05,.045,'Selector training: one separate calibration cell per rotation. Selector choices are identical across banks.\nAverage 3 rotations within each test cell, then average 4 cells. This is not an ensemble or a comparison to pooled-selector training.',fontsize=9,color='#586775')
    save(fig,'paired_predictor_bank_mae')

    fig,ax=plt.subplots(figsize=(10.5,5.9));fig.subplots_adjust(left=.10,right=.97,top=.84,bottom=.27)
    selected=('FixedContext10','CalibrationBest','Adaptive','ErrorAware');x=np.arange(len(selected));width=.32
    for j,bank in enumerate(BANKS):
        values=[regret(bank,m) for m in selected];xx=x+(j-.5)*width
        ax.bar(xx,values,width,color=COLOR[bank],label=BANK_LABEL[bank])
        for pos,value in zip(xx,values):ax.text(pos,value+.012,f'{value:.3f}',ha='center',fontsize=9)
    ax.set_xticks(x,['Fixed LB10','Calibration\nfixed','Classification\nselector','Error-predicting\nselector'])
    ax.set_ylim(0,max(regret(b,m) for b in BANKS for m in selected)*1.30)
    ax.set_ylabel('MAE above the bank-specific Oracle (percentage points)');ax.grid(axis='y')
    ax.legend(frameon=False,loc='upper left',fontsize=10)
    fig.suptitle('Does keeping the label-generating bank solve selection?',x=.05,ha='left',fontsize=15,weight='bold')
    fig.text(.10,.06,'Regret = selected absolute error minus the minimum error among that bank\'s three candidates.\nThe same selector is used before and after refitting. Lower is better; all test cells and rotations are included.',fontsize=9,color='#586775')
    save(fig,'paired_selection_regret')

    names={'FixedContext10':'고정 LB10','FixedContext20':'고정 LB20','FixedContext50':'고정 LB50',
        'CalibrationBest':'별도 1셀 오차로 고른 고정 길이','LabelMajority':'별도 1셀 다수 라벨 고정',
        'Adaptive':'분류형 Selector','ErrorAware':'오차 예측형 Selector','Oracle':'Oracle (사후 비교)'}
    lines=['# 세 번째 실험: Predictor 재학습과 Selector 성능','',
        '이번 실험은 **학습용 정답을 만든 Predictor를 재학습하면서 최적 길이가 달라지는 것이 선택 실패를 설명하는지** 확인하는 통제 실험입니다.',
        '최적 길이 변화 자체는 확인했지만, Predictor를 유지하는 것만으로 Selector 성능이 개선되지는 않았습니다.','',
        '## 비교한 조건','',
        '- 각 분할에서 2개 셀로 Predictor를 학습하고, 별도 1개 셀의 예측 오차로 Selector를 학습했습니다. 남은 1개 셀에서 평가했습니다.',
        '- 같은 Selector를 고정한 채 ① 원래 2셀 Predictor 유지, ② 별도 학습 셀까지 포함한 3셀 Predictor 사용을 비교했습니다.',
        '- 두 조건에서 Selector의 가중치·정규화·선택 길이와 테스트 샘플은 동일합니다. Predictor 후보만 교체했습니다.',
        '- 모든 분할의 Selector를 평가 전에 학습했습니다. 기존 Predictor 48개를 재사용하고, Selector 24개를 새로 학습했습니다.','',
        '## 1. 최적 길이는 실제로 바뀌었습니다.','',
        f'동일 테스트 샘플에서 Predictor를 2셀 학습 후보에서 3셀 학습 후보로 바꾸자, 사후 최적 길이의 변경률은 셀별 평균 **{macro_flip*100:.1f}%**였습니다.',
        f'전체 조합 기준으로는 {len(shifts)}개 중 {flips}개({flips/len(shifts)*100:.1f}%)입니다. 이는 62개 목표를 세 번씩 평가한 결과이며 독립 표본 186개를 의미하지 않습니다.','',
        '| 테스트 셀 | 최적 길이 변경률 | 고유 목표 수 | 평가 조합 수 |',
        '|---|---:|---:|---:|']
    for row in shift_rows:
        lines.append(f'| {row["outer_test_battery"]} | {float(row["label_flip_rate"])*100:.1f}% | {row["unique_targets"]} | {row["target_rotations"]} |')
    lines += ['',f'2셀 후보의 사후 최적 길이를 3셀 후보에 그대로 적용하면, 3셀 Oracle보다 평균 **{transfer_penalty*100:.3f}%p**의 추가 오차가 남았습니다. 이는 테스트 정답을 사용한 진단값이며 배포 가능한 모델 성능이 아닙니다.','',
        '## 2. Predictor를 유지해도 Selector가 고정 LB10보다 좋지는 않았습니다.','',
        '| 방법 | 2셀 Predictor 유지 MAE (%p) | 3셀 Predictor 사용 MAE (%p) |',
        '|---|---:|---:|']
    for method in METHODS:
        lines.append(f'| {names[method]} | {mae("FrozenPair",method):.3f} | {mae("RefitThree",method):.3f} |')
    lines += ['', '각 테스트 셀에서 세 교대 결과를 평균한 뒤, 네 셀을 동일 가중 평균했습니다. 여러 모델의 예측을 평균한 앙상블 결과가 아닙니다.','',
        '## 3. 재학습 후 오차와 선택의 추가 오차는 오히려 줄었습니다.','',
        '| Selector | 전체 MAE: 유지 → 재학습 (%p) | 후보 내 Oracle 대비 추가 오차: 유지 → 재학습 (%p) |',
        '|---|---|---|']
    for method in ('Adaptive','ErrorAware'):
        lines.append(f'| {names[method]} | {mae("FrozenPair",method):.3f} → {mae("RefitThree",method):.3f} | {regret("FrozenPair",method):.3f} → {regret("RefitThree",method):.3f} |')
    lines += ['', '추가 오차는 선택한 예측의 절대오차에서 해당 후보 묶음의 최소 절대오차를 뺀 값입니다.',
        '길이의 정답이 바뀌는 현상과 실제 SOH 오차의 증가는 같은 의미가 아닙니다. 이번에는 Predictor 재학습으로 후보 예측이 바뀌면서 동일 선택의 오차가 감소했습니다.','',
        '## 얻은 인사이트','',
        '- **학습용 라벨과 실제 사용 후보 사이의 차이는 존재합니다.** 따라서 Predictor 재학습 후에도 길이 정답이 그대로 유지된다고 가정하면 안 됩니다.',
        '- **그 차이만으로 현재 Selector의 성능 부족을 설명할 수는 없습니다.** 라벨을 만든 후보를 그대로 유지해도 고정 LB10을 넘지 못했고, 재학습 후에는 전체 MAE와 후보 내 추가 오차가 모두 감소했습니다.',
        '- 이번 통제 조건에서는 후보를 고정해도 다른 배터리에 맞는 길이를 고르는 데 어려움이 남았습니다. Context 정보, 모델 가정, 적은 학습 셀 중 무엇이 주된 원인인지는 이 실험만으로 확정할 수 없습니다.','',
        '## 해석 시 주의할 점','',
        '- 이번 Selector는 한 셀에서 학습했습니다. 두 번째 실험의 Selector는 세 셀의 OOF 자료를 합쳐 학습했으므로, 실험 간 절대 오차를 그대로 비교해 재학습 효과를 주장하지 않습니다.',
        '- 12개 분할은 네 테스트 셀과 세 Selector 학습 셀 교대를 뜻합니다. 독립 테스트 셀이 12개로 늘어난 것이 아닙니다.',
        '- 모든 결과는 이전에 확인한 네 셀에서 얻은 탐색 결과입니다. 원인의 유일성이나 통계적 유의성을 입증하지 않았습니다.',
        '- 추론 시간을 다시 측정하지 않았으며, 비용 개선을 주장하지 않습니다.','',
        '## 다음 판단','',
        '이번 결과만으로 Predictor 재학습을 없애는 방향을 채택할 근거는 얻지 못했습니다. 기존 결과를 보존하고, 추가 셀을 확보할 경우 개발·최종 평가 셀을 먼저 분리한 뒤 같은 비교를 재검증하는 것이 다음 단계입니다.','',
        '## 산출물','',
        '- `summary.csv`: 두 후보 조건의 전체 MAE와 후보 내 추가 오차',
        '- `label_shift.csv`, `label_shift_by_battery.csv`: 최적 길이 변경 진단',
        '- `bank_effects.csv`, `paired_comparisons.csv`: 동일 Selector의 조건별 차이',
        '- `verification.json`: 데이터 역할·학습·예측·집계 검증 PASS',
        '- `figures/best_length_shift.png`: 최적 길이 변경률',
        '- `figures/paired_predictor_bank_mae.png`: 후보 유지·재학습의 정확도 비교',
        '- `figures/paired_selection_regret.png`: 후보 내 최선 대비 추가 오차','']
    (root/'FINDINGS.md').write_text('\n'.join(lines),encoding='utf-8');files.append(root/'FINDINGS.md')
    (figures/'manifest.json').write_text(json.dumps(dict(protocol_sha256=sig,
        artifact_manifest_sha256=digest(root/'artifact_manifest.json'),verification_sha256=digest(root/'verification.json'),
        script_sha256=digest(Path(__file__)),source_sha256={name:digest(root/name) for name in sources},
        files_sha256={str(f.relative_to(root)):digest(f) for f in files}),indent=2)+'\n')
    print(root/'FINDINGS.md')
    for path in files:
        if path.suffix=='.png':print(path)


if __name__=='__main__':main()
