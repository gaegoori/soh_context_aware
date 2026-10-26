"""Plot verified outer-test results without retraining or modifying saved results.

Run from the repository root with NumPy and Matplotlib installed:
    python scripts/plot_adaptive_results.py

Outputs: outputs/adaptive_selector_v1/figures/{*.png,*.pdf,*.svg}.
MAE is converted from fractional SOH to percentage points (not MAPE).
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
METHODS = ('Fixed10', 'Fixed20', 'Fixed50', 'GlobalBestOOF', 'LabelMajority',
           'Adaptive', 'ContextOnlyRidge', 'Oracle')
MAIN = ('Fixed50', 'GlobalBestOOF', 'Adaptive', 'ContextOnlyRidge')
LABEL = dict(Fixed10='Fixed LB10', Fixed20='Fixed LB20', Fixed50='Fixed LB50',
             GlobalBestOOF='OOF-selected fixed', LabelMajority='Majority-label fixed',
             Adaptive='Adaptive selector', ContextOnlyRidge='Context-only ridge',
             Oracle='Oracle (diagnostic only)')
COLOR = dict(Fixed10='#8CABC4', Fixed20='#5C84A6', Fixed50='#2E526F',
             GlobalBestOOF='#76848E', LabelMajority='#AFBBC4', Adaptive='#8A5CB5',
             ContextOnlyRidge='#198A7D', Oracle='#AFB4BA')
INK = '#223242'
GRAY = '#637283'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def load_data(root):
    check = json.loads((root/'verification.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    artifacts = json.loads((root/'artifact_manifest.json').read_text())
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    assert check['status'] == 'PASS'
    assert check['protocol_sha256'] == artifacts['protocol_sha256'] == signature
    sources = ('summary.csv', 'metrics_by_battery.csv', 'metrics_per_seed.csv', 'selections.csv')
    for name in sources:
        assert digest(root/name) == artifacts['files_sha256'][name], f'Changed result: {name}'
    summary = {r['method']: r for r in read_csv(root/'summary.csv')}
    battery_rows = read_csv(root/'metrics_by_battery.csv')
    by_battery = {(r['outer_test_battery'], r['method']): r for r in battery_rows}
    seed_rows = read_csv(root/'metrics_per_seed.csv')
    by_seed = {(r['outer_test_battery'], int(r['seed']), r['method']): r for r in seed_rows}
    selections = read_csv(root/'selections.csv')
    batteries, seeds = protocol['batteries'], protocol['seeds']
    assert set(summary) == set(METHODS)
    assert len({r['sample_id'] for r in selections}) == len(selections) == check['unique_test_samples']
    for method in METHODS:
        macro = [np.mean([float(by_seed[b, s, method]['mae']) for b in batteries]) for s in seeds]
        np.testing.assert_allclose(np.mean(macro), float(summary[method]['battery_macro_mae']), rtol=1e-12)
        np.testing.assert_allclose(np.std(macro, ddof=1) if len(seeds)>1 else 0.,
                                   float(summary[method]['macro_mae_seed_sd']), rtol=1e-12, atol=1e-15)
    counts = {b: Counter(int(r['selected_lookback']) for r in selections
                         if r['outer_test_battery'] == b) for b in batteries}
    for b in batteries:
        assert sum(counts[b].values()) == int(by_battery[b, 'Adaptive']['samples_per_seed'])
    return dict(summary=summary, battery=by_battery, batteries=batteries, seeds=seeds,
                counts=counts, samples=len(selections), signature=signature,
                hashes={name: digest(root/name) for name in sources})


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 14,
        'axes.titleweight': 'bold', 'axes.labelsize': 11, 'axes.labelcolor': INK,
        'text.color': INK, 'xtick.color': GRAY, 'ytick.color': GRAY,
        'axes.edgecolor': '#B8C3CC', 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.axisbelow': True, 'grid.color': '#E6EBEF', 'grid.linewidth': .7,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white', 'legend.frameon': False,
    })


def overview_mae(ax, data, title):
    summary = data['summary']
    value = np.array([float(summary[m]['battery_macro_mae'])*100 for m in METHODS])
    sd = np.array([float(summary[m]['macro_mae_seed_sd'])*100 for m in METHODS])
    y = np.arange(len(METHODS))
    bars = ax.barh(y, value, height=.58, color=[COLOR[m] for m in METHODS],
                  edgecolor='white', linewidth=.5, zorder=2)
    bars[-1].set_hatch('///')
    bars[-1].set_edgecolor('#FFFFFF')
    for i, (v, e) in enumerate(zip(value, sd)):
        if e > 0:
            ax.errorbar(v, i, xerr=e, fmt='none', ecolor=INK, elinewidth=1., capsize=3, zorder=3)
        ax.text(v+e+.15, i, f'{v:.2f}', va='center', fontsize=10.5, fontweight='bold')
    ax.axhline(6.5, color='#B8C3CC', linewidth=.7, linestyle=(0, (3, 3)))
    ax.set_yticks(y, [LABEL[m] for m in METHODS], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(value+sd)+1.4)
    ax.set_xlabel('Battery-macro SOH MAE (percentage points)')
    ax.set_title(title, loc='left', pad=14)
    ax.grid(axis='x')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)


def battery_mae(ax, data, title):
    batteries, rows = data['batteries'], data['battery']
    x, width = np.arange(len(batteries)), .18
    for i, method in enumerate(MAIN):
        values = [float(rows[b, method]['mae_mean'])*100 for b in batteries]
        sd = [float(rows[b, method]['mae_seed_sd'])*100 for b in batteries]
        ax.bar(x+(i-1.5)*width, values, width, yerr=sd, color=COLOR[method],
               label=LABEL[method], error_kw=dict(ecolor=INK, capsize=2, elinewidth=.8), zorder=2)
    ax.set_xticks(x, [f'{b}\nn = {sum(data["counts"][b].values())}' for b in batteries])
    ymax = max(float(rows[b, m]['mae_mean'])+float(rows[b, m]['mae_seed_sd'])
               for b in batteries for m in MAIN)*100
    ax.set_ylim(0, ymax*1.2)
    ax.set_ylabel('SOH MAE (percentage points)')
    ax.set_title(title, loc='left', pad=14)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis='y')
    ax.legend(fontsize=9, loc='upper right', handlelength=1.2, labelspacing=.65)


def cost_accuracy(ax, data, title):
    summary = data['summary']
    annotations = {
        'Fixed10': ('LB10', (-19, 14)), 'Fixed20': ('LB20', (-45, -4)),
        'Fixed50': ('LB50', (20, -22)), 'GlobalBestOOF': ('OOF fixed', (26, -1)),
        'LabelMajority': ('Majority fixed', (18, 29)),
        'Adaptive': ('Adaptive', (-17, 15)), 'ContextOnlyRidge': ('Context only', (16, 2)),
    }
    for method in METHODS[:-1]:
        x = float(summary[method]['latency_median_ms'])
        y = float(summary[method]['battery_macro_mae'])*100
        ax.scatter(x, y, s=105 if method in ('Adaptive', 'ContextOnlyRidge') else 68,
                   color=COLOR[method], edgecolor='white', linewidth=.7, zorder=4)
        label, offset = annotations[method]
        ax.annotate(label, (x, y), xytext=offset, textcoords='offset points',
                    fontsize=10, color=INK,
                    arrowprops=dict(arrowstyle='-', color='#9AA7B1', lw=.7), zorder=5)
    ax.text(.04, .045, 'Lower and further left is better', transform=ax.transAxes,
            fontsize=9.5, color=GRAY)
    ax.set_xlim(0, max(float(summary[m]['latency_median_ms']) for m in METHODS[:-1])*1.24)
    ax.set_ylim(0, max(float(summary[m]['battery_macro_mae'])*100 for m in METHODS[:-1])+2)
    ax.set_xlabel('Median inference time per sample (ms)')
    ax.set_ylabel('Battery-macro SOH MAE (percentage points)')
    ax.set_title(title, loc='left', pad=14)
    ax.grid(alpha=.85)


def lookback_distribution(ax, data, title):
    batteries, counts = data['batteries'], data['counts']
    x = np.arange(len(batteries))
    bottom = np.zeros(len(batteries))
    for lb in (10, 20, 50):
        n = np.array([counts[b][lb] for b in batteries])
        heights = np.array([100*counts[b][lb]/sum(counts[b].values()) for b in batteries])
        ax.bar(x, heights, .62, bottom=bottom, color=COLOR[f'Fixed{lb}'],
               edgecolor='white', linewidth=1, label=f'LB{lb}', zorder=2)
        for i, (height, number) in enumerate(zip(heights, n)):
            if number:
                ax.text(i, bottom[i]+height/2, str(number), ha='center', va='center',
                        color=INK if lb==10 else 'white', fontweight='bold', fontsize=12)
        bottom += heights
    ax.set_xticks(x, [f'{b}\nn = {sum(counts[b].values())}' for b in batteries])
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel('Share of test samples (%)')
    ax.set_title(title, loc='left', pad=14)
    ax.grid(axis='y')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.005), ncol=3, fontsize=9,
              facecolor='white', framealpha=1., frameon=True, edgecolor='none')


def save_figure(fig, folder, stem):
    paths = []
    for suffix in ('png', 'pdf', 'svg'):
        path = folder/f'{stem}.{suffix}'
        fig.savefig(path, dpi=240)
        paths.append(path)
    plt.close(fig)
    return paths


def generate(root, output):
    data = load_data(root)
    style()
    output.mkdir(parents=True, exist_ok=True)
    notes = (
        f"{len(data['batteries'])} held-out batteries | {data['samples']} unique SOH measurements | "
        f"{len(data['seeds'])} predictor seeds | Error bars: +/-1 seed SD, not confidence intervals",
        'Latency: CPU, batch=1; includes context calculation and selected-model inference; excludes raw preprocessing and model loading.',
        'OOF fixed: length chosen using training OOF errors. Oracle: post-hoc choice among LB10/20/50, not deployable. Stack labels: sample counts.',
    )
    fig, axes = plt.subplots(2, 2, figsize=(17, 11.8))
    fig.subplots_adjust(left=.155, right=.975, bottom=.145, top=.85, hspace=.52, wspace=.30)
    fig.text(.035, .955, 'Adaptive temporal selection: first outer-battery evaluation',
             fontsize=22, fontweight='bold', va='top')
    fig.text(.035, .915, 'Accuracy, inference cost and selected history length', fontsize=13, color=GRAY)
    overview_mae(axes[0, 0], data, '(a) Accuracy across methods')
    battery_mae(axes[0, 1], data, '(b) Accuracy by held-out battery')
    cost_accuracy(axes[1, 0], data, '(c) Accuracy versus inference cost')
    lookback_distribution(axes[1, 1], data, '(d) Adaptive lookback selection')
    for y, note in zip((.082, .056, .030), notes):
        fig.text(.035, y, note, fontsize=10, color=GRAY)
    files = save_figure(fig, output, 'figure_00_results_overview')
    specs = [
        ('figure_01_method_mae', overview_mae, 'SOH error across prediction methods', (.29, .96), notes[0]+'\n'+notes[2].split(' Stack')[0]),
        ('figure_02_battery_mae', battery_mae, 'SOH error by held-out battery', (.11, .97), notes[0]),
        ('figure_03_accuracy_cost', cost_accuracy, 'SOH accuracy versus inference cost', (.11, .97), notes[1]+'\nOracle is excluded because it requires the test SOH target.'),
        ('figure_04_lookback_selection', lookback_distribution, 'Lookback selected for each test battery', (.11, .97),
         f"{data['samples']} unique test samples; one deterministic selector per outer fold.\nNumbers inside the bars are sample counts; each sample is counted once, not once per predictor seed."),
    ]
    for stem, draw, title, margins, note in specs:
        fig, ax = plt.subplots(figsize=(11.5, 7))
        fig.subplots_adjust(left=margins[0], right=margins[1], bottom=.23, top=.87)
        draw(ax, data, title)
        fig.text(.04, .055, note, fontsize=9, color=GRAY, linespacing=1.8, va='bottom')
        files.extend(save_figure(fig, output, stem))
    captions = '''결과 그림 안내

figure_00_results_overview: 전체 결과를 4개 패널로 정리한 그림.
figure_01_method_mae: 방법별 SOH 오차. 낮을수록 좋음. 단위는 %p이며 MAPE가 아님.
figure_02_battery_mae: 테스트 배터리별 오차. n은 독립 배터리 수가 아니라 해당 셀의 측정 샘플 수.
figure_03_accuracy_cost: 오차와 추론 시간. 왼쪽 아래에 가까울수록 오차가 작고 빠름.
figure_04_lookback_selection: 배터리별 선택 길이 비율. 막대 안 숫자는 실제 샘플 개수.

막대의 오차선은 5개 Predictor seed의 표준편차이며 신뢰구간이 아님.
Context-only는 같은 결정론적 회귀 모델이므로 seed 표준편차가 0임.
OOF-selected fixed는 테스트를 제외한 학습 측 OOF 오차로 선택한 고정 길이.
Oracle은 세 길이의 Predictor 중 테스트 정답에 가장 가까운 예측을 사후 선택한 진단값.
추론 시간에는 Context 재계산·정규화·Selector·선택한 Predictor 실행이 포함됨.
원시 센서 전처리·파일 읽기·모델 로딩은 시간 측정에서 제외됨.

PNG: 발표 자료 삽입용, 240 dpi.
PDF: 논문/인쇄용 벡터 파일.
SVG: 편집 가능한 벡터 파일. 글자는 DejaVu Sans 글꼴이 필요할 수 있음.
원본 실험의 코드·설정·모델·평가 결과는 변경하지 않고 저장된 CSV로 그림만 생성함.
'''
    (output/'CAPTIONS.txt').write_text(captions, encoding='utf-8')
    provenance = dict(protocol_sha256=data['signature'], input_csv_sha256=data['hashes'],
        plotting_source_sha256=digest(Path(__file__)),
        versions=dict(numpy=np.__version__, matplotlib=matplotlib.__version__),
        error_unit='SOH percentage points; source fractional MAE multiplied by 100',
        uncertainty='plus/minus one SD across predictor seeds; not a confidence interval',
        files_sha256={p.name: digest(p) for p in files})
    (output/'figure_manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(figures=5, formats=['png', 'pdf', 'svg'], output=str(output),
                         verified_input_rows=data['samples']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=PROJECT/'outputs/adaptive_selector_v1')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    generate(args.results.resolve(), (args.output or args.results/'figures').resolve())
