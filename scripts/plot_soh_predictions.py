"""Matplotlib SOH trajectories and predicted-versus-measured scatter plots.

Requires NumPy and Matplotlib. Run from the project root:
    python scripts/plot_soh_predictions.py

Reads existing verified results only. No model fitting, new inference, or seed selection.
Trajectory means summarize repeated runs; they are not ensemble evaluation metrics.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
METHODS = ('Fixed50', 'Adaptive', 'ContextOnlyRidge')
LABELS = {'Fixed50': 'Fixed LB50', 'Adaptive': 'Adaptive selector', 'ContextOnlyRidge': 'Context-only ridge'}
COLORS = {'Fixed50': '#31688E', 'Adaptive': '#8956AE', 'ContextOnlyRidge': '#178879'}
CELL_COLORS = ('#C2694F', '#3C70AC', '#259280', '#9770B1')
CELL_MARKERS = ('o', 's', '^', 'D')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def load(results, data_root=None):
    protocol = json.loads((results/'protocol.json').read_text())
    verified = json.loads((results/'verification.json').read_text())
    manifest = json.loads((results/'artifact_manifest.json').read_text())
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    assert verified['status'] == 'PASS'
    assert verified['protocol_sha256'] == manifest['protocol_sha256'] == signature
    for filename in ('predictions.csv', 'summary.csv'):
        assert sha(results/filename) == manifest['files_sha256'][filename]
    root = data_root or Path(protocol['source_data_root'])
    assert sha(root/'samples.csv') == protocol['data_sha256']['samples.csv']
    samples = {r['sample_id']: r for r in csv_rows(root/'samples.csv')}
    summary = {r['method']: r for r in csv_rows(results/'summary.csv')}
    batteries, seeds = protocol['batteries'], protocol['seeds']
    raw = [r for r in csv_rows(results/'predictions.csv') if r['method'] in METHODS]
    by_key = {(r['outer_test_battery'], r['sample_id'], int(r['seed']), r['method']): r for r in raw}
    assert len(by_key) == len(raw)
    assert set(by_key) == {(r['battery'], sid, s, m) for sid, r in samples.items()
                           for s in seeds for m in METHODS}
    data = {}
    for cell in batteries:
        rows = sorted([r for r in samples.values() if r['battery'] == cell],
                      key=lambda r: float(r['target_start_time_s']))
        ids = [r['sample_id'] for r in rows]
        cycles = np.array([int(r['history_end_exclusive']) for r in rows])
        truth = np.array([float(r['soh'])*100 for r in rows])
        assert np.all(np.diff(cycles) > 0)
        prediction = {}
        for method in METHODS:
            values = np.array([[float(by_key[cell, sid, s, method]['soh_pred'])*100
                                for sid in ids] for s in seeds])
            assert np.isfinite(values).all()
            for sid, target in zip(ids, truth):
                for s in seeds:
                    np.testing.assert_allclose(float(by_key[cell, sid, s, method]['soh_true'])*100,
                                               target, atol=1e-12)
            prediction[method] = values
        np.testing.assert_allclose(prediction['ContextOnlyRidge'],
                                   np.repeat(prediction['ContextOnlyRidge'][:1], len(seeds), axis=0))
        data[cell] = dict(ids=ids, cycles=cycles, truth=truth, prediction=prediction)
    for method in METHODS:
        macro_mae = np.mean([np.abs(data[b]['prediction'][method]-data[b]['truth']).mean()
                             for b in batteries])
        np.testing.assert_allclose(macro_mae, 100*float(summary[method]['battery_macro_mae']), rtol=1e-12)
    return dict(cells=data, batteries=batteries, seeds=seeds, summary=summary,
                protocol_sha256=signature, input_sha256={str(p): sha(p) for p in
                (results/'predictions.csv', results/'summary.csv', root/'samples.csv')})


def configure():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.labelsize': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#AEB9C3', 'axes.axisbelow': True,
        'grid.color': '#E4E8EC', 'grid.linewidth': .7,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})


def save(fig, output, name):
    files = []
    for extension in ('png', 'pdf', 'svg'):
        path = output/f'{name}.{extension}'
        fig.savefig(path, dpi=240)
        files.append(path)
    plt.close(fig)
    return files


def trajectory(data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), sharey=True)
    fig.subplots_adjust(left=.075, right=.98, top=.81, bottom=.18, hspace=.38, wspace=.16)
    fig.suptitle('Measured and predicted SOH over battery usage', x=.075, y=.968,
                 ha='left', fontsize=19, fontweight='bold')
    fig.text(.075, .918, 'One held-out battery per panel; models trained on the other three cells',
             fontsize=11, color='#5B6874')
    handles = [Line2D([], [], color='#202020', lw=2, marker='o', markersize=5,
                      markerfacecolor='white', label='Measured SOH')]
    for method in METHODS:
        handles.append(Line2D([], [], color=COLORS[method], lw=1.8,
                              ls='--' if method=='Fixed50' else '-', label=LABELS[method]))
    fig.legend(handles=handles, ncol=4, loc='upper left', bbox_to_anchor=(.067, .90),
               fontsize=10.5, handlelength=2.5, columnspacing=2)
    lower, upper = 100., 0.
    for ax, cell in zip(axes.flat, data['batteries']):
        d = data['cells'][cell]
        x, truth = d['cycles'], d['truth']
        for method in METHODS:
            values = d['prediction'][method]
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1) if len(values)>1 else np.zeros_like(mean)
            lower, upper = min(lower, (mean-sd).min(), truth.min()), max(upper, (mean+sd).max(), truth.max())
            if method != 'ContextOnlyRidge':
                ax.fill_between(x, mean-sd, mean+sd, color=COLORS[method], alpha=.13, linewidth=0)
            ax.plot(x, mean, color=COLORS[method], lw=1.8,
                    ls='--' if method=='Fixed50' else '-', zorder=3)
        ax.plot(x, truth, color='#202020', marker='o', mfc='white', mew=1.3,
                markersize=4.5, lw=1.6, zorder=5)
        ax.set_title(f'{cell}  |  n = {len(x)}', loc='left', pad=9)
        ax.set_xlabel('Completed RW cycles at input cutoff')
        ax.grid(alpha=.9)
        ax.margins(x=.025)
    axes[0, 0].set_ylabel('SOH (%)')
    axes[1, 0].set_ylabel('SOH (%)')
    axes[0, 0].set_ylim(np.floor((lower-2)/5)*5, min(105, np.ceil((upper+2)/5)*5))
    for y, note in zip((.106, .078, .050), (
        f"Colored curves: mean predictions over {len(data['seeds'])} seeds. Shading: +/-1 seed SD (not a confidence interval).",
        'Means summarize repeated runs; the reported MAE is the mean of individual-run errors, not an ensemble score.',
        'Measured SOH is paired with the preceding RW history. Lines connect measured targets for readability; they are not dense SOH labels.',
    )):
        fig.text(.075, y, note, fontsize=9, color='#5B6874')
    return fig


def parity(data):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=.055, right=.98, bottom=.24, top=.76, wspace=.14)
    fig.suptitle('Predicted versus measured SOH', x=.055, y=.962,
                 ha='left', fontsize=19, fontweight='bold')
    fig.text(.055, .907, 'Points closer to the dashed diagonal indicate more accurate predictions',
             fontsize=11, color='#5B6874')
    handles = [Line2D([], [], color=CELL_COLORS[i], marker=CELL_MARKERS[i], linestyle='none',
                      markersize=6, label=b) for i, b in enumerate(data['batteries'])]
    fig.legend(handles=handles, ncol=len(handles), loc='upper left', bbox_to_anchor=(.052, .877),
               fontsize=10.5, columnspacing=2)
    all_values = np.concatenate([np.concatenate([d['truth']]+[d['prediction'][m].ravel() for m in METHODS])
                                 for d in data['cells'].values()])
    lo, hi = np.floor((all_values.min()-2)/5)*5, np.ceil((all_values.max()+2)/5)*5
    for ax, method in zip(axes, METHODS):
        for i, cell in enumerate(data['batteries']):
            d = data['cells'][cell]
            values = d['prediction'][method]
            if method == 'ContextOnlyRidge':
                values = values[:1]  # Deterministic baseline; do not overplot identical repeats.
            target = np.tile(d['truth'], len(values))
            ax.scatter(target, values.ravel(), s=24, color=CELL_COLORS[i], marker=CELL_MARKERS[i],
                       alpha=.60 if method=='ContextOnlyRidge' else .36, edgecolors='none', zorder=3)
        ax.plot([lo, hi], [lo, hi], '--', color='#555555', lw=1, zorder=2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(MultipleLocator(10)); ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.set_title(LABELS[method], loc='left', pad=12)
        error = 100*float(data['summary'][method]['battery_macro_mae'])
        ax.text(.05, .93, f'Battery-macro MAE = {error:.2f} pp', transform=ax.transAxes, fontsize=9,
                 va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=.88, pad=3))
        ax.set_xlabel('Measured SOH (%)')
        ax.grid(alpha=.9)
    axes[0].set_ylabel('Predicted SOH (%)')
    total = sum(len(d['truth']) for d in data['cells'].values())
    notes = [
        f"Fixed / Adaptive: all {len(data['seeds'])} seeds shown ({total*len(data['seeds'])} points per panel). Context-only: {total} deterministic predictions, plotted once.",
        f"The {total} targets come from {len(data['batteries'])} held-out batteries; repeated seeds are not independent measurements.",
        'MAE is averaged equally across batteries and then across seeds. pp = SOH percentage points, not MAPE.',
    ]
    for y, note in zip((.14, .10, .06), notes):
        fig.text(.055, y, note, fontsize=9, color='#5B6874')
    return fig


def generate(results, data_root, output):
    data = load(results, data_root)
    configure()
    output.mkdir(parents=True, exist_ok=True)
    files = save(trajectory(data), output, 'figure_05_soh_curves')
    files += save(parity(data), output, 'figure_06_prediction_scatter')
    manifest = dict(protocol_sha256=data['protocol_sha256'], input_sha256=data['input_sha256'],
                    plotting_script_sha256=sha(Path(__file__)),
                    matplotlib=matplotlib.__version__, numpy=np.__version__,
                    files_sha256={p.name: sha(p) for p in files})
    (output/'figure_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'CAPTIONS.txt').write_text('''추가 Matplotlib 결과 그림

figure_05_soh_curves
배터리별 실제 SOH(검정 점)와 예측값 비교.
가로축: 예측 입력까지 완료된 RW cycle 수. 세로축: SOH(%).
색 선은 5개 seed 예측의 평균이며 음영은 seed 표준편차. 앙상블 성능을 평가한 그림이 아님.
실제 SOH 측정점 사이의 선은 가독성을 위한 연결이며 연속 정답이 존재한다는 뜻이 아님.
Context-only 회귀는 결정론적이므로 음영 없음. 길이 선택이 LB50이면 Fixed와 Adaptive가 겹칠 수 있음.

figure_06_prediction_scatter
가로축: 실제 SOH, 세로축: 예측 SOH. 대각선에 가까울수록 예측 오차가 작음.
Fixed50과 Adaptive는 5개 seed의 예측을 모두 표시. Context-only는 중복 예측을 한 번만 표시.
색/모양은 배터리를 구분. 표시한 MAE는 각 배터리의 MAE를 동일 가중 평균한 후 seed를 평균한 값.

PNG는 발표용, PDF/SVG는 벡터 파일. 기존 모델과 평가 수치는 변경하지 않음.
재생성: python scripts/plot_soh_predictions.py (NumPy와 Matplotlib 필요).
''', encoding='utf-8')
    print(json.dumps(dict(output=str(output), figures=2, formats=['png', 'pdf', 'svg']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=PROJECT/'outputs/adaptive_selector_v1')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    generate(args.results.resolve(), args.data_root.resolve() if args.data_root else None,
             (args.output or args.results/'figures/predictions').resolve())
