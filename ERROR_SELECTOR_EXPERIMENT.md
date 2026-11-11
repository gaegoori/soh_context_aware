# OOF 오차 예측형 Selector 비교 실험

## 목적과 범위

기존 신호 기반 LB10·20·50 Predictor를 고정하고, Selector의 학습 목표를 바꾸었을 때
고정 길이 및 기존 분류형 Selector보다 SOH 예측 오차가 감소하는지 확인합니다.
기존 `outputs/adaptive_selector_v1/`의 Predictor 60개를 읽어서 사용하며 재학습하지 않습니다.
길이별 Context Ridge Predictor 실험은 별도 계획이며 이번 실행에 포함하지 않습니다.

## 고정 설정

- 입력: 기존과 동일한 인과적 Context 15개. 정규화·결측 대체는 각 분할의 학습 셀만으로 계산합니다.
- 정답: 각 학습 샘플에서 LB10·20·50의 **OOF 절대오차를 5개 seed에 걸쳐 평균**한 값 세 개입니다.
  평균 예측값의 오차가 아닙니다. 해당 샘플의 배터리는 라벨 생성 Predictor 학습에서 제외되어 있습니다.
- 모델: 출력별 독립 Ridge 회귀, 각 출력의 `mean((예상 오차 - OOF 오차)^2) + 0.1 * ||w||²`를 최소화합니다.
  절편은 규제하지 않고, 샘플 가중치는 모두 같습니다. float64 선형해법이며 하이퍼파라미터를 탐색하지 않습니다.
- 선택: Context에서 계산한 세 오차 점수 중 최솟값에 해당하는 길이를 선택합니다. 정확한 동점은 짧은 길이를 우선합니다.
- 선형 외삽으로 음수 오차 점수가 나올 수 있습니다. 이번에는 순위를 보존하기 위해 자르지 않으며 음수 발생을 별도로 기록합니다.
  점수가 실제 오차의 잘 보정된 추정치라고 가정하지 않습니다.
- 추론: 선택된 길이의 Predictor 하나만 실행합니다. 실제 SOH 정답과 사후 최적 길이는 추론 입력이 아닙니다.

## 분할 및 비교

기존 RW3~RW6의 4개 외부 분할을 그대로 사용합니다. 각 외부 분할에서 학습 측 3개 셀의
OOF 자료로 새 Selector 하나를 학습합니다. 네 Selector의 학습을 모두 끝낸 뒤 평가합니다.
비교군은 Fixed10/20/50, GlobalBestOOF, LabelMajority, 기존 Adaptive, 새 ErrorAware,
ContextOnlyRidge, 사후 Oracle입니다. 모든 기존 예측값이 원본 실행과 일치하는지 검사합니다.

주 지표는 배터리별 MAE를 동일 가중 평균한 값입니다. 새 방법과 기존 Adaptive·GlobalBestOOF의
배터리별 차이, Oracle까지 남은 차이, 평균 선택 길이를 함께 저장합니다.
기록된 SD는 같은 5개 Predictor seed에 대한 변동이며 배터리 모집단의 신뢰구간이 아닙니다.
Ridge Selector는 결정적이므로 seed별로 같은 Selector를 반복 학습하지 않습니다.

기존 네 셀의 결과를 보고 새 방법을 설계했으므로 **탐색 실험**입니다. 새로운 셀에서의 확인이 필요합니다.
라벨 생성 Predictor는 2개 셀, 평가용 Predictor는 3개 셀로 학습되어 최적 길이가 달라질 수 있다는 한계도 유지됩니다.

## 실행

저장소 루트에서 기존 전처리·OOF·Adaptive 실험 결과가 존재해야 합니다.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-repro/bin/python -m unittest discover -s scripts -p 'test_error_selector.py' -v
PYTHONDONTWRITEBYTECODE=1 .venv-repro/bin/python -m scripts.experiments.train_error_selector
PYTHONDONTWRITEBYTECODE=1 .venv-repro/bin/python -m scripts.validation.verify_error_selector
```

`--phase fit`과 `--phase evaluate`로 학습 및 평가를 나눌 수 있습니다.
출력 기본값은 `outputs/error_selector_v1/`이며 기존 실행 결과를 덮어쓰지 않습니다.
새 학습 실행에는 빈 출력 경로가 필요합니다. 중단된 평가만 재개할 때는 `--phase evaluate`를 사용합니다.

추론 시간은 동일 실행에서 모든 배포 가능한 방법을 다시 측정합니다.
CPU 1 thread, batch 1, 3회 워밍업 후 20회 반복하며 방법·샘플 실행 순서를 섞습니다.
Context 계산·선택·정규화·선택 Predictor 실행을 포함합니다.
원시 데이터 로딩, cycle 분할·리샘플링, 모델 로딩, 누적 메타데이터 초기 구축은 제외합니다.

## 로컬 산출물

- `protocol.json`, `protocol_frozen.json`: 실행 전 고정한 설정과 입력·코드 해시
- `models/`, `fitting_complete.json`: 학습 행 목록, 오차 회귀 계수, 네 모델의 학습 완료 시점
- `predictions.csv`, `selections.csv`: SOH 예측, 선택 길이 및 세 예상 오차 점수
- `summary.csv`, `metrics_by_battery.csv`, `paired_comparisons.csv`: 전체·셀별 결과 및 짝지은 차이
- `selection_diagnostics.csv`: seed 평균 오차 기준 사후 최적 길이와 선택의 추가 오차
- `latency_samples.csv`: 이번 실행에서 측정한 원시 지연 기록
- `verification.json`: OOF 정답 재구성, 독립 선형해법, 모델 재실행 및 집계 검증

GitHub에는 코드와 실행 문서만 포함하며 데이터·모델·결과는 기존 `.gitignore` 규칙을 따릅니다.

## 결과 그림과 요약 문서

검증 후 NumPy와 Matplotlib이 설치된 Python 환경에서 실행합니다.

```bash
python -m scripts.plotting.plot_error_selector
```

`outputs/error_selector_v1/FINDINGS.md`와 `figures/`의 PNG·SVG 두 종류 비교 그림을 생성합니다.
원본 결과 CSV는 변경하지 않으며, 그림은 전체 비교군과 네 셀을 모두 표시합니다.
