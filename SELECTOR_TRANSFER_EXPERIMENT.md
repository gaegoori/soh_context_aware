# 세 번째 실험: Predictor 재학습과 길이 선택의 변화

상태: **2026-10-27 실행 및 독립 검증 완료.** 결과·그림은 `outputs/selector_transfer_v1/`에 보관합니다.

## 질문

Selector 정답을 만든 Predictor와 실제 사용 Predictor가 달라지는 것이 길이 선택에 영향을 주는지 확인합니다.
같은 외부 테스트 샘플에 대해 **2개 셀로 학습한 후보**와 **3개 셀로 재학습한 후보**를 비교합니다.
OOF 라벨 샘플을 포함해 재학습한 모델의 학습 잔차와 OOF 오차를 비교하는 방식은 사용하지 않습니다.

## 통제 설계

외부 셀 RW6, Selector 학습 셀 RW3인 예입니다.

| 역할 | 사용 셀 |
|---|---|
| 원래 Predictor 후보 학습 | RW4, RW5 |
| 해당 Predictor의 미학습 예측으로 Selector 학습 | RW3 |
| 평가용 재학습 Predictor | RW3, RW4, RW5 |
| 두 조건의 동일한 외부 테스트 | RW6 |

1. `temporal_context_ridge_v1`의 검증된 2셀·3셀 Ridge 후보를 그대로 읽습니다.
2. 원래 2셀 Predictor가 RW3에서 낸 오차로 분류형·오차 예측형 Selector를 학습합니다.
3. **동일한 Selector와 동일한 선택 길이**를 두 Predictor 조건에 적용합니다.
4. RW6의 동일한 SOH 목표에서 오차와 후보 내 Oracle 대비 추가 오차를 비교합니다.

각 외부 셀마다 Selector 학습 셀을 세 번 교대하므로 4×3=12개 분할입니다.
분류형 12개와 오차 예측형 12개를 새로 학습하고, Predictor는 추가로 학습하지 않습니다.
모든 Selector를 학습한 뒤 평가합니다. 규제값 0.1, 기존 모델 구조와 특징을 고정합니다.
예측 시 현재 정답 SOH·배터리 ID·Stage를 입력하지 않습니다.

## 두 가지 분석

### 같은 테스트 샘플에서 최적 길이가 얼마나 달라지는가?

각 2셀 후보 묶음과 3셀 후보 묶음에서 실제 오차가 가장 작은 길이를 사후 계산합니다.
길이 변경률, 1·2위 오차 간격, 2셀 후보의 최적 길이를 3셀 후보에 적용했을 때 남는 추가 오차를 기록합니다.
이 분석은 테스트 정답을 쓰는 **진단용**이며, 실제 Selector의 입력이나 학습 정답으로 사용하지 않습니다.

### Predictor를 유지하면 Selector가 더 잘 선택하는가?

동일한 Selector로 두 조건에서 평가합니다. 비교 대상은 각 조건의 FixedContext10/20/50,
CalibrationBest, LabelMajority, 분류형·오차 예측형 Selector, 사후 Oracle입니다.
CalibrationBest는 **이번 한 개 Selector 학습 셀의 오차**로 고른 고정 길이이며, 앞선 세 셀 OOF 고정 선택과 구분합니다.

`regret = 선택한 후보의 절대오차 − 해당 후보 묶음의 최소 절대오차`입니다.
재학습으로 Predictor 자체 정확도도 달라질 수 있으므로 SOH MAE와 regret을 함께 봅니다.
regret 변화는 음수일 수도 있습니다. Predictor 재학습이 항상 선택을 악화시킨다고 가정하지 않습니다.

## 집계와 해석 범위

- 동일 외부 셀에서 세 Selector 학습 셀 교대의 결과를 평균하고, 외부 네 셀을 동일 가중 평균합니다.
- 186개 샘플×분할 조합은 **62개 목표를 세 번 평가한 것**입니다. 186개 독립 표본이나 12개 독립 테스트 셀로 해석하지 않습니다.
- 교대 결과 평균은 앙상블 예측이 아니며, 테스트 성능이 좋은 교대만 선택하지 않습니다.
- 이번 Selector는 1개 셀로 학습합니다. 앞선 두 번째 실험의 Selector는 세 셀 OOF를 합쳐 학습했습니다.
  따라서 두 실험 사이 절대 성능 차이를 Predictor 재학습의 효과라고 단정하지 않습니다.
- 같은 테스트 셀에서 후보 학습 셀 수가 2→3개로 늘어나는 효과를 측정합니다. 추가되는 셀 구성, 회귀 계수와 정규화 변화가 함께 포함됩니다.
- 현재 결과는 이미 본 네 셀에서의 탐색 진단입니다. Context 부족, 학습 셀 부족, 후보 순위 변화 중 하나가 유일한 원인임을 확정하지 않습니다.
- 이번에는 추론 시간을 다시 측정하지 않으며 계산 효율 개선을 주장하지 않습니다.

## 실행

검증된 `outputs/temporal_context_ridge_v1/`가 있어야 합니다.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-repro/bin/python -m scripts.experiments.train_selector_transfer
PYTHONDONTWRITEBYTECODE=1 .venv-repro/bin/python -m scripts.validation.verify_selector_transfer
```

기본 출력은 `outputs/selector_transfer_v1/`입니다. 기존 폴더를 덮어쓰지 않으며,
새 실행에는 빈 `--output` 경로가 필요합니다. 학습과 평가는 `--phase fit`, `--phase evaluate`로 분리할 수 있습니다.

Matplotlib과 NumPy가 설치된 환경에서 그림과 설명을 생성합니다.

```bash
python -m scripts.plotting.plot_selector_transfer
```

주요 파일은 `summary.csv`, `metrics_by_split.csv`, `metrics_by_battery.csv`, `bank_effects.csv`,
`label_shift.csv`, `label_shift_by_battery.csv`, `selector_diagnostics.csv`, `FINDINGS.md`입니다.
`verification.json`은 데이터 역할·정규화·회귀 해법·동일 선택·사후 진단·집계를 독립 재계산한 결과입니다.
데이터·모델·성능 결과는 로컬에 보관하고, GitHub 공개 범위는 코드와 실행 문서입니다.
