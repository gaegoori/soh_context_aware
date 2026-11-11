# 확장 데이터의 Selector 학습·개발 검증

기존 Context Ridge Predictor를 고정하고 길이 선택 규칙을 비교합니다.
개발용 8개 배터리의 평가이며 최종 평가용 RW15·16·19·20은 사용하지 않습니다.
이전 개발 결과를 본 후 설계한 탐색 실험으로, 최종 일반화 검증과 구분합니다.

## 비교 방법

| 방법 | 선택 규칙 |
|---|---|
| FixedContext10/20/50 | 해당 길이의 기존 Predictor를 항상 사용 |
| GlobalBestOOF | 학습 측 셀 평균 OOF 절대오차가 가장 작은 길이를 고정 |
| Classifier | 15개 Context로 OOF 오차 최소 길이를 분류 |
| ErrorAware | 15개 Context로 세 길이의 절대오차를 예측하여 최소값 선택 |
| GatedError | 같은 오차 예측으로 기본 길이 대비 예상 개선이 threshold보다 클 때만 전환 |
| Oracle | 검증 정답을 알고 고르는 사후 참고값 |

Selector 입력은 기존 C10 15개입니다. 최근 운용 변화 변수를 새로 추가하지 않습니다.
Predictor의 길이에 따라 최근 통계 9개가 바뀌며, 누적·마지막 cycle 변수 6개는 공통입니다.
원신호 신경망을 새로 학습하는 실험이 아닙니다.

분류기는 선형 multinomial logistic, 오차 예측기는 3개 Ridge입니다. 기존 정규화 강도 0.1을 유지합니다.
평균 대체·표준화는 각 모델의 학습 샘플에서만 계산하며 전부 결측인 열은 비활성화합니다.
학습은 샘플별 동일 가중치, 주 평가는 셀별 MAE 동일 가중 평균입니다. 결정론적 학습이므로 seed 반복은 없습니다.

## 배터리 분할 및 threshold 선택

외부 개발 검증: 1개 셀을 제외하고 나머지 7개 셀의 OOF 라벨로 Selector를 학습합니다.
검증에는 이미 7개 셀로 학습한 고정 Predictor 후보를 사용합니다.

threshold 선택에는 별도의 내부 배터리 검증이 필요합니다.
기존 OOF 라벨을 그대로 나누면, 보정할 셀이 다른 라벨의 Predictor 학습에 포함될 수 있어 아래와 같이 다시 생성합니다.

1. 7개 학습 셀 중 1개를 threshold 보정용으로 제외합니다.
2. 남은 6개 안에서 Predictor 학습 5개 / Selector 라벨 생성 1개를 교대합니다.
3. 이 라벨과 Context로 오차 예측형 Selector를 학습합니다.
4. 보정 셀에서 길이를 선택하고, 6개 셀로 학습한 후보의 실제 SOH 오차로 threshold를 비교합니다.
5. 7개 보정 셀의 MAE를 동일 가중 평균하여 threshold를 정합니다.
6. 외부 검증 전에 최종 Selector와 threshold를 모두 확정합니다.

기본 길이 역시 각 Selector 학습 셀의 OOF 오차만으로 정하며 LB50을 강제하지 않습니다.
threshold 후보는 SOH 절대오차 기준 `0, 0.05, 0.1, 0.2, 0.5, 1, 2 %p`와 `전환하지 않음`입니다.
코드의 SOH는 0~1 단위여서 각각 `0, .0005, .001, .002, .005, .01, .02, None`으로 저장합니다.
예상 개선이 threshold와 같으면 기본 길이를 유지합니다.
내부 MAE가 최소값과 1e-12 이내로 같으면 전환 횟수가 적은 기준, 그다음 더 큰 threshold를 선택합니다.

고유한 5셀 Predictor 168개, 내부 오차 Selector 56개, 외부 오차 Selector 8개와 분류 Selector 8개를 학습합니다.
모든 모델·threshold를 저장한 뒤 외부 개발 검증 예측을 계산합니다.
5→6, 6→7개 셀로 Predictor 학습 수가 바뀌는 라벨 전이 차이는 남아 있습니다.

## 실행

기존 `expanded_predictor_baseline_v1`의 검증 결과가 PASS이고 파일 해시가 일치해야 합니다.
완료된 출력 폴더를 덮어쓰지 않습니다. 재실행 시 새로운 `--output`을 지정합니다.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-repro/bin/python -m scripts.tests.test_expanded_selector

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 .venv-repro/bin/python -m scripts.experiments.train_expanded_selector

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-repro/bin/python -m scripts.validation.verify_expanded_selector
```

`outputs/expanded_selector_v1/`에 protocol, 모델, 내부 라벨, 보정 예측·threshold별 오차,
최종 선택과 예측, 전체·셀별·운용 조건별 표, `FINDINGS.md`, 독립 검증을 저장합니다.
검증은 모든 신규 Ridge를 augmented least squares로 재계산하고 분류기의 최적 조건,
배터리 분리, threshold 선택, 기존 Predictor 예측 유지와 최종 집계를 확인합니다.

PASS는 구현 검증이며 성능 향상이나 유의성을 의미하지 않습니다.
평균 Lookback은 계산 시간과 다르며, 이번 실험에서는 실제 지연 시간을 측정하지 않습니다.
학습된 오차 예측이 음수일 수 있어 순위 점수로 사용하고 임의 clipping은 하지 않습니다.
결과·모델·라벨·데이터는 Git에서 제외하며 코드와 실행 문서만 공개 대상입니다.
