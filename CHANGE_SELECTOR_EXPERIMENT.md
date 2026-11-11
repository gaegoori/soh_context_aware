# 운용 변화 Context와 비선형 Selector

기존 Predictor를 유지하면서 Selector 입력과 모델의 두 요인을 분리해 비교합니다.
개발용 RW3·4·5·6·13·14·17·18만 사용하며 최종 평가용 RW15·16·19·20은 사용하지 않습니다.
이전 개발 결과를 본 뒤 설계한 탐색 실험이며 독립 최종 평가와 구분합니다.

## 비교 구성

| 입력 | 선형 오차 예측기 | 비선형 오차 예측기 |
|---|---|---|
| 기존 Context 15개 | Base Linear | Base Forest |
| 기존 Context + 변화 6개 | Change Linear | Change Forest |

네 Selector는 같은 OOF 절대오차를 학습하고, 예상 오차가 최소인 길이를 고릅니다.
동일 Predictor의 LB10·20·50 후보와 동일 측정 SOH 샘플을 사용합니다.
이번에는 threshold를 적용하지 않아 입력·모델 변경 효과를 구분합니다.

같은 네 입력·모델 조합으로 SOH를 직접 예측하는 기준선도 학습합니다.
추가 정보 자체의 효과와 길이 선택의 기여를 구분하기 위한 비교입니다.
기존 고정 LB10·20·50, 학습 측 OOF로 선택한 고정 길이, 사후 Oracle도 보고합니다.

## 추가 Context

완료된 최근 10개 cycle을 이전 5개 `[n-10:n-5]`와 최근 5개 `[n-5:n]`로 나눕니다.
다음 항목마다 `최근 평균 − 이전 평균`을 계산합니다.

- 전압 (V), 전류 (A), 온도 (°C)
- 실제 방전 시간 (s), 방전량 (Ah)
- 온도 결측 비율 (0~1)

신호 평균은 각 구간의 5 × 128점 중 유효한 값으로 계산합니다.
한쪽 구간의 온도가 전부 결측이면 온도 변화도 NaN으로 남깁니다.
학습 셀의 평균으로 결측을 대체하며, 학습에서 전부 결측인 열은 비활성화합니다.
결측률 변화도 별도 특징으로 제공하지만 센서 오류가 해결됐다는 의미는 아닙니다.
추가 특징은 예측 시점 이전 자료만 사용하며, SOH·미래 시점·수명 단계·셀 ID는 입력하지 않습니다.
기존 Context에도 평균·변동성·방전 시간 추세가 있으므로, 이는 변화 정보를 구체적으로 보강하는 비교입니다.

## 고정한 학습 설정

- 선형: 기존 Ridge, mean MSE + 0.1 × squared weight norm, 무벌점 intercept
- 비선형: RandomForestRegressor, 200개 트리, 최대 깊이 3, 최소 leaf 8, bootstrap, 모든 특징 후보, seed 42, CPU 1 thread
- Selector Forest: 세 길이의 절대오차를 하나의 multioutput forest로 예측
- Direct Forest: SOH 한 값을 예측
- 표준화·결측 대체: 각 모델의 학습 샘플에서만 계산
- 학습: 샘플별 동일 가중치; 주 평가: 배터리별 MAE의 동일 가중 평균
- 하이퍼파라미터 탐색 없음; 단일 RF seed 비교이며 seed 안정성은 아직 평가하지 않음

Random Forest 설정은 [scikit-learn 공식 문서](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestRegressor.html)를 따릅니다.
모델은 pickle 대신 scaler·회귀 가중치·트리 배열을 NPZ로 저장합니다.
내보낸 트리는 sklearn의 float32 입력 비교 방식을 재현합니다.

## 분할과 평가

8개 셀 중 1개를 검증에서만 사용합니다. 나머지 7개에서 생성된 기존 OOF 라벨만 해당 Selector에 제공합니다.
라벨을 생성한 Predictor는 해당 라벨 셀도 제외한 6개 셀로 학습한 모델입니다.
길이별 평가 Predictor는 기존의 7셀 학습 모델을 그대로 유지합니다. 6→7셀 재학습에 따른 후보 차이는 남아 있습니다.
직접 SOH 예측 기준선은 같은 7개 학습 셀의 SOH만 사용합니다.

프로토콜과 입력·코드 해시를 고정한 후 Selector 32개와 직접 예측기 32개를 모두 학습합니다.
모든 학습이 완료된 뒤 외부 개발 검증 결과를 계산합니다.
특징 추가 효과, 모델 변경 효과, 두 변경의 결합, 같은 입력·모델의 직접 예측 대비 차이를 사전에 정한 비교로 보고합니다.
검증 결과로 특정 모델의 설정을 다시 조정하거나 최종 평가 셀을 사용하지 않습니다.

## 실행

기존 `.venv-repro`의 패키지를 유지하며 추가 라이브러리를 별도 폴더에 설치할 수 있습니다.
`requirements-selector-extra.txt`의 버전은 Python 3.9 및 기존 NumPy·SciPy 환경에 맞춰 고정했습니다.

```sh
python3 -m pip install --no-deps --target .venv-selector-extra -r requirements-selector-extra.txt

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.venv-selector-extra \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-repro/bin/python -m scripts.tests.test_change_selector

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.venv-selector-extra \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv-repro/bin/python -m scripts.experiments.train_change_selector

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.venv-selector-extra \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-repro/bin/python -m scripts.validation.verify_change_selector
```

완료된 출력은 덮어쓰지 않습니다. 재실행하려면 새 `--output`을 지정합니다.
기존 두 실험 `expanded_predictor_baseline_v1`, `expanded_selector_v1`의 PASS와 파일 해시를 확인합니다.

로컬 출력: `outputs/change_selector_v1/`

- `protocol.json`, `features.npz`, `feature_names.json`, `change_features.csv`
- `models/`, `decisions.csv`, `predictions.csv`
- `summary.csv`, `metrics_by_battery.csv`, `metrics_by_condition.csv`
- `contrasts_by_battery.csv`, `contrasts_summary.csv`, `FINDINGS.md`, `verification.json`

독립 검증은 인과적 특징 재구성, 학습 전용 통계, Ridge의 별도 최소제곱 계산,
Forest 재현 및 bootstrap leaf 평균, 독립 트리 순회, 기존 예측값 재현, 결과 집계를 확인합니다.
PASS는 구현 검증이며 성능 향상이나 통계적 유의성을 의미하지 않습니다.
평균 Lookback은 추론 비용이 아니며 이번에는 지연 시간을 측정하지 않습니다.
코드와 실행 문서만 GitHub 공개 대상이며 데이터·모델·결과는 로컬에 보관합니다.
