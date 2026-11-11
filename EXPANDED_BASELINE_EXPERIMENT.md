# 확장 데이터 품질 처리와 개발용 Predictor 기준 실험

추가 셀을 포함한 환경에서 고정 길이 Predictor의 기준 성능과 후보 내 길이 선택의 여지를 확인합니다.
Selector 재학습 및 threshold 도입은 이 단계에 포함하지 않습니다.
모든 성능 평가는 개발용 배터리 안에서 수행합니다.

## 품질 기준을 먼저 고정

`expanded_v1`을 유지한 채 원본 MAT에서 `expanded_qc_v1`을 생성합니다.

- 상온 시험의 원신호 온도 유효 범위는 `[0, 100] C`입니다. 범위 밖 값은 보간 전에 NaN 처리합니다.
- 원래 온도 센서 값이 전부 교정되었다는 뜻은 아닙니다. 양의 온도에도 센서 오류가 남을 수 있습니다.
- 마지막 입력 cycle 종료부터 기준 방전 시작까지 최대 120시간인 측정만 주 예측 대상으로 사용합니다.
- 더 긴 간격은 `excluded_samples.csv`에 보존합니다. 측정 자체가 틀렸다는 뜻이 아니라 이 실험의 최대 예측 기간을 넘었다는 뜻입니다.
- 120시간은 예측 오차로 최적화한 값이 아닌, 모델 평가 전에 고정한 운영상 한도입니다. 고정된 예측 시차는 아닙니다.
- 모든 셀에 같은 규칙을 적용합니다. 전류 상한은 확장판의 5.25A를 유지합니다.
- 기존 셀의 역할과 초기 기준 용량 Q0 제공 가정은 유지합니다.

## 개발·최종 평가 역할

개발 셀은 RW3, RW4, RW5, RW6, RW13, RW14, RW17, RW18입니다.
최종 평가용 RW15, RW16, RW19, RW20은 품질 전처리만 진행하며 예측 성능을 계산하지 않습니다.

학습 스크립트는 `samples_development.csv`와 개발 셀의 canonical 파일만 읽습니다.
`samples_final_test.csv`나 최종 평가 셀의 신호 배열은 이 실험 입력에 포함하지 않습니다.

## 고정한 모델과 검증 방식

| 항목 | 설정 |
|---|---|
| Predictor | 동일한 15차원 Context Ridge, LB10/20/50 |
| 길이 의미 | 공통 이력·마지막 cycle 변수 6개 + 최근 L회 통계 9개 |
| 정규화 | 해당 모델의 학습 샘플에서만 평균·표준편차 계산 |
| 결측 | 학습 평균으로 대체; 학습에서 전부 결측인 열은 모든 역할에서 비활성화 |
| 손실 | mean squared SOH error + 0.1 × squared weight norm; intercept 무벌점 |
| 샘플 가중치 | 학습 샘플별 동일 가중치 |
| Seed | 결정론적 해를 한 번 계산하며 반복 seed를 만들지 않음 |
| 개발 검증 | 8개 LOBO 분할: 학습 7개 / 검증 1개 셀 |
| 내부 OOF | 각 학습 7개 안에서 학습 6개 / 라벨 1개 교대 |
| 고정 길이 선택 | 내부 OOF의 배터리별 MAE를 동일 가중 평균해 최소 길이 선택 |
| Oracle | 개발 검증 정답을 사용한 샘플별 사후 최소 오차; 배포 모델 아님 |
| 주 지표 | 8개 개발 셀의 MAE 동일 가중 평균 |

내부 OOF 후보 168개와 개발 검증 후보 24개를 모두 학습한 뒤 검증 예측을 계산합니다.
하이퍼파라미터 탐색, Selector 학습, threshold 조정, 추론 시간 측정은 수행하지 않습니다.
내부 라벨 생성 후보는 6개 셀, 검증 후보는 7개 셀로 학습하므로 후보 변화에 따른 라벨 전이 차이는 남아 있습니다.

후속 Selector는 `oof_labels.csv`를 **검증 셀별로 분리하여** 사용해야 합니다.
다른 검증 분할의 라벨까지 섞으면 해당 검증 셀을 학습한 Predictor의 정보가 유입될 수 있습니다.

## 실행

프로젝트 루트에서 실행합니다. 기존 출력 폴더는 덮어쓰지 않습니다.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-repro/bin/python -m scripts.preprocessing.prepare_expanded_quality
PYTHONDONTWRITEBYTECODE=1 .venv-repro/bin/python -m scripts.validation.verify_expanded_quality

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 .venv-repro/bin/python -m scripts.experiments.train_expanded_baselines

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-repro/bin/python -m scripts.validation.verify_expanded_baselines
```

## 로컬 산출물

- `data/processed/expanded_qc_v1/QUALITY_REPORT.md`: 공통 품질 규칙과 데이터 변경 내역
- `outputs/expanded_predictor_baseline_v1/protocol.json`: 평가 전 고정한 설정과 입력·코드 해시
- `models/`: 내부 OOF와 개발 검증용 Ridge 가중치 및 학습 셀·샘플 목록
- `oof_predictions.csv`, `oof_labels.csv`: 검증 셀을 학습에서 제외한 후속 Selector용 자료
- `summary.csv`, `metrics_by_battery.csv`, `metrics_by_condition.csv`: 개발 성능
- `selection_potential.csv`, `oracle_gap_by_battery.csv`: 사후 선택 여지 진단
- `FINDINGS.md`, `INTERPRETATION.md`, `verification.json`: 로컬 결과와 독립 검증

품질 검증은 원신호 온도를 별도로 보간하여 확인합니다. 학습 검증은 학습 통계와 특징을 독립 재구성하고,
모든 Ridge 가중치를 augmented least squares로 다시 풀어 확인합니다.
PASS는 코드·데이터·집계의 검증이며, Selector 성능 개선이나 최종 일반화의 확인이 아닙니다.
데이터 및 결과는 Git에서 제외하고 코드와 실행 문서만 공개 대상에 포함합니다.
