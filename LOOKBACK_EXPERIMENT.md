# Cycle 단위 OOF Lookback 라벨 실험

## 목적과 분할

완료된 RW 방전 cycle을 입력으로 실제 기준 SOH를 예측한다. 외부 테스트 셀은
최종 평가를 위해 제외하고, 나머지 셀에서 학습하지 않은 샘플의 lookback 오차를
비교한다. 이 단계에서는 selector를 학습하거나 외부 테스트 예측을 수행하지 않는다.

외부 테스트가 RW6일 때:

| Predictor 학습 | 라벨 생성 (기존 split의 validation) | 외부 테스트 제외 |
|---|---|---|
| RW4, RW5 | RW3 | RW6 |
| RW3, RW5 | RW4 | RW6 |
| RW3, RW4 | RW5 | RW6 |

외부 셀 네 가지 × 내부 라벨 셀 세 가지 = 12개 fold. 기존 main_v1의 split,
학습 전용 결측 대체/정규화, 공통 50-cycle 유효 이력 샘플을 이용한다.

## 고정 설정

- LB10/20/50; cycle shape 128×4 (V/I/dV/T).
- Cycle encoder: Linear(512,64) → ReLU → Linear(64,32) → ReLU.
- Cycle embedding 평균 → Linear(32,32) → ReLU → Linear(32,1).
- LB별 모델을 독립 학습하되 같은 seed에서 초기 가중치를 동일하게 재설정.
- Full-batch MSE, 샘플별 동일 가중치, Adam lr=0.001, 고정 200 epochs.
- 라벨 생성 셀로 early stopping/최적 checkpoint/학습률을 선택하지 않음.
- Seeds: 42, 0, 1, 2, 3. CPU 4 threads, deterministic algorithms.
- 200 epochs는 사전에 정한 시작 설정으로 최적임을 주장하지 않음.

## 실행

프로젝트 루트에서:

```bash
# 작은 단위 검증
PYTHONDONTWRITEBYTECODE=1 .venv-repro/bin/python -m scripts.tests.test_lookback_oof

# Seed 42 전체 36개 모델
PYTHONDONTWRITEBYTECODE=1 .venv-repro/bin/python -m scripts.experiments.train_lookback_oof --seeds 42
PYTHONDONTWRITEBYTECODE=1 .venv-repro/bin/python -m scripts.validation.verify_lookback_oof

# 동일 출력 폴더에서 5 seeds로 확장: 완료된 seed 42는 재사용
PYTHONDONTWRITEBYTECODE=1 .venv-repro/bin/python -m scripts.experiments.train_lookback_oof --seeds 42 0 1 2 3
PYTHONDONTWRITEBYTECODE=1 .venv-repro/bin/python -m scripts.validation.verify_lookback_oof
```

기본 출력은 `outputs/context_labels_v1/`. 데이터/모델 코드/설정 해시가 다르면
기존 실행에 덮어쓰지 않고 중단한다. 새 설정은 `--output`으로 새 디렉터리를 지정한다.
중단된 실행은 같은 명령으로 재개할 수 있다. 같은 출력에 두 실행을 동시에 돌리지 않는다.

## 라벨 규칙

각 sample/seed/LB의 `abs(predicted_soh - true_soh)`를 저장한다. 대표 라벨은
LB별 **seed 평균 절대오차**가 가장 작은 길이이며, 정확한 동률은 짧은 길이를 택한다.
예측값을 seed 평균한 다음 오차를 계산하는 앙상블 평가는 사용하지 않는다.

- `error_gap_to_second_best`: seed 평균 오차의 1위와 2위 차이.
- `seed_agreement`: 대표 라벨과 개별 seed 최적 라벨이 일치한 비율.
- `labels_by_seed.csv`: 각 seed의 오차와 최적 길이를 보존.

## Selector 연결

`selector_data/test_RW*/dataset.npz`의 `context_raw` 15개만 입력으로 사용한다.
원본 NaN과 `context_observed_mask`를 보존하므로 selector 학습 데이터에서 결측
대체와 정규화를 새로 fit한다. 서로 다른 내부 fold의 정규화된 Context를 합치지 않는다.
마스크 자체를 추가 feature로 사용하는 실험은 별도 ablation으로 정의한다.

`best_lookback`은 10/20/50, `class_label`은 0/1/2이다. CSV의 실제 SOH, Stage,
오차, 라벨 안정성, Battery ID 및 sample ID는 설명용이며 입력 feature가 아니다.
외부 테스트별 데이터셋을 하나로 합쳐 공통 selector를 학습하면 안 된다.
후속 selector 내부 검증에서는 predictor 라벨 생성의 의존성까지 포함한 분할을 설계한다.

## 해석 범위

같은 기준 측정을 외부 fold별로 반복 사용하므로 라벨 행 수가 독립 측정 수와 같지는 않다.
배터리 간 성능 차이와 Stage 결과는 표본 수와 반복 사용을 고려해 해석한다.
Early/Middle/Late는 SOH 기반 사후 분석이다.
여기서 계산한 Fixed/Oracle 오차는 **내부 OOF 진단**으로 외부 테스트 성능이 아니다.
모델은 평균 pooling으로 순서를 구별하지 않으며, LB별 독립 학습의 영향도 포함한다.
온도 제외 실험, selector 학습, 외부 테스트용 3개 셀 재학습, 추론 비용 비교는 후속 단계다.

## GitHub에 포함하는 범위

이 저장소에는 코드와 실행 문서만 추가한다. 이번 실험의 원본·전처리 데이터,
Context, 예측값, 선택 라벨, 결과 보고서 및 모델 가중치는 로컬에서 생성하고
업로드하지 않는다. 기존에 게시된 예비 실험 파일은 이전 이력으로 보존한다.
