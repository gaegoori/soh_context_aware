# Context-aware Adaptive Temporal Selection for Battery SOH Estimation

**배터리의 상태에 맞춰 과거 기록의 길이를 선택하고, 이를 이용해 배터리 건강 상태(SOH)를 예측하는 연구입니다.**

기존 Fixed Lookback 방식은 모든 배터리와 예측 시점에 동일한 길이의 과거 기록을 사용합니다.
본 연구는 배터리의 운용 조건과 열화 상태에 따라 예측에 유리한 기록의 길이가 달라질 수 있다는 가설에서 출발합니다.

이를 위해 **기록의 길이를 선택하는 Selector**와 **SOH를 예측하는 Predictor**를 결합합니다.
Selector는 전압·전류·온도와 사용 이력을 요약한 Battery Context를 보고 최근 `10 / 20 / 50 cycles` 중 길이를 선택합니다.
선택된 길이의 Predictor는 해당 구간의 요약 통계와 누적 사용 이력 등을 이용해 SOH를 예측합니다.

고정 길이 방식과 비교하여, **시점별 길이 선택이 예측 정확도와 계산 효율성을 높일 수 있는지 검증**하는 것이 목표입니다.

현재는 **12개 배터리 중 개발용 8개 셀에서 Predictor와 여러 Selector 방식을 비교**했습니다.
최종 평가용 4개 셀은 별도로 보존하고 있으며, 아직 성능 평가에 사용하지 않았습니다.

## 1. 제안 방법

| 구성 | 역할 |
|---|---|
| **Predictor** | 최근 L개 cycle의 요약 통계와 공통 이력 정보로 SOH 예측 |
| **Selector** | Battery Context를 보고 LB10·20·50 중 사용할 길이 선택 |

현재 Predictor는 길이별로 독립 학습하는 **Context Ridge 회귀 모델**입니다.
각 모델의 입력은 공통 변수 6개와 최근 L개 cycle에서 계산하는 통계 9개로 구성됩니다.

- **공통 변수 6개:** 완료 cycle 수, 경과 시간, 누적 방전량, 마지막 cycle의 방전 시간·방전량·온도 결측률
- **구간 통계 9개:** 방전 시간의 평균·표준편차·추세, 전압·전류·온도의 평균·표준편차

L은 미래 예측 기간이 아니라 **최근 통계를 계산하는 과거 구간의 길이**입니다.
세 Predictor는 동일한 시점의 동일한 SOH 정답을 예측합니다. 누적 사용 이력은 모든 길이에 공통으로 포함됩니다.

Selector 학습에는 Predictor가 학습하지 않은 배터리에서 계산한 **OOF(out-of-fold) 예측 오차**를 사용합니다.
가장 작은 오차의 길이를 분류하거나, 길이별 오차를 예측하여 선택하는 방식을 비교했습니다.
실제 예측 시에는 정답 SOH 없이 Context로 길이를 선택하고, 해당 Predictor 하나를 실행합니다.

기본 Selector 입력은 최근 10회 통계를 포함한 Context 15개입니다.
추가 실험에서는 최근 5회와 이전 5회의 운용 차이 6개를 더한 21개 입력도 비교했습니다.
정답 SOH, 미래 정보, 배터리 ID는 모델 입력에 포함하지 않습니다.

## 2. 데이터와 분할

NASA Randomized Battery Usage Dataset의 상온 배터리 12개를 사용합니다.

| 운용 조건 | 학습·검증용 | 최종 평가용 |
|---|---|---|
| 균등 분포 전류 | RW3, RW4, RW5, RW6 | — |
| 저전류 중심 | RW13, RW14 | RW15, RW16 |
| 고전류 중심 | RW17, RW18 | RW19, RW20 |
| **배터리 수** | **8개** | **4개** |

- **개발 검증:** 8개 셀 중 7개로 학습하고 1개에서 검증하는 Leave-One-Battery-Out 교차검증
- **OOF 생성:** 각 학습용 7개 안에서 Predictor 학습 6개·라벨 생성 1개를 교대
- **최종 평가:** RW15·RW16·RW19·RW20은 학습과 모델 선택에서 제외
- **정규화·결측 대체:** 해당 분할의 학습 데이터에서만 통계 계산

SOH는 실제 기준 방전 용량을 각 셀의 첫 유효 기준 용량 `Q0`로 나눈 값입니다.
기준 측정 전에 완료된 cycle만 입력하며, 새 배터리에도 초기 기준 용량이 제공되는 설정입니다.
현재 학습 데이터는 `data/processed/expanded_qc_v1/`이며, 공통으로 유효한 50-cycle 이력을 가진 샘플을 비교합니다.
온도 품질 처리와 입력 종료 후 최대 120시간의 기준 측정 범위를 적용했습니다.

## 3. 진행 상황과 평가

| 실험 | 상태 |
|---|---|
| 데이터 확장 및 품질 처리 | 완료 |
| LB10·20·50 Context Ridge 학습 및 OOF 오차 생성 | 완료 |
| 분류형·오차 예측형·threshold 적용 Selector 비교 | 완료 |
| 운용 변화 정보 추가 및 선형·Random Forest Selector 비교 | 완료 |
| 선택 확률로 가중한 OOF 오차를 최소화하는 Selector | **계획 단계** |
| 예약한 4개 셀의 최종 성능 평가 | **미실시** |

고정 LB10·20·50, 학습 측 OOF로 고른 고정 길이, Adaptive 방식,
Context로 SOH를 직접 예측하는 모델을 비교합니다.
Oracle은 실제 정답을 보고 후보 중 최소 오차를 고르는 **사후 참고값**입니다.
주 지표는 배터리별 SOH MAE를 동일 가중 평균한 값이며, RMSE도 함께 확인합니다.

현재 비교는 개발용 셀을 이용한 탐색 실험입니다. 구현 검증 PASS가 성능 개선을 의미하지는 않습니다.
최근 확장 실험에서는 실제 추론 시간을 측정하지 않았으며, 평균 Lookback만으로 속도 향상을 판단하지 않습니다.
다음 실험에서는 Predictor를 고정하고 Selector의 학습 손실 변경이 최종 SOH 오차를 줄이는지 확인할 계획입니다.

## 4. 실행 방법

기준 실행 환경은 Python 3.9.6, macOS arm64 CPU입니다. 새 환경은 프로젝트 최상위에서 구성합니다.

```bash
python3 -m venv .venv-repro
.venv-repro/bin/python -m pip install -r requirements.txt -r requirements-selector-extra.txt
```

원본 데이터는 별도로 준비해야 합니다. 아래 문서 순서대로 데이터 준비·학습·검증을 진행합니다.
기존 `.venv-selector-extra` 폴더를 사용하는 환경의 실행 방법은 운용 변화 실험 문서에 있습니다.

| 순서 | 실행 문서 |
|---|---|
| 1. 원본 자료 준비·전처리 | [DATA_EXPANSION.md](DATA_EXPANSION.md) |
| 2. 품질 처리·Predictor 학습 | [EXPANDED_BASELINE_EXPERIMENT.md](EXPANDED_BASELINE_EXPERIMENT.md) |
| 3. Selector 비교 | [EXPANDED_SELECTOR_EXPERIMENT.md](EXPANDED_SELECTOR_EXPERIMENT.md) |
| 4. 운용 변화 정보·비선형 모델 비교 | [CHANGE_SELECTOR_EXPERIMENT.md](CHANGE_SELECTOR_EXPERIMENT.md) |

실행 명령은 `python -m scripts.<폴더>.<파일명>` 형식입니다. 예를 들어 아래 명령은 실행 옵션만 표시합니다.

```bash
.venv-repro/bin/python -m scripts.experiments.train_expanded_baselines --help
```

완료된 실험의 출력은 덮어쓰지 않습니다. 재실행할 때는 새 `--output` 경로를 지정하고,
후속 실험의 입력 경로도 해당 결과에 맞춰 설정합니다.
결과는 `outputs/` 아래에 저장하며, `summary.csv`, `metrics_by_battery.csv`, `FINDINGS.md`에서 확인합니다.

## 5. 저장소 구성

```text
scripts/
├── preprocessing/  데이터 준비·전처리
├── experiments/    학습·평가 실행
├── models/         특징 계산·모델 정의
├── validation/     저장 데이터·모델·결과 검증
├── tests/          코드 동작 검사
├── plotting/       결과 그래프 생성
└── README.md       파일별 역할과 실행 안내
```

전체 파일 설명은 [scripts/README.md](scripts/README.md)를 참고하세요.
**GitHub에는 코드와 실행 문서만 포함**합니다.
원본·전처리 데이터, 학습 모델, 예측값, 결과 보고서, 생성된 그림과 실행 환경은 로컬에 보관합니다.

## 6. 이전 실험

초기 연구는 RW3~RW6의 4개 셀과 cycle 신호 시퀀스를 입력받는 신경망 Predictor를 사용했습니다.
초기 실험의 5개 seed 평균 라벨·추론 시간 측정은 해당 설정의 결과이며,
현재의 결정론적 Ridge 학습과 구분합니다. Random Forest 추가 실험은 seed 42 한 번으로 비교했습니다.

- [초기 Predictor·OOF 라벨](LOOKBACK_EXPERIMENT.md) / [최초 Adaptive Selector](ADAPTIVE_SELECTOR_EXPERIMENT.md)
- [Context·Predictor 입력 비교](CONTEXT_HISTORY_EXPERIMENT.md) / [오차 예측형 Selector](ERROR_SELECTOR_EXPERIMENT.md)
- [Context Ridge 재설계](TEMPORAL_CONTEXT_REDESIGN.md) / [Predictor 재학습과 Selector 전이](SELECTOR_TRANSFER_EXPERIMENT.md)

데이터 출처와 다운로드 안내는 [DATA_EXPANSION.md](DATA_EXPANSION.md)에 있습니다.
관련 문헌: Bole, B., Kulkarni, C., & Daigle, M. (2014). *Adaptation of an Electrochemistry-based Li-Ion Battery Model to Account for Deterioration Observed Under Randomized Use.* Annual Conference of the Prognostics and Health Management Society.
