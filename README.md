# Adaptive Temporal Context Selection for Battery SOH Estimation

**배터리의 운용 이력과 열화 상태에 따라 SOH 예측에 필요한 과거 정보의 길이를 선택하는 연구**입니다.

NASA Randomized Battery Usage Dataset을 사용해 `Lookback = 10 / 20 / 50`을 비교하고,
예측 시점까지 관측된 Battery Context로 적절한 길이를 선택하는 Adaptive Context Selector를 개발합니다.
현재는 cycle 단위 전처리, Fixed Lookback Predictor 학습, 미학습 샘플의 예측 오차 비교,
selector 학습용 lookback 라벨 생성까지 구현·실행했습니다.
Adaptive Selector의 성능과 계산 효율 개선 여부는 후속 실험에서 검증합니다.

## 1. 연구 배경

Fixed Lookback 방식은 검증 데이터에서 하나의 과거 구간 길이를 선택하고,
이후 모든 배터리와 예측 시점에 동일한 길이를 적용합니다.
그러나 배터리별 충·방전 패턴, 전류, 온도, 사용 이력과 열화 양상이 다르면
SOH를 예측할 때 유용한 과거 정보의 범위도 달라질 수 있습니다.

짧은 이력은 최근 상태를 반영하는 데 유리할 수 있고,
긴 이력은 장기적인 변화나 운용 패턴을 파악하는 데 도움이 될 수 있습니다.
어느 조건에서 어떤 길이가 유리한지는 배터리별·열화 단계별 비교로 확인합니다.

> **연구 가설:** SOH 예측에 유리한 temporal context의 길이는 배터리와 예측 시점에 따라
> 달라질 수 있으며, 관측 가능한 Battery Context를 이용해 이 길이를 선택할 수 있다.

## 2. Research Questions

| 질문 | 검증 내용 |
|---|---|
| **RQ1. 배터리마다 유리한 길이가 다른가?** | 동일 조건의 LB10·20·50 Predictor를 학습하고 배터리별 오차와 순위의 seed 안정성을 비교합니다. |
| **RQ2. 같은 배터리에서도 열화 단계에 따라 달라지는가?** | Early / Middle / Late 단계별 오차와 lookback 선택 분포를 분석합니다. |
| **RQ3. Battery Context로 길이를 선택할 수 있는가?** | 운용 이력에서 생성한 Context를 입력받아 LB10·20·50을 선택하는 selector를 학습합니다. |
| **RQ4. Adaptive Selection이 정확도와 효율성에 도움이 되는가?** | Fixed Lookback 대비 SOH 예측 오차, 평균 입력 길이, 실제 추론 비용을 비교합니다. |

여기서 **optimal lookback**은 지정한 후보·Predictor·학습 설정에서 관측된 오차가 가장 작은 길이입니다.
배터리에 필요한 정보 길이가 모델과 무관하게 결정된다는 의미는 아닙니다.

## 3. 전체 연구 흐름

```mermaid
flowchart TD
    A["NASA RW3 / RW4 / RW5 / RW6"] --> B["Cycle 구성 · 품질 처리 · 정답 생성"]
    B --> C["Cycle 신호: 128 × 4"]
    B --> D["Battery Context: 15 features"]
    C --> E["배터리 단위 분할 · 학습 통계로 정규화"]
    E --> F["Fixed Predictor: LB10 / LB20 / LB50"]
    F --> G["미학습 샘플의 동일 SOH 정답 예측"]
    G --> H["오차 비교 · 최적 Lookback 라벨 생성"]
    H --> I["Context + Lookback Label"]
    D --> I
    I -.-> J["후속: Adaptive Context Selector 학습"]
    J -.-> K["후속: 길이 선택 → SOH 예측"]
    K -.-> L["후속: 외부 테스트 정확도 · 비용 비교"]
```

| 단계 | 현재 상태 |
|---|---|
| 원본 데이터의 cycle 구성, 품질 처리, SOH 정답 생성 | 완료 |
| Predictor 입력과 selector용 Context 생성 | 완료 |
| 배터리 단위 분할 및 학습 전용 정규화 | 완료 |
| 동일 조건의 LB10·20·50 학습 및 미학습 샘플 예측 | 완료 |
| Seed별·배터리별·Stage별 내부 오차 집계와 선택 라벨 생성 | 완료 |
| Adaptive Context Selector 학습 및 Predictor 연결 | 예정 |
| 외부 테스트 성능, 계산 비용 및 추가 ablation 검증 | 예정 |

## 4. Dataset과 예측 문제

사용 데이터는 NASA Randomized Battery Usage Dataset 중
`Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post`의
리튬이온 배터리 **RW3, RW4, RW5, RW6**입니다.
원본에는 시간, 전압, 전류, 온도, 충·방전 단계 설명과 기준 방전 기록이 포함됩니다.

### Cycle 입력

하나의 RW cycle은 여러 random-walk 방전 step과 그 사이의 짧은 휴지를 합친
**한 번의 완전한 RW 방전 구간**으로 정의합니다.
충전 신호와 방전 종료 후의 휴지는 입력에서 제외합니다.

- 전체 방전 구간을 시간축 기준 128점으로 선형 보간합니다.
- 채널은 `Voltage / Current / dV / Temperature`입니다.
- `dV`는 보간한 전압의 인접 차분이며 시간 미분 `dV/dt`가 아닙니다.
- 온도 범위 밖의 값을 결측으로 처리하고 관측 마스크를 보존합니다.
- 부적합 cycle의 위치를 유지하며, 해당 cycle이 최근 50회에 포함된 샘플은 제외합니다.
- 결측 대체와 정규화는 각 fold의 학습 데이터 통계만 사용합니다.

기준 SOH 측정 전에 완료된 cycle 수가 `n`일 때 입력은 다음과 같습니다.

| Lookback | 입력 구간 | 입력 형태 |
|---|---|---|
| LB10 | `x[n-10:n]` | `10 × 128 × 4` |
| LB20 | `x[n-20:n]` | `20 × 128 × 4` |
| LB50 | `x[n-50:n]` | `50 × 128 × 4` |

세 모델은 **동일한 샘플의 동일한 기준 SOH 측정값**을 예측합니다.
따라서 비교에 사용하는 샘플은 공통으로 유효한 50-cycle 이력을 가져야 합니다.

### SOH Ground Truth

기준 방전 전류를 적분해 용량을 계산하고, 각 배터리의 첫 유효 기준 용량으로 나눕니다.

$$
Q_k = \frac{1}{3600}\int I_k(t)\,dt,
\qquad
SOH_k = \frac{Q_k}{Q_0}
$$

주 평가에는 **실제로 측정한 기준 용량 기반 SOH**를 사용합니다.
동일 RW 이력 뒤 반복된 기준 측정은 첫 유효 측정만 사용합니다.
보간 SOH는 보조 라벨로 별도 저장하며 주 평가와 Context 입력에는 사용하지 않습니다.

모든 cycle에 실제 SOH 정답이 있는 것은 아닙니다.
또한 마지막 RW 방전과 이후 기준 측정 사이에 준비 충전·휴지·진단 시간이 존재하므로,
현재 문제는 **완료된 방전 이력으로 이후 기준 진단에서 측정한 SOH를 예측하는 설정**입니다.
테스트 배터리에도 초기 기준 용량 `Q0`가 제공된다고 가정합니다.

## 5. 데이터 분할과 Predictor 학습

배터리 하나를 외부 테스트로 제외한 뒤, 나머지 세 배터리 안에서
**학습 2개 / 라벨 생성 1개**를 번갈아 구성합니다.
예를 들어 외부 테스트가 RW6인 경우는 다음과 같습니다.

| 내부 실험 | Predictor 학습 | 미학습 예측·라벨 생성 | 외부 테스트 |
|---|---|---|---|
| A | RW4, RW5 | RW3 | RW6 |
| B | RW3, RW5 | RW4 | RW6 |
| C | RW3, RW4 | RW5 | RW6 |

외부 테스트 네 가지와 내부 라벨 생성 셀 세 가지를 조합해 총 12개 내부 fold를 사용합니다.
코드의 `validation` split은 이 단계에서 **OOF(out-of-fold) 예측과 라벨 생성용**으로 사용합니다.
이 배터리의 오차로 early stopping, epoch 또는 checkpoint를 선택하지 않습니다.
외부 테스트의 정답은 selector 학습 라벨에 사용하지 않습니다.

| 항목 | 설정 |
|---|---|
| Lookbacks | 10, 20, 50 cycles |
| Cycle encoder | `Flatten(128×4) → Linear(512,64) → ReLU → Linear(64,32) → ReLU` |
| Temporal aggregation | 과거 cycle embedding의 평균 |
| Prediction head | `Linear(32,32) → ReLU → Linear(32,1)` |
| 학습 방식 | LB별 독립 학습, 같은 seed에서 동일한 초기 가중치 |
| Loss / batch | MSE / full batch, 기준 측정 샘플별 동일 가중치 |
| Optimizer | Adam, learning rate `0.001` |
| 학습 길이 | 고정 200 epochs |
| Seeds | `42, 0, 1, 2, 3` |
| 실행 | CPU, deterministic algorithms |

200 epochs는 사전에 정한 시작 설정입니다.
후속 설정 탐색에서는 라벨 생성 대상 배터리를 모델 선택 과정에서도 제외해야 합니다.

## 6. Optimal Lookback 라벨 생성

학습에 포함되지 않은 샘플 `i`에 대해 lookback `L`, seed `s`별 절대오차를 계산합니다.

$$
e_{i,L,s} = |\hat y_{i,L,s} - y_i|
$$

대표 라벨은 **각 길이의 seed 평균 절대오차**가 가장 작은 후보로 정합니다.

$$
\bar e_{i,L} = \frac{1}{S}\sum_{s=1}^{S}e_{i,L,s},
\qquad
L_i^* = \underset{L\in\{10,20,50\}}{\arg\min}\;\bar e_{i,L}
$$

예측값을 먼저 seed 평균한 뒤 오차를 계산하는 앙상블 평가와는 계산 순서가 다릅니다.
정확한 동률이면 짧은 길이를 선택합니다.
대표 라벨과 함께 다음 정보를 보존합니다.

| 필드 | 의미 |
|---|---|
| `mean_error_lb10/20/50` | 후보별 seed 평균 절대오차 |
| `best_lookback` | 선택된 길이: 10 / 20 / 50 |
| `class_label` | 선택된 후보의 클래스: 0 / 1 / 2 |
| `error_gap_to_second_best` | 1위와 2위 후보의 평균 오차 차이 |
| `seed_agreement` | 대표 라벨과 seed별 최적 후보가 일치한 비율 |

외부 테스트별로 selector 학습 데이터셋을 따로 생성합니다.
각 데이터셋을 합쳐 공통 selector를 학습하면 제외해야 할 테스트 배터리 정보가 섞일 수 있습니다.

## 7. Battery Context와 Adaptive Selector

현재 Context는 **예측 입력의 마지막 cycle까지 관측된 운용 이력**에서 생성한 15개 변수입니다.

| 구분 | 변수 | 개수 |
|---|---|---:|
| 누적 운용 이력 | 완료된 cycle 수, 경과 시간, 누적 RW 방전량 | 3 |
| 최근 cycle | 방전 시간, 방전량, 온도 결측 비율 | 3 |
| 최근 10회 방전 시간 | 평균, 표준편차, 추세 | 3 |
| 최근 10회 전압 | 평균, 표준편차 | 2 |
| 최근 10회 전류 | 평균, 표준편차 | 2 |
| 최근 10회 온도 | 평균, 표준편차 | 2 |
| **합계** | | **15** |

Selector용 데이터에는 원본 Context와 결측 마스크를 저장합니다.
서로 다른 내부 fold의 정규화된 Context를 섞지 않고,
selector 학습 데이터에서 결측 대체와 정규화를 새로 수행하도록 구성했습니다.

실제 SOH, 보간 SOH, Stage, Battery ID, 미래 수명 및 예측 오차는 Context에 포함하지 않습니다.
SOH trend도 현재 입력에는 없으며, 후속 도입 시 추론 시점에 사용 가능한 과거 정보로 정의해야 합니다.

개발할 selector는 Context로 `P(LB10), P(LB20), P(LB50)`을 추정하고,
선택한 길이의 이력을 SOH Predictor에 전달합니다.
현재는 학습용 Context와 라벨이 준비된 단계이며 selector 모델은 후속 구현 대상입니다.

## 8. 평가 계획과 해석 범위

### 비교 대상

| 방법 | 평가 목적 |
|---|---|
| Fixed LB10 / LB20 / LB50 | 길이별 기본 예측 성능 |
| Validation-selected Fixed | 내부 검증에서 하나의 길이를 정해 외부 테스트 전체에 적용 |
| Context-only SOH Predictor | Context만으로 직접 SOH를 예측할 때와 비교 |
| Adaptive Selector + Predictor | 샘플별 길이 선택의 효과 |
| Oracle | 정답을 알고 후보 중 최소 오차 예측을 선택한 사후 비교 기준 |

SOH 정확도는 MAE·RMSE·MAPE로 평가하고, 배터리별 결과와 배터리에 동일 가중치를 주는 평균을 보고합니다.
효율성은 평균 선택 lookback과 실제 추론 시간을 함께 측정합니다.
추론 비용에는 Context 생성, selector 실행, Predictor 실행 및 캐시 갱신을 포함합니다.
입력 길이 감소가 실행 시간 감소와 비례한다고 가정하지 않습니다.

### Stage 분석

| Stage | 실제 기준 SOH 구간 |
|---|---|
| Early | `SOH ≥ 0.9` |
| Middle | `0.8 ≤ SOH < 0.9` |
| Late | `SOH < 0.8` |

Stage는 정답을 이용한 사후 분석용 구분입니다.
단계별 표본 수와 seed별 순위 변화를 함께 확인하며, 미래 수명을 기준으로 구간을 나누지 않습니다.

### 해석 시 고려할 점

- 같은 기준 측정이 여러 외부 fold에 재사용되므로 라벨 행 수와 독립 측정 수를 구분합니다.
- 실제 기준 정답이 희소하고 같은 배터리의 인접 샘플은 독립적이지 않습니다.
- 온도 결측 처리의 영향을 확인하기 위해 온도 제외 ablation이 필요합니다.
- 현재 평균 pooling은 cycle 순서를 구별하지 않습니다.
- LB별 독립 학습 결과에는 입력 길이와 학습된 가중치의 영향이 함께 포함됩니다.
- 내부 OOF 오차와 Oracle 결과는 최종 외부 테스트 성능이나 Adaptive Selector의 성능을 의미하지 않습니다.

## 9. 실행 방법

### 환경과 데이터 준비

기준 실행 환경은 Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1, PyTorch 2.8.0,
macOS arm64 CPU입니다. 패키지 버전은 [requirements-repro.txt](requirements-repro.txt)에 고정했습니다.

```bash
python3 -m venv .venv-repro
.venv-repro/bin/python -m pip install -r requirements-repro.txt
```

원본 MAT 파일을 다음 위치에 배치합니다.

```text
data/raw/
├── RW3.mat
├── RW4.mat
├── RW5.mat
└── RW6.mat
```

### 전처리부터 라벨 생성까지

프로젝트 루트에서 실행합니다.

```bash
# Cycle 전처리와 원본 대조 검증
.venv-repro/bin/python scripts/preprocess_main.py
.venv-repro/bin/python scripts/verify_preprocessing.py

# 라벨 규칙 및 학습 재현성 단위 검증
.venv-repro/bin/python scripts/test_lookback_oof.py

# Seed 42로 전체 흐름 실행
.venv-repro/bin/python scripts/train_lookback_oof.py --seeds 42
.venv-repro/bin/python scripts/verify_lookback_oof.py

# 5개 seed로 확장: 동일 설정의 완료된 모델은 재사용
.venv-repro/bin/python scripts/train_lookback_oof.py --seeds 42 0 1 2 3
.venv-repro/bin/python scripts/verify_lookback_oof.py
```

전처리는 `data/processed/main_v1/`, 학습·라벨 생성은 `outputs/context_labels_v1/`에 저장합니다.
코드·데이터·설정이 달라지면 `--output outputs/context_labels_v2`처럼 새 출력 경로를 사용합니다.
중단된 실행은 같은 명령으로 재개할 수 있습니다.
상세 프로토콜은 [LOOKBACK_EXPERIMENT.md](LOOKBACK_EXPERIMENT.md)를 참조하세요.

## 10. 저장소 구성

| 파일 | 역할 |
|---|---|
| [scripts/preprocess_main.py](scripts/preprocess_main.py) | Cycle 구성, 정답·Context 생성, 분할·정규화 |
| [scripts/battery_data.py](scripts/battery_data.py) | 동일 샘플의 LB10·20·50 입력 로더 |
| [scripts/train_lookback_oof.py](scripts/train_lookback_oof.py) | Predictor 학습, OOF 예측, 라벨 생성 |
| [scripts/verify_preprocessing.py](scripts/verify_preprocessing.py) | 원본·전처리·분할 검증 |
| [scripts/verify_lookback_oof.py](scripts/verify_lookback_oof.py) | 모델·예측·라벨·Context 검증 |
| [scripts/test_lookback_oof.py](scripts/test_lookback_oof.py) | 라벨 계산 및 학습 재현성 단위 검증 |
| [scripts/reproduce.py](scripts/reproduce.py) | 이전 step 단위 캐시를 사용하는 과거 재구성 실험 |

현재 연구는 **코드와 실행 문서만 공개**합니다.
원본·전처리 데이터, Context, 예측값, 선택 라벨, 결과 보고서 및 모델 가중치는 로컬에서 생성하며 Git에서 제외합니다.

루트의 `experiment.py`와 `outputs/` 바로 아래에 이미 게시된 파일은 이전 step 단위 예비 실험의 이력입니다.
입력 단위·모델·정답·학습 조건이 현재 cycle 단위 실험과 다릅니다.
이전 실행 설명은 [docs/legacy_preliminary.md](docs/legacy_preliminary.md)에 보존했습니다.

## Reference

데이터셋 사용 시 원본 설명에서 요청한 다음 논문을 인용합니다.

Brian Bole, Chetan Kulkarni, and Matthew Daigle.
*Adaptation of an Electrochemistry-based Li-Ion Battery Model to Account for Deterioration Observed Under Randomized Use.*
Annual Conference of the Prognostics and Health Management Society, 2014.
