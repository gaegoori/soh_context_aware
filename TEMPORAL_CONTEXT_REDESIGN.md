# Temporal Context 재설계: 통계 산출 범위 선택을 통한 SOH 예측

설계일: 2026-10-26 (Asia/Seoul). **2026-10-27 구현·실행 및 독립 검증 완료.**

실행 시 분류형 Selector와 오차 예측형 Selector를 함께 비교하도록 확장했습니다. 성능 수치와 그림은 로컬 결과 폴더에 보관합니다.

## 1. 연구 목표와 변경점

**배터리 상태에 따라 SOH 예측에 유용한 과거 기록의 범위가 달라지는지 확인하고, 관측 가능한 짧은 Context로 그 범위를 선택한다.**

기존의 Predictor–Selector 구조와 Lookback 10/20/50을 유지한다. 이번에는 강한 기준선으로 확인된 Context ridge를 Predictor로 사용한다. 같은 종류의 통계를 계산하는 범위만 10/20/50으로 바꾸어 길이의 영향을 비교한다.

- 기존 Predictor: `L × 128 × 4` 신호를 받는 신경망.
- 새 Predictor: 선택한 `L` 구간에서 만든 **15개 통계·이력 변수**를 받는 Ridge 회귀.
- Selector: 예측 시점까지 관측한 고정된 짧은 Context로 L을 선택.
- 이번 연구가 직접 비교하는 것은 **과거 구간의 통계 요약 범위**다. 원시 시계열의 순서를 모두 학습하는 모델이나 배터리의 본질적 기억 길이를 평가하는 것은 아니다.

현재 잘 작동한 기존 Context 15개에는 이미 최근 10 cycles의 통계가 들어 있다. 따라서 기존 Context-only ridge를 새 실험의 **FixedContext10**으로 정확히 재현하는 것이 출발점이다.

## 2. Research Questions

| 질문 | 검증 |
|---|---|
| RQ1. 최근 구간의 통계는 누적 사용 이력·마지막 cycle 정보에 추가 이득을 주는가? | Usage3, LastState6, FixedContext10/20/50 비교 |
| RQ2. 같은 종류의 통계를 어떤 길이에서 계산하는 것이 유리한가? | 동일 15개 변수와 Ridge 설정으로 10/20/50 비교 |
| RQ3. 길이별 오차 차이가 배터리·상태에 따라 달라지고, 이를 짧은 Context로 예측할 수 있는가? | OOF 오차·라벨 분석 후 Selector의 외부 셀 성능 확인 |
| RQ4. Adaptive 방식은 고정 길이보다 정확하거나 실제 계산 비용이 낮은가? | 동일 샘플에서 정확도·선택 길이·총 추론 시간 측정 |

## 3. 데이터와 예측 시점

- 기존 `data/processed/main_v1/`의 RW3/RW4/RW5/RW6와 **62개 측정 SOH 목표**를 재사용한다.
- 각 샘플은 동일한 유효 50-cycle 이력을 확보하며, 모든 방법이 동일한 목표를 예측한다.
- 마지막 입력은 `history_end_exclusive=n`의 `n-1` cycle이다. L개 입력은 `x[n-L:n]`이다.
- 기준 방전 시작 전에 완료된 RW 이력만 사용한다. 기준 방전으로 측정한 현재 정답, 보간 SOH, Stage, cell ID, 미래 수명은 입력에서 제외한다.
- 누적 방전량은 과거 RW 방전의 적분량이다. 현재 목표의 기준 용량을 입력으로 사용하지 않는다.
- 현재 설정은 완료된 RW 방전 후, 이어지는 기준 진단 SOH를 예측한다. 테스트 셀의 초기 기준 용량 Q0가 주어진다는 기존 가정을 유지한다.
- 이 네 셀의 이전 결과를 이미 보았으므로 **탐색 실험**이다. 최종 확증에는 사전에 분리한 새로운 셀·운용 조건이 필요하다.

## 4. Predictor 입력: 고정 6개 + 길이별 통계 9개

### A. 모든 L에서 동일한 정보: F6

| 번호 | 변수 |
|---|---|
| 1 | 현재까지 완료한 RW cycle 수 |
| 2 | 현재까지 경과 일수 |
| 3 | 현재까지 누적 RW 방전량 Ah |
| 4 | 마지막 RW cycle의 방전 시간 |
| 5 | 마지막 RW cycle의 방전량 Ah |
| 6 | 마지막 RW cycle의 온도 결측 비율 |

### B. 최근 L cycles에서 계산하는 정보: H9(L)

| 묶음 | 통계 | 변수 수 |
|---|---|---:|
| 방전 시간 | cycle별 duration의 평균·표준편차·선형 기울기 | 3 |
| 전압 | 최근 L개 cycle의 128점 신호 전체에서 평균·표준편차 | 2 |
| 전류 | 위와 동일 | 2 |
| 온도 | 위와 동일, NaN 제외 | 2 |
| 합계 | | 9 |

`C_L = [F6, H9(L)]`로 15차원 입력을 만든다.

- L10: 기존 Context 15개와 같은 값·순서·결측 규칙.
- L20/L50: H9의 계산 범위만 변경. F6는 그대로 유지.
- 표준편차는 ddof=0, duration 기울기는 실제 cycle 인덱스 `0..L-1`에 대한 최소제곱 기울기다.
- V/I/T는 resampled 값으로 계산한다. dV는 현재 15개 Context에 없는 변수이므로 이번 비교에 새로 추가하지 않는다.
- 기존 C10을 통째로 넣고 중복되는 통계를 덧붙이지 않는다. 세 후보의 입력 변수 수·종류를 15개로 맞춘다.
- 누적 변수는 L보다 오래된 이력을 요약한 상태다. 따라서 L은 모델이 접근한 모든 과거 정보의 최대 길이가 아니라 **최근 통계를 산출하는 창의 길이**다.

## 5. 모델과 고정 학습 설정

| 방법 | 입력 | 역할 |
|---|---|---|
| Usage3 | F6의 1–3번 | 누적 사용 이력만으로 설명되는 성능 |
| LastState6 | F6 전체 | 마지막 cycle 정보까지 사용한 기준선 |
| FixedContext10 | F6 + H9(10) | 기존 강한 Context ridge 재현 / 주 기준선 |
| FixedContext20 | F6 + H9(20) | 중간 범위의 통계 |
| FixedContext50 | F6 + H9(50) | 긴 범위의 통계 |

모두 같은 Ridge 방법을 사용한다. 고정 후보 세 개는 각각 15개 가중치와 절편으로 구성하고 독립적으로 학습한다. Usage3/LastState6는 변수 수가 다른 원인 분석용 기준선이다.

목적함수: `mean((y - Xw - b)^2) + 0.1 * ||w||²`, 절편 b는 규제하지 않는다. float64 선형해법으로 학습하고 첫 실행에서는 파라미터를 탐색하지 않는다. 기존 ridge 구현의 규약을 따른다.

scikit-learn의 Ridge는 제곱오차 **합**을 사용하므로 직접 대체할 경우 `alpha = N_train × 0.1`로 맞춰야 같은 목적함수가 된다. [공식 Ridge 문서](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

- 각 split·L의 학습 목표 행에서만 변수별 결측 대체 평균·정규화 통계를 산출한다.
- 검증/라벨 생성/외부 셀에는 해당 학습 통계를 적용한다. 셀별로 따로 정규화하지 않는다.
- 학습에서 전부 결측인 열은 모든 역할에서 비활성 0열로 유지하고 기록한다. 모델 입력 크기는 유지한다.
- 학습에서 상수인 열은 표준편차를 1로 대체한다. 보간 SOH로 정답 수를 늘리지 않는다.
- 결정적인 모델이므로 같은 데이터를 5개 seed로 반복하여 독립 실행처럼 집계하지 않는다.

## 6. 셀 단위 분할과 두 단계 학습

### 외부 셀을 RW6로 둔 예

| 내부 회전 | Predictor 학습 | OOF 예측·새 라벨 생성 | 외부 평가 |
|---|---|---|---|
| A | RW4, RW5 | RW3 | RW6 |
| B | RW3, RW5 | RW4 | RW6 |
| C | RW3, RW4 | RW5 | RW6 |

1. 각 행에서 C10/C20/C50 Predictor를 2개 셀로 학습한다.
2. 학습하지 않은 1개 셀의 같은 목표들에서 세 예측과 오차를 얻는다.
3. 세 회전의 OOF 결과를 합쳐 RW3/RW4/RW5의 Selector 학습 자료를 만든다.
4. 최종 Predictor 3개와 Usage3/LastState6는 RW3/RW4/RW5 전체로 다시 학습한다.
5. Selector까지 학습을 마친 뒤 RW6에서 모든 방법을 평가한다.
6. RW3/RW4/RW5/RW6를 각각 외부 셀로 바꾸어 4회 반복한다.

모든 정규화와 Selector 학습은 해당 역할의 학습 셀만 사용한다. **이번 내부 3회전은 OOF 라벨 생성용이며 하이퍼파라미터 검증용으로 재사용하지 않는다.** 첫 설계는 파라미터를 고정하여 이중 사용을 피한다. 이후 튜닝이 필요하면 라벨 대상 셀을 제외한 학습 데이터 내부에 별도 선택 절차를 둔다. [평가와 모델 선택 분리](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)

배터리 한 셀을 하나의 그룹으로 분리하며, 같은 셀의 인접 목표를 무작위로 학습/평가에 섞지 않는다. [Leave-one-group-out](https://scikit-learn.org/stable/modules/cross_validation.html#leave-one-group-out)

전체 계획: OOF Predictor 12분할 × 3길이 = 36개, 최종 Predictor 4분할 × 3길이 = 12개, Usage3/LastState6 4분할 × 2 = 8개. 총 56개 Ridge Predictor·대조군 학습과 분류형 Selector 4개, 오차 예측형 Selector 4개입니다.

## 7. 새 OOF 정답과 Selector

### 정답 생성

샘플 i에서 `e_i,L = abs(yhat_i,L - y_i)`를 계산하고 `L*_i = argmin_L e_i,L`로 정한다. 정확한 동률은 짧은 길이를 선택한다. seed 평균은 사용하지 않는다.

기존 신경망의 OOF 라벨은 새 Predictor에 재사용하지 않는다. 새 예측·라벨은 `outputs/temporal_context_ridge_v1/`에 분리한다.

저장 항목: sample ID, outer/inner cell, 세 예측·절대오차, 최적 길이, 최저 오차와 두 번째 오차의 차이, C10 원본과 결측 마스크. 오차 차이를 함께 보고 거의 동등한 후보와 큰 차이가 있는 후보를 구분한다. 미세한 라벨 차이만으로 상태별 최적 길이가 존재한다고 주장하지 않는다.

### Selector의 역할

- 입력: **고정된 C10 15개 변수**. 누적 상태와 최근 10-cycle 정보만 사용한다.
- 출력: Lookback 10/20/50에 대한 확률과 선택 길이.
- 모델: 기존과 같은 L2 규제 softmax 선형 분류기.
- 손실: 평균 교차엔트로피 + `0.1/2 × ||W||²`, 절편 비규제, 가중치·절편 0 초기화, L-BFGS-B maxiter=500 / ftol=1e-12 / gtol=1e-8, 수렴 확인. 클래스 가중치나 oversampling 없음.
- 학습용 scaler: 세 내부 회전의 **원본 C10**을 모은 뒤 해당 외부 fold의 학습 셀 자료에 한 번 fit한다. 내부 fold마다 정규화된 값을 혼합하지 않는다.
- 학습 라벨이 2종류면 관측된 두 후보만 학습하고 나머지 확률은 0으로 둔다. 1종류면 그 후보를 고정 선택하고 이를 명시한다. 모든 길이가 나타나도록 라벨을 강제로 만들지 않는다.
- 테스트 예측 전에는 C20/C50이나 세 Predictor를 모두 계산하여 최적값을 찾지 않는다. C10으로 L을 고른 뒤 필요한 C_L과 해당 Predictor 하나만 계산한다.
- L10을 고르면 이미 계산한 C10을 재사용한다. L20/L50이면 F6를 재사용하고 H9(L)만 추가 산출한다.

학습 정답은 2개 셀로 학습한 Predictor의 오차, 실제 사용 Predictor는 3개 셀로 재학습한 모델의 오차다. 이 차이로 선택이 달라질 수 있으므로, 학습 라벨 정확도만으로 성공을 판단하지 않는다.

### 추가 비교: 오차 예측형 Selector

- 입력과 정규화는 분류형과 동일한 C10을 사용합니다.
- 세 Ridge 후보의 **새 OOF 절대오차**를 각각 예측하는 선형 회귀 세 개를 학습합니다. 기존 신경망 라벨을 재사용하지 않습니다.
- 출력별 평균제곱오차 + `0.1 × ||w||²`, 절편 비규제, 균일 샘플 가중치로 고정합니다.
- 예상 오차 점수가 가장 작은 길이를 선택합니다. 정확한 동률은 짧은 길이를 선택합니다.
- 선형 외삽으로 음수 점수가 나올 수 있어 개수를 기록합니다. 이번에는 순위 보존을 위해 자르지 않습니다.
- 학습 목표는 기존 분류형과 다르지만 입력·학습 행·Predictor 후보·평가 샘플은 동일합니다.

## 8. 최종 비교와 판단

모든 구성을 새 외부 결과를 보기 전에 고정하고 모든 학습을 끝낸다. 외부 4개 셀 전부의 결과를 보고한다. 결과가 좋은 셀이나 길이만 남기지 않는다.

| 비교 대상 | 정의 |
|---|---|
| Usage3 / LastState6 | 누적 사용 이력과 마지막 cycle 기준선 |
| FixedContext10 / 20 / 50 | 같은 통계를 고정 L에서 계산 |
| GlobalBestOOF | 내부 OOF의 셀별 MAE를 동일 가중 평균해 고른 L, 동률은 짧은 L |
| LabelMajority | 학습 OOF에서 가장 흔한 L, 동률은 짧은 L |
| Adaptive | 고정 C10 분류형 Selector가 고른 L의 Predictor |
| ErrorAware | 고정 C10으로 길이별 OOF 오차를 예측한 뒤 최소 점수의 L을 선택 |
| Oracle | 최종 Predictor 3개의 외부 샘플별 실제 오차 중 최솟값. 사후 진단 전용 |

### 주 평가

1. 셀별 SOH MAE·RMSE와 4개 셀의 동일 가중 평균.
2. 동일 샘플·셀에서 Adaptive − FixedContext10, Adaptive − GlobalBestOOF 차이.
3. 각 셀에서 개선/악화 방향과 크기. 독립 셀은 4개이며, 샘플 62개를 독립 셀 62개처럼 취급하지 않는다.
4. Oracle까지 남은 오차 차이와 선택 후 실제 추가 오차(regret). Selector의 분류 정확도는 보조 지표다.
5. 선택 길이 분포·평균, stage별 성능과 표본 수. Stage는 실제 SOH 기반 사후 분석용이며 입력이 아니다.

### 다음 연구 단계로 넘어가는 판단

- Fixed20/50 또는 Oracle이 Fixed10보다 의미 있게 낮은 오차를 보이는지 **차이의 절대 크기와 셀별 방향**으로 확인한다. 아주 작은 차이는 별도로 표시한다.
- 다음 독립 검증의 우선 후보는 전체 셀 평균 MAE가 감소하고 4개 중 3개 이상 셀에서 같은 방향으로 개선된 구성으로 정한다. 이는 탐색 단계의 선택 규칙이며 통계적 유의성 기준이 아니다.
- Adaptive가 GlobalBestOOF와 Fixed10 모두에 이득을 보이면 적응형 선택을 독립 데이터에서 확인한다.
- Oracle에는 여유가 있지만 Adaptive가 실패하면 길이 선택의 학습 가능성·오차 차이·분포 이동을 점검한다.
- 모든 셀에서 한 길이가 유리하면 해당 길이의 장점을 분석한다. 길이별 추가 이득이 없으면 어떤 조건에서 짧은 요약이 충분한지로 연구 범위를 구체화한다.
- 현재 외부 결과로 골라낸 구성의 성능은 선택과 독립된 새 셀에서 다시 확인해야 한다.

## 9. 효율성 평가

L이 바뀌어도 Ridge 입력은 항상 15개다. **평균 선택 L 감소가 회귀 계산량이나 지연 감소를 뜻하지 않는다.** 효율은 실제 측정으로 별도 검증한다.

- 같은 CPU 1 thread, batch=1, 5회 warmup 후 샘플·방법별 100회 반복.
- 방법 실행 순서는 고정된 난수 seed로 섞어 순서 효과를 줄인다. 이 seed는 학습 반복을 의미하지 않는다.
- Fixed 방법: H9(L) 산출 + 정규화 + Predictor.
- Adaptive: C10 산출 + Selector 정규화·선택 + 선택한 추가 H9(L) + Predictor 정규화·예측.
- F6와 누적 메타데이터는 모든 방법에서 같은 인과적 캐시를 사용한다. 필요한 창 통계를 캐시할 경우 그 업데이트 비용과 저장 비용을 별도로 보고한다.
- 원시 segmentation/resampling, 디스크 로딩, 최초 누적 캐시 구축은 제외하며 측정 범위를 명시한다.
- 총 시간 중앙값/p95, 정확도, 평균 L을 함께 보고한다. adaptive가 최대 L=50에 접근할 수 있도록 보관하는 버퍼가 있으므로 평균 L을 저장 메모리 절감량으로 바꾸어 표현하지 않는다.
- 정확도만 개선되고 비용이 늘어도 그 결과를 그대로 보고하며 효율 개선을 주장하지 않는다.

## 10. 구현 순서와 완료 조건

1. **가변 창 Context 추출:** `context_at_L(history,n,L)`을 새 모듈에 구현. 기존 `context_at` 및 완료된 실험의 해시 대상 코드는 수정하지 않는다.
2. **재현 검사:** L10의 원본 Context·NaN 위치와 기존 C10이 일치하는지, FixedContext10이 기존 ridge 예측을 재현하는지 확인한다.
3. **인과성/분할 검사:** n 이후 원시값과 현재 목표를 변조해도 입력이 같아야 한다. scaler에 평가 셀이나 라벨 대상 셀이 들어가지 않아야 한다.
4. **고정 설정 비교:** 36개 OOF 후보와 20개 최종 회귀를 학습하고 새 라벨·오차 간격을 저장한다.
5. **Selector:** 사전 고정 설정으로 외부 fold별 학습. 한 클래스/두 클래스 예외를 기록한다.
6. **전체 평가:** 모든 fit 완료 후 외부 평가·시간 측정. 저장된 모델에서 예측과 지표를 독립 재현한다.
7. **결과 정리:** 셀×길이 성능 표, Fixed10 대비 오차 차이, SOH 예측 곡선, 오차 간격·선택 분포, 정확도–지연 그림.
8. **독립 확인 준비:** 서로 다른 운용 조건과 충분한 과거 기록을 가진 추가 셀을 확보하고 실험 설정을 동결한다.

구현 파일: `scripts/models/temporal_context_models.py`, `scripts/experiments/train_temporal_context.py`, `scripts/validation/verify_temporal_context.py`, `scripts/tests/test_temporal_context.py`, `scripts/plotting/plot_temporal_context.py`.

기존 결과·코드·프로토콜은 보존했습니다. 새 결과와 그림은 Git에서 제외되는 `outputs/temporal_context_ridge_v1/`에 저장합니다. 공개 범위는 코드와 실행 문서입니다.


## 11. 실행과 결과 확인

저장소 루트에서 실행합니다. 기존 전처리 및 검증된 `adaptive_selector_v1` 결과가 있어야 합니다.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-repro/bin/python -m unittest discover -s scripts -p 'test_temporal_context.py' -v
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-repro/bin/python -m scripts.experiments.train_temporal_context
PYTHONDONTWRITEBYTECODE=1 .venv-repro/bin/python -m scripts.validation.verify_temporal_context
```

기존 출력 폴더를 덮어쓰지 않습니다. 신규 실행에는 빈 `--output` 경로가 필요하며, 학습과 평가는 `--phase fit` / `--phase evaluate`로 분리할 수 있습니다.

NumPy와 Matplotlib이 있는 환경에서 그림과 결과 설명을 생성합니다.

```bash
python -m scripts.plotting.plot_temporal_context
```

- `protocol.json`: 실행 전 고정 설정, 입력 및 코드 해시
- `reproduction.json`: 기존 C10 특징 및 Context Ridge 학습 계수 재현
- `oof_predictions.csv`, `labels.csv`: 새 Predictor의 오차 및 Selector 정답
- `summary.csv`, `metrics_by_battery.csv`, `paired_comparisons.csv`: 전체·셀별 결과
- `selections.csv`, `selection_diagnostics.csv`: 선택 점수와 사후 오차 차이
- `verification.json`: 독립 특징 재구성·회귀 해법·예측·집계 검증
- `FINDINGS.md`, `figures/`: 결과 설명 및 전체 비교·셀별 차이·SOH 곡선·정확도와 비용 그림
