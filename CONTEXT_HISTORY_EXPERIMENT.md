# Context와 과거 이력의 역할 분리 실험

## 목적과 해석 범위

첫 Adaptive 실험에서 Context-only ridge가 모든 고정 길이 Predictor와 Oracle보다 낮은 오차를 보였다. Selector 개선에 앞서 입력 정보와 순서 표현을 분리하여 확인한다.

**이전 결과를 이미 확인한 RW3–RW6에서 수행하는 탐색 실험이다.** 각 fold의 학습에는 해당 외부 셀을 넣지 않지만, 연구 방향 선택에 이전 테스트 결과가 사용되었으므로 새 독립 검증으로 해석하지 않는다. 확증은 별도 셀에서 필요하다.

## 고정 설정

- 기존 62개 측정 SOH 목표 사용, 3개 셀 학습 / 1개 셀 평가를 4회 회전.
- 모든 LB에서 동일 목표 샘플 사용. 보간 SOH, 현재 목표 SOH, stage, cell ID를 입력에서 제외.
- LB 10/20/50, seed 42/0/1/2/3, full-batch Adam(lr=.001), MSE, 200 epochs, CPU 4 threads. 조기 종료/튜닝 없음.
- 주 지표: 4개 셀의 MAE를 동일 가중 평균. seed 표준편차는 독립 셀에 대한 신뢰구간이 아님.
- 전처리 통계: 신호와 cycle 메타데이터는 학습 50-cycle window의 고유 cycle 합집합, Context는 학습 목표 시점만 사용.
- 설정·코드·입력 해시를 저장한 뒤 240개 신경망과 16개 ridge 학습을 모두 완료하고 평가.
- 결과를 보고 추가 설정을 시도하지 않는다. 이 실험에서는 새 추론 시간 측정이나 효율 개선 주장을 하지 않는다.

## A. Context feature 묶음 제거

모두 같은 ridge 회귀, L2=.1, 절편 비규제. 각 셀 분할당 한 번만 학습하는 결정적 모델이다.

| 설정 | 사용 변수 |
|---|---|
| ContextAll | 기존 15개 전체 |
| ContextNoDurationAh | 누적/최근 방전량, 최근 방전 시간, 최근 10개 방전 시간 평균·표준편차·기울기를 제거한 9개 |
| ContextUsageOnly | 완료 cycle 수, 경과 일수, 누적 방전량 3개 |
| ContextNoTemperature | 온도 평균·표준편차와 온도 결측 비율을 제거한 12개 |

Context는 최근 10-cycle 통계와 누적 이력을 포함한다. History-free 기준선이 아니다. 과거의 방전량은 용량과 밀접할 수 있으며, 실제 추론 시 관측 가능한지 확인해야 한다. 제거 실험은 상관된 변수 묶음에 대한 의존성을 보여 주며 개별 변수의 인과적 기여를 식별하지 않는다.

## B. Predictor 입력 단계별 추가

| 설정 | resampled 신호 | cycle별 duration/Ah | 15 Context | cycle 순서 추세 |
|---|---|---|---|---|
| WaveMean | 사용 | 제외 | 제외 | 제외 |
| WaveMetaMean | 사용 | 사용 | 제외 | 제외 |
| WaveMetaContextMean | 사용 | 사용 | 사용 | 제외 |
| WaveMetaContextTrend | 사용 | 사용 | 사용 | 사용 |

모든 설정은 동일한 파라미터 텐서와 seed별 동일 초기값을 가진다. 비활성 입력은 0으로 대체하므로 활성화된 정보와 유효 자유도에는 차이가 있다.

- cycle encoder: `[512 신호 + 2 metadata] → 64 → 32`, ReLU.
- 집계: embedding 평균 32개 + 순서 추세 32개 + Context 15개.
- 순서 추세: `sum(z_i * p_i) / sum(p_i²)`, `p=linspace(-1, 1, L)`.
- SOH head: `79 → 32 → 1`, ReLU.
- 추세는 선형적인 순서 정보만 표현한다. GRU 등의 일반적인 시계열 모델을 평가한 것은 아니다.
- 기존 Fixed 결과는 별도 HistoricalFixed로 표시. 새 WaveMean은 구조를 맞춘 대조군이므로 기존 Fixed 모델과 수치적으로 동일하지 않다.

각 길이에서 입력 추가 전후 오차를 비교하고, 각 모델에서 LB20/50과 LB10을 비교한다. 사후 Oracle은 샘플별 실제 SOH를 사용하여 세 길이의 최소 오차를 선택한 진단용 결과이며 배포 가능한 성능이 아니다.

## 실행

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-repro/bin/python scripts/test_context_history.py
PYTHONDONTWRITEBYTECODE=1 .venv-repro/bin/python -u scripts/train_context_history.py
PYTHONDONTWRITEBYTECODE=1 .venv-repro/bin/python scripts/verify_context_history.py
# Matplotlib가 설치된 별도 Python으로:
python scripts/plot_context_history.py
```

동일 프로토콜에서 완료된 모델은 해시를 검증하여 재사용한다. 설정이나 학습 코드가 바뀌면 다른 출력 디렉터리를 지정해야 한다.

## 산출물

`outputs/context_history_v1/`: protocol, scalers, 모델, 학습 이력, 예측값, 지표, 입력 추가/길이 비교, 검증 기록, 보고서와 그림. **모두 로컬 전용이며 GitHub에는 코드와 실행 문서만 게시한다.**
