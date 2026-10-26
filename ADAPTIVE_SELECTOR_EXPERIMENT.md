# Adaptive Selector: 외부 배터리 평가 실행 규약

기존 [OOF 실험](LOOKBACK_EXPERIMENT.md)의 Context–Lookback 자료로 Selector를 학습하고,
외부 테스트를 제외한 3개 셀로 Predictor를 재학습한다. 설정을 먼저 저장하고 모든 학습을
완료한 뒤 외부 테스트를 평가한다. 이 문서는 실행 방법이며 결과 수치는 로컬 출력에 저장한다.

## 데이터와 학습

- NASA RW3/RW4/RW5/RW6 중 한 셀을 테스트로 제외하는 4개 외부 분할.
- 분할마다 학습 셀 3개, 테스트 셀 1개. 같은 62개 측정 SOH 샘플을 모든 방법에 사용한다.
- Selector 입력은 추론 시점에 계산 가능한 Context 15개만 사용한다.
  실제 SOH, 배터리 ID, 열화 단계, Predictor 오차는 입력에 넣지 않는다.
- Selector 정답은 해당 외부 분할의 기존 OOF 평균 절대오차가 최소인 LB10/20/50이다.
  외부 분할 사이의 학습 라벨을 합치지 않는다.
- Context의 결측 대체 평균과 정규화 통계는 학습 샘플에서만 계산한다.
  Predictor 신호 정규화는 해당 final 분할의 학습 셀 통계를 사용한다.

| 구성 | 사전 고정 설정 |
|---|---|
| Selector | 15입력·3출력 softmax 선형 분류, 평균 CE + 0.5×0.1×가중치 제곱합 |
| 최적화 | zero initialization, SciPy L-BFGS-B, maxiter=500, ftol=1e-12, gtol=1e-8 |
| Context-only | 선형 ridge SOH 회귀, 평균 MSE + 0.1×가중치 제곱합 |
| Predictor | 기존 CyclePredictor, 3개 학습 셀, Adam lr=0.001, 200 epochs, full batch |
| Predictor seed | 42, 0, 1, 2, 3; 길이별 동일 초기 가중치 |
| 평가 | 모든 학습 완료 후 외부 테스트; 설정 탐색·early stopping 없음 |

Selector와 ridge의 절편에는 규제를 적용하지 않는다. 표본 가중치는 균등하다.
Predictor는 4분할 × 3길이 × 5seed = 60개, Selector와 ridge는 각각 4개다.
seed별 Predictor를 앙상블하지 않고 각각 평가한다.

기존 OOF 라벨을 임의로 다시 나누는 검증은 라벨 생성 Predictor의 학습 의존성을
분리하지 못할 수 있다. 이번 첫 실험에서는 하이퍼파라미터를 고정하고 별도 튜닝을
하지 않는다. 향후 튜닝에는 라벨 생성까지 포함한 더 깊은 배터리 단위 분할이나
독립 셀이 필요하다. [OOF 기반 stacking 안내](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html),
[학습 데이터에만 전처리 적합](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

## 비교와 해석

| 방법 | 의미 |
|---|---|
| Fixed10 / Fixed20 / Fixed50 | 각 고정 길이의 Predictor |
| GlobalBestOOF | 학습 측 OOF의 배터리별 MAE를 균등 평균하여 하나의 길이 선택 |
| LabelMajority | 학습 측에서 가장 흔한 Lookback 라벨을 모든 테스트에 적용 |
| Adaptive | Context로 길이를 선택하고 해당 Predictor 하나만 실행 |
| ContextOnlyRidge | 같은 Context로 SOH 직접 예측 |
| Oracle | 테스트 오차를 보고 사후적으로 고른 참고값; 배포·시간 비교 제외 |

주 지표는 배터리 4개의 MAE를 동일 가중 평균한 값이다. 전체 샘플 MAE/RMSE,
배터리별·단계별 결과, 같은 seed에서의 Adaptive−baseline 차이도 저장한다.
SOH 0.01은 1%p다. seed 표준편차는 학습 초기값에 따른 변화이며 배터리 모집단의 신뢰구간이 아니다.
선택 정확도는 재학습 Predictor의 테스트 seed 평균 오차로 만든 사후 라벨에 대한 진단이다.

## 추론 시간

CPU 1 thread, batch=1, 3회 워밍업 후 방법·샘플 순서를 섞어 20회 측정한다.
시작점은 메모리에 준비된 리샘플링 신호와 누적 cycle 메타데이터다.
Adaptive/ContextOnly는 매 요청마다 Context 15개를 다시 계산한다.
Context 정규화·길이 선택·선택한 구간 추출/정규화·한 Predictor 실행·스칼라 반환을 포함한다.
파일 입출력, 원시 데이터의 cycle 분할/리샘플링, 모델 로딩, 메타데이터 최초 캐시 구축은 제외한다.
평균 L 감소를 속도 개선으로 간주하지 않는다. raw sensor부터의 전체 운영 지연은 별도 평가가 필요하다.

## 실행

저장소 루트에서 NumPy/SciPy/PyTorch가 설치된 환경을 사용한다.

```bash
source .venv-repro/bin/activate
python scripts/verify_lookback_oof.py
python scripts/test_adaptive_selector.py
python scripts/train_adaptive_selector.py --phase fit
python scripts/train_adaptive_selector.py --phase evaluate
python scripts/verify_adaptive_selector.py
```

`protocol.json`은 학습 전에 데이터·코드 해시와 설정을 고정한다. 변경된 설정으로 기존
출력을 덮어쓰지 않으며 새로운 `--output outputs/adaptive_selector_<name>`이 필요하다.
중단된 학습은 완료된 모델을 재사용한다. 완료된 평가를 재실행하면 기존 결과를 보존한다.

기본 출력 `outputs/adaptive_selector_v1/`:

- `REPORT.md`, `summary.csv`: 해석 범위와 전체 결과.
- `predictions.csv`, `selections.csv`: SOH 예측과 선택 확률.
- `metrics_by_battery.csv`, `metrics_by_stage.csv`, `paired_comparisons.csv`: 세부 비교.
- `selection_diagnostics.csv`: 테스트에서 사후 계산한 선택 진단.
- `latency_samples.csv`: 모든 요청의 원시 시간.
- `models/`: Selector/ridge 파라미터, Predictor checkpoint, 학습 샘플 목록/이력.
- `verification.json`: 해시·분할·정규화·저장 모델의 예측·집계 독립 재검증.

라벨 교사는 2개 셀로 학습했고 최종 Predictor는 3개 셀로 학습하므로 최적 길이가
이전되지 않을 수 있다. 독립 셀 4개, RW3 온도 결측, 기준 SOH 측정 지연,
Early 샘플 부족, cycle 순서를 보존하지 않는 mean pooling도 해석에 포함해야 한다.
외부 테스트를 본 이후의 수정은 탐색 연구로 구분하며 새 독립 데이터로 재검증해야 한다.

**코드와 실행 문서만 공개한다. 데이터·라벨·Context·가중치·예측·결과 보고서는 로컬 출력으로 보관한다.**
