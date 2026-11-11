# scripts 파일 안내

파일을 역할별 하위 폴더로 구분합니다. 학습 설정과 데이터·결과물의 위치는 그대로 유지합니다.

| 폴더 | 역할 |
|---|---|
| `preprocessing/` | 데이터 준비 |
| `experiments/` | 실험 실행 |
| `models/` | 모델·특징 계산 |
| `validation/` | 저장 데이터·모델·결과 검증 |
| `tests/` | 코드 동작 검사 |
| `plotting/` | 결과 그래프 생성 |

## 실행 방법

프로젝트 최상위 폴더에서 `python -m`으로 실행합니다. 하위 파일을 직접 실행하는 `python scripts/...py` 형식 대신 아래 형식을 사용합니다.

```sh
.venv-repro/bin/python -m scripts.experiments.train_expanded_baselines --help
.venv-repro/bin/python -m scripts.validation.verify_expanded_baselines --help

# 추가 sklearn 설치 폴더를 사용하는 경우
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.venv-selector-extra \
  .venv-repro/bin/python -m unittest discover -s scripts/tests -v
```

학습 명령의 `--help`를 제외하면 실제 학습이 실행됩니다. 완료된 실험의 출력 폴더를 재사용하지 말고, 각 실행 문서에 따라 새 `--output`을 지정합니다.

## 현재 연구에서 보는 순서

1. `preprocessing/prepare_expanded_data.py` → `prepare_expanded_quality.py`: 자료 준비
2. `experiments/train_expanded_baselines.py`: LB10·20·50 Predictor와 OOF 오차
3. `experiments/train_expanded_selector.py`: 기존 Selector 비교
4. `experiments/train_change_selector.py`: 변화 Context·비선형 모델 비교

개발용 8개 셀에서 실험하며 최종 평가용 4개 셀은 별도로 예약되어 있습니다. 선택 확률로 가중한 SOH 오차를 학습하는 새 실험은 아직 추가하지 않았습니다.

## 데이터 준비

| 파일 | 역할 |
|---|---|
| [battery_data.py](preprocessing/battery_data.py) | 전처리 데이터 로더 |
| [prepare_expanded_data.py](preprocessing/prepare_expanded_data.py) | 추가 배터리 자료 준비 및 셀 역할 분할 |
| [prepare_expanded_quality.py](preprocessing/prepare_expanded_quality.py) | 확장 데이터의 온도·예측 시차 품질 처리 |
| [preprocess_main.py](preprocessing/preprocess_main.py) | Cycle 구성·신호 보간·Context 및 SOH 정답 생성 |

## 실험 실행

| 파일 | 역할 |
|---|---|
| [train_adaptive_selector.py](experiments/train_adaptive_selector.py) | 실험 실행: 최초 분류형 Selector와 외부 셀 평가 |
| [train_change_selector.py](experiments/train_change_selector.py) | 실험 실행: 운용 변화 정보·선형 및 Random Forest 비교 |
| [train_context_history.py](experiments/train_context_history.py) | 실험 실행: Context 변수 제거 및 신경망 입력 보강 |
| [train_error_selector.py](experiments/train_error_selector.py) | 실험 실행: 예상 오차가 가장 작은 길이를 선택 |
| [train_expanded_baselines.py](experiments/train_expanded_baselines.py) | 실험 실행: 개발용 8개 셀의 Predictor 및 OOF 오차 생성 |
| [train_expanded_selector.py](experiments/train_expanded_selector.py) | 실험 실행: 확장 데이터의 분류·오차 예측·threshold 비교 |
| [train_lookback_oof.py](experiments/train_lookback_oof.py) | 실험 실행: 초기 신경망 Predictor 및 OOF 길이 라벨 |
| [train_selector_transfer.py](experiments/train_selector_transfer.py) | 실험 실행: Predictor 재학습에 따른 길이 선택 변화 |
| [train_temporal_context.py](experiments/train_temporal_context.py) | 실험 실행: 길이별 Context Ridge Predictor와 Selector |

## 모델·특징 계산

| 파일 | 역할 |
|---|---|
| [adaptive_models.py](models/adaptive_models.py) | 기본 Context·분류형 Selector·직접 SOH 회귀 |
| [change_selector_models.py](models/change_selector_models.py) | 운용 변화 6개 변수 및 회귀 모델 |
| [context_history_models.py](models/context_history_models.py) | 신경망 입력 보강 구조 |
| [error_selector_models.py](models/error_selector_models.py) | 오차 회귀 Selector |
| [expanded_selector_models.py](models/expanded_selector_models.py) | 분류·오차 예측·threshold 선택 규칙 |
| [temporal_context_models.py](models/temporal_context_models.py) | 길이별 요약 통계 및 Ridge Predictor |

## 저장 데이터·모델·결과 검증

| 파일 | 역할 |
|---|---|
| [verify_adaptive_selector.py](validation/verify_adaptive_selector.py) | 독립 검증: 최초 분류형 Selector와 외부 셀 평가 |
| [verify_change_selector.py](validation/verify_change_selector.py) | 독립 검증: 운용 변화 정보·선형 및 Random Forest 비교 |
| [verify_context_history.py](validation/verify_context_history.py) | 독립 검증: Context 변수 제거 및 신경망 입력 보강 |
| [verify_error_selector.py](validation/verify_error_selector.py) | 독립 검증: 예상 오차가 가장 작은 길이를 선택 |
| [verify_expanded_baselines.py](validation/verify_expanded_baselines.py) | 독립 검증: 개발용 8개 셀의 Predictor 및 OOF 오차 생성 |
| [verify_expanded_data.py](validation/verify_expanded_data.py) | 독립 검증: 확장 자료 출처·정답·셀 역할 점검 |
| [verify_expanded_quality.py](validation/verify_expanded_quality.py) | 독립 검증: 품질 처리와 원신호 비교 |
| [verify_expanded_selector.py](validation/verify_expanded_selector.py) | 독립 검증: 확장 데이터의 분류·오차 예측·threshold 비교 |
| [verify_lookback_oof.py](validation/verify_lookback_oof.py) | 독립 검증: 초기 신경망 Predictor 및 OOF 길이 라벨 |
| [verify_preprocessing.py](validation/verify_preprocessing.py) | 독립 검증: 원본·전처리 데이터·분할 점검 |
| [verify_selector_transfer.py](validation/verify_selector_transfer.py) | 독립 검증: Predictor 재학습에 따른 길이 선택 변화 |
| [verify_temporal_context.py](validation/verify_temporal_context.py) | 독립 검증: 길이별 Context Ridge Predictor와 Selector |

## 코드 동작 검사

| 파일 | 역할 |
|---|---|
| [test_adaptive_selector.py](tests/test_adaptive_selector.py) | 동작 검사: 최초 분류형 Selector와 외부 셀 평가 |
| [test_change_selector.py](tests/test_change_selector.py) | 동작 검사: 운용 변화 정보·선형 및 Random Forest 비교 |
| [test_context_history.py](tests/test_context_history.py) | 동작 검사: Context 변수 제거 및 신경망 입력 보강 |
| [test_error_selector.py](tests/test_error_selector.py) | 동작 검사: 예상 오차가 가장 작은 길이를 선택 |
| [test_expanded_selector.py](tests/test_expanded_selector.py) | 동작 검사: 확장 데이터의 분류·오차 예측·threshold 비교 |
| [test_lookback_oof.py](tests/test_lookback_oof.py) | 동작 검사: 초기 신경망 Predictor 및 OOF 길이 라벨 |
| [test_temporal_context.py](tests/test_temporal_context.py) | 동작 검사: 길이별 Context Ridge Predictor와 Selector |
| [test_script_layout.py](tests/test_script_layout.py) | 폴더 이동과 해시 검증의 무결성 검사 |

## 결과 그래프 생성

| 파일 | 역할 |
|---|---|
| [plot_adaptive_results.py](plotting/plot_adaptive_results.py) | 그래프: 초기 Adaptive 오차·비용·선택 분포 |
| [plot_context_history.py](plotting/plot_context_history.py) | 그래프: Context 변수 제거 및 신경망 입력 보강 |
| [plot_error_selector.py](plotting/plot_error_selector.py) | 그래프: 예상 오차가 가장 작은 길이를 선택 |
| [plot_selector_transfer.py](plotting/plot_selector_transfer.py) | 그래프: Predictor 재학습에 따른 길이 선택 변화 |
| [plot_soh_predictions.py](plotting/plot_soh_predictions.py) | 그래프: 실제·예측 SOH 변화 및 산점도 |
| [plot_temporal_context.py](plotting/plot_temporal_context.py) | 그래프: 길이별 Context Ridge Predictor와 Selector |

## 실행 문서

실험 설계 문서는 저장소 최상위에 유지합니다.
- [ADAPTIVE_SELECTOR_EXPERIMENT.md](../ADAPTIVE_SELECTOR_EXPERIMENT.md)
- [CHANGE_SELECTOR_EXPERIMENT.md](../CHANGE_SELECTOR_EXPERIMENT.md)
- [CONTEXT_HISTORY_EXPERIMENT.md](../CONTEXT_HISTORY_EXPERIMENT.md)
- [DATA_EXPANSION.md](../DATA_EXPANSION.md)
- [ERROR_SELECTOR_EXPERIMENT.md](../ERROR_SELECTOR_EXPERIMENT.md)
- [EXPANDED_BASELINE_EXPERIMENT.md](../EXPANDED_BASELINE_EXPERIMENT.md)
- [EXPANDED_SELECTOR_EXPERIMENT.md](../EXPANDED_SELECTOR_EXPERIMENT.md)
- [LOOKBACK_EXPERIMENT.md](../LOOKBACK_EXPERIMENT.md)
- [SELECTOR_TRANSFER_EXPERIMENT.md](../SELECTOR_TRANSFER_EXPERIMENT.md)
- [TEMPORAL_CONTEXT_REDESIGN.md](../TEMPORAL_CONTEXT_REDESIGN.md)

## 경로와 과거 결과의 추적

- `paths.py`: 프로젝트 루트·이동된 코드 경로 처리 및 코드 해시 검사
- `source_layout.json`: 이전 파일명, 새 경로, 이동 전후 코드 해시 기록
- `__init__.py`: 하위 폴더를 Python 패키지로 사용하기 위한 파일
- 이동 전 원본: `.local_archive/script_layout_20261111/`에 로컬 보관. GitHub 업로드 제외.
- 과거 결과의 프로토콜·모델·예측값·해시는 변경하지 않습니다.
- 과거 코드 해시를 검사할 때 등록된 이동 전후 해시와 원본 백업을 모두 확인합니다. 이후 코드가 다시 바뀌거나 백업이 없으면 이 예외는 허용하지 않습니다. 데이터·결과 파일은 항상 정확한 바이트 일치를 요구합니다.
- 이동 전부터 코드 해시가 달랐던 과거 실험은 이 처리로 통과시키지 않습니다.
- 신규 실행에서는 현재 코드 및 경로 처리 코드의 해시를 기록합니다.

`test_*.py`는 코드 검사이며 최종 평가용 셀 테스트가 아닙니다. 검증 PASS는 구현 검증을 뜻하며 성능 개선을 뜻하지 않습니다.
