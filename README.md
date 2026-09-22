# Battery SOH: Context-aware Lookback Selection

NASA Randomized Battery Usage Dataset (RW3–RW6)의 운용 이력으로 SOH를 예측하고,
LB10·20·50의 미학습 샘플 오차를 비교해 Adaptive Selector용 라벨을 생성하는 연구 코드입니다.

현재 파이프라인은 완전한 RW 방전 cycle 구성, 실제 기준 용량 기반 SOH 정답,
Context 15개 생성, 배터리 단위 분할, Predictor 학습 및 OOF 라벨 생성을 포함합니다.

이번 업데이트는 **코드와 실행 문서만** 포함합니다. 원본·전처리 데이터, Context,
예측값, 선택 라벨, 결과 보고서 및 모델 가중치는 로컬에서 생성하며 업로드하지 않습니다.

## 환경과 데이터 준비

사용 환경은 Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1, PyTorch 2.8.0,
macOS arm64 CPU입니다. 패키지 버전은 `requirements-repro.txt`에 고정했습니다.

```bash
python3 -m venv .venv-repro
.venv-repro/bin/python -m pip install -r requirements-repro.txt
```

NASA Randomized Battery Usage Dataset의
`Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post`에 포함된
`RW3.mat`, `RW4.mat`, `RW5.mat`, `RW6.mat`을 `data/raw/`에 배치합니다.

## 실행

```bash
# Cycle 전처리와 검증
.venv-repro/bin/python scripts/preprocess_main.py
.venv-repro/bin/python scripts/verify_preprocessing.py

# 라벨 규칙 및 학습 재현성 단위 검증
.venv-repro/bin/python scripts/test_lookback_oof.py

# Seed 42로 전체 흐름 실행
.venv-repro/bin/python scripts/train_lookback_oof.py --seeds 42
.venv-repro/bin/python scripts/verify_lookback_oof.py

# 5개 seed로 확장: 같은 설정의 완료된 모델은 재사용
.venv-repro/bin/python scripts/train_lookback_oof.py --seeds 42 0 1 2 3
.venv-repro/bin/python scripts/verify_lookback_oof.py
```

전처리는 `data/processed/main_v1/`, 학습·라벨 생성은
`outputs/context_labels_v1/`에 저장합니다. 두 경로는 Git에서 제외합니다.
코드·데이터·설정이 달라지면 `--output outputs/context_labels_v2`처럼
새 출력 경로를 사용합니다.

상세한 분할, 라벨 규칙, 출력 형식은 [LOOKBACK_EXPERIMENT.md](LOOKBACK_EXPERIMENT.md)를 참조하세요.

## 실험 정의

- 한 cycle: 여러 RW 방전 step과 내부의 짧은 휴지를 합친 완전한 방전 구간.
- 입력: `L × 128 × 4`, 채널은 Voltage / Current / dV / Temperature.
- 주 정답: 실제 reference-discharge 전류 적분 용량 / 첫 유효 기준 용량.
- 분할: 학습 2개·라벨 생성 1개·외부 테스트 1개 셀, 총 12개 내부 fold.
- LB10·20·50은 동일 정답 시점과 동일 샘플을 사용.
- 정규화·결측 대체는 각 fold의 학습 데이터 통계만 사용.
- Predictor: cycle별 shared MLP encoder → 평균 pooling → MLP head.
- LB별 독립 모델, 같은 seed에서 초기 가중치 일치, full-batch MSE / Adam lr=0.001.
- 고정 200 epochs; 라벨 생성 셀로 early stopping이나 checkpoint 선택을 하지 않음.
- 라벨: 길이별 seed 평균 절대오차가 최소인 후보. 정확한 동률은 짧은 길이 선택.
- Context 원본과 결측 마스크를 보존하고, selector 학습 단계에서 별도 정규화.
- 외부 테스트 예측과 Adaptive Selector 학습은 이 스크립트의 범위 밖.

## 코드 구조

| 파일 | 역할 |
|---|---|
| `scripts/preprocess_main.py` | Cycle 구성, 정답·Context 생성, 분할·정규화 |
| `scripts/battery_data.py` | 공통 샘플의 LB10·20·50 입력 로더 |
| `scripts/train_lookback_oof.py` | Predictor 학습, OOF 예측, 라벨 생성 |
| `scripts/verify_preprocessing.py` | 원본·전처리·분할 검증 |
| `scripts/verify_lookback_oof.py` | 모델·예측·라벨·Context 검증 |
| `scripts/test_lookback_oof.py` | 라벨 계산 및 학습 재현성 단위 검증 |
| `scripts/reproduce.py` | 이전 step 단위 캐시를 사용하는 과거 재구성 실험 |

## 이전 예비 실험

루트의 `experiment.py`와 `outputs/` 바로 아래에 이미 게시된 파일은 기존 step 단위
예비 실험의 이력입니다. 이전 설명은 [legacy_preliminary.md](docs/legacy_preliminary.md)에
보존했습니다. 입력 단위·pooling·라벨·학습 조건이 현재 cycle 단위 파이프라인과 다릅니다.

## 해석 범위

같은 기준 측정이 여러 외부 fold에 사용되므로 라벨 행 수와 독립 측정 수를 구분해야 합니다.
Stage는 SOH 기반 사후 분석이며 selector 입력에 사용하지 않습니다. 실제 SOH,
미래 측정에서 보간한 SOH, Battery ID, 선택 라벨 및 예측 오차도 Context 입력에서 제외합니다.
현재 평균 pooling은 cycle 순서를 구별하지 않고 LB별 독립 학습의 영향도 포함합니다.

데이터셋 사용 시 원본 설명에서 요청한 다음 논문을 인용합니다:
Brian Bole, Chetan Kulkarni, and Matthew Daigle,
"Adaptation of an Electrochemistry-based Li-Ion Battery Model to Account for
Deterioration Observed Under Randomized Use", Annual Conference of the
Prognostics and Health Management Society, 2014.
