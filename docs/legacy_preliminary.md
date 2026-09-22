# Fixed-lookback battery SOH preliminary experiment

NASA Randomized Battery Usage Dataset (RW3--RW6)에 대해 동일한 SOH predictor에서
lookback만 10, 20, 50 cycle로 바꾸어 Leave-One-Battery-Out (LOBO) 평가를 수행한다.

## 실행

```bash
source .venv/bin/activate
python experiment.py inspect
python experiment.py run --epochs 30
```

기본 데이터 경로는 다운로드 폴더의
`Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab`이다.
다른 위치라면 `--data-dir PATH`를 지정한다. 빠른 파이프라인 점검은
`python experiment.py run --epochs 2 --max-cycles 300`으로 할 수 있다.

결과는 `outputs/`에 저장된다. `data_structure.csv`, `battery_results.csv`,
`lookback_summary.csv`, `stage_results.csv`, 네 개의 figure PNG, fold별 예측값과
학습 이력이 포함된다. 전처리 cache는 `cache/`에 저장된다.

## 실험 정의

- 한 cycle: `discharge (random walk)` step 한 개, 시간축 128점 보간
- 입력 feature: Voltage, Current, dV, Temperature
- SOH: reference-discharge 적분 capacity / 최초 reference capacity
- cycle별 SOH: reference capacity를 실제 측정 시각 기준으로 선형 보간
- split: 각 fold에서 test battery 1개, validation battery 1개, 나머지 2개 train
- 공정 비교: 모든 lookback의 target은 index 49 이후로 고정
- normalization: 각 fold의 train batteries만으로 산출
- model: cycle-wise shared MLP encoder + mean/std/latest temporal pooling + MLP head
- seed: 42, loss: MSE, optimizer: Adam

`--max-cycles`는 smoke test 전용이며 최종 연구 결과에는 사용하지 않는다.
