# NASA 상온 운용 조건 확장 데이터 준비

기존 RW3~RW6에 상온 저전류 편향 RW13~RW16과 고전류 편향 RW17~RW20을 추가합니다.
이 단계는 데이터 확보·품질 점검·전처리·평가 셀 예약까지만 수행합니다.
Predictor/Selector 학습, threshold 선택, 최종 평가 성능 계산은 수행하지 않습니다.

## 원본과 무결성

NASA의 [PCoE Randomized Battery Usage Data Set](https://zenodo.org/records/15277374)을 사용합니다.

| 묶음 | 셀 | 공개 MD5 |
|---|---|---|
| `7. RW_Skewed_Low_Room_Temp_DataSet_2Post.zip` | RW13~RW16 | `14130c74dd2caa3c6fbda83df26eb55d` |
| `5. RW_Skewed_High_Room_Temp_DataSet_2Post.zip` | RW17~RW20 | `9a0fabc687ae3a8d990c9d6f2f0412d4` |

공식 다운로드 주소:

- https://data.nasa.gov/docs/legacy/ames/7.RW_Skewed_Low_Room_Temp_DataSet_2Post.zip
- https://data.nasa.gov/docs/legacy/ames/5.RW_Skewed_High_Room_Temp_DataSet_2Post.zip

아카이브의 MD5·ZIP CRC를 확인한 후 MAT와 원본 Rmd 설명서만 추출합니다.
MAT별 SHA256, 소속 아카이브, 원본 메타데이터를 로컬 데이터 폴더에 기록합니다.
기존 원본 파일이 다른 내용으로 존재하면 덮어쓰지 않고 중단합니다.

## 전처리 기준

`preprocess_main.build_cell`을 재사용하여 `expanded_v1`에 별도 저장합니다.
기존 `main_v1`은 변경하지 않으며, 기존 학습 스크립트도 계속 `main_v1`을 기본값으로 사용합니다.
추가 데이터를 학습하려면 새 데이터 경로와 분할 계획을 명시적으로 적용해야 합니다.

- 완전한 RW 방전 cycle을 128점의 V/I/dV/T로 변환합니다.
- 실제 기준 방전 전류를 적분한 용량을 첫 유효 기준 용량 Q0로 나눠 SOH를 구성합니다.
- 기준 방전 전에 종료된 cycle만 입력합니다. LB10/20/50에 공통으로 유효한 50-cycle 이력이 있어야 합니다.
- 같은 입력 이력의 반복 기준 측정은 첫 유효 측정만 사용합니다.
- 보간 SOH는 보조 자료로만 저장하며 측정 정답 수에 포함하지 않습니다.
- 각 셀의 Q0가 제공되는 설정이며, 새 배터리에 초기 용량 정보가 전혀 없는 설정이 아닙니다.
- 결측 마스크를 보존합니다. 정규화 및 결측 대체 통계는 후속 학습의 해당 학습 셀에서만 추정해야 합니다.

**확장판의 명시적 차이:** 원본 README의 명목 최대 부하는 5A입니다.
확장판은 측정 편차/과도 응답을 고려한 5% 여유를 두어 전류 유효 범위를 모든 셀에서 `[-0.2, 5.25] A`로 설정합니다.
이는 센서 교정으로 확인한 오차 범위가 아니라 문서화된 전처리 허용치이며, 원전류를 자르거나 보정하지 않습니다.
`main_v1`의 기본 상한은 계속 5.0A입니다. 전류 상한 이외의 전처리 기준은 같습니다.
원본 MAT 설명과 README에 전류 상한 표현이 다른 경우 두 설명을 모두 보존하며, README의 부하 확률표와 실제 신호를 근거로 이 허용치를 적용합니다.

온도는 기존 `[-40, 100] C` 범위를 유지합니다. 범위 안의 영하 온도도 상온 자료에서는 센서 이상일 수 있어 별도로 표시합니다.
입력 종료와 기준 측정 사이 120시간 초과 지연 역시 표시합니다. 120시간은 보고용 임계값이며 샘플 제외 기준이 아닙니다.
품질 표시가 있는 샘플은 원인과 처리 방침을 문서화한 뒤 학습·평가 프로토콜을 확정해야 합니다.

## 셀 역할

모델 성능을 확인하기 전에, 추가 묶음마다 숫자 ID가 낮은 두 셀은 개발용, 높은 두 셀은 최종 평가용으로 정합니다.
이미 분석한 기존 네 셀은 모두 개발용입니다. 샘플 수나 예측 성능에 따라 셀을 재선택하지 않습니다.

- 개발: RW3, RW4, RW5, RW6, RW13, RW14, RW17, RW18
- 최종 평가 예약: RW15, RW16, RW19, RW20

최종 평가 셀은 원신호 품질 확인 및 정답 생성만 수행합니다. 학습, 기본 길이 선택, threshold 조정에는 사용하지 않습니다.
개발 셀 내에서도 배터리 단위 검증을 사용해야 합니다. 검증 셀을 제외한 나머지 셀 안에서
OOF 라벨을 다시 생성해야 하며, Selector 학습 라벨을 만든 Predictor가 해당 검증 셀을 학습해서는 안 됩니다.
현재 스크립트는 이 원칙을 `split_plan.json`에 기록하며, 모델 학습용 폴드·scaler는 아직 생성하지 않습니다.

## 실행

프로젝트 루트에서, 두 ZIP의 로컬 경로를 지정합니다. 출력 버전이 이미 있으면 중단합니다.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-repro/bin/python -m scripts.preprocessing.prepare_expanded_data \
  --high-archive 'data/raw/source_archives/5. RW_Skewed_High_Room_Temp_DataSet_2Post.zip' \
  --low-archive 'data/raw/source_archives/7. RW_Skewed_Low_Room_Temp_DataSet_2Post.zip' \
  --version expanded_v1

PYTHONDONTWRITEBYTECODE=1 .venv-repro/bin/python -m scripts.validation.verify_expanded_data \
  --version expanded_v1
```

재생성은 `--version expanded_v2` 등 새로운 이름을 사용합니다. 동일한 원본·규칙이라면 기존 셀의 신호와 정답이 정확히 재현되는지 검증합니다.

## 로컬 산출물

- `data/raw/RW13.mat`~`RW20.mat`: 추가 원본
- `data/raw/source_archives/`, `data/raw/source_docs/`: 체크섬 확인한 ZIP과 원본 설명서
- `data/processed/expanded_v1/`: canonical 신호, Context, 측정 SOH, cell inventory, 품질 표시, 분할 계획
- `DATA_REPORT.md`, `verification.json`: 실제 데이터 규모·주의 항목 및 독립 검증 결과

원본, 전처리 자료, 실제 결과 보고서는 기존 `.gitignore` 규칙으로 Git에서 제외합니다.
공개 저장소에는 코드와 이 실행 문서만 포함합니다.
검증 PASS는 데이터 생성·분할 불변식 확인을 뜻하며, Selector 성능 개선이나 센서 정확성의 입증이 아닙니다.
