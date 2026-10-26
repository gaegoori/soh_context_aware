# 연결형 SOH 연구 구조도

`soh_connected_framework.drawio`를 draw.io에 끌어 넣으면 박스, 글자, 화살표를 직접 수정할 수 있습니다.

## 그림 읽는 순서

**Battery history → Battery Context → Selector → Last L cycles → Predictor → SOH estimate → Accuracy & cost**

- 가운데의 실선이 주된 예측 흐름입니다.
- 아래쪽 실선은 실제 과거 신호를 선택한 길이의 입력창에 공급합니다. Predictor는 Context만으로 SOH를 예측하는 모델이 아닙니다.
- 위쪽 점선은 학습 배터리에서만 수행하는 라벨 생성 경로입니다. 미학습 배터리에 대한 LB10·20·50의 절대오차를 seed별로 구한 뒤 평균하여, 평균 오차가 가장 작은 길이를 Selector의 정답으로 사용합니다.
- 실제 예측 때는 위쪽 라벨 생성 경로가 실행되지 않습니다. 테스트 배터리의 실제 SOH나 최적 Lookback 정답은 Selector 입력이 아닙니다.
- `Predictor f_L`은 Selector가 선택한 길이에 대응하는 Predictor 하나입니다. 그림에서 L은 실제 아래첨자로 표시됩니다.
- 최종 평가에서는 외부 테스트를 제외한 3개 셀로 Predictor들을 재학습합니다. 이 세부 분할은 단순한 구조도를 유지하기 위해 그림 대신 본문에서 설명합니다.
- 총 추론 시간에는 Context 추출, Selector, 선택된 Predictor의 계산을 모두 포함합니다.

Predictor·Selector 학습과 외부 셀 평가까지 구현했습니다. 이 그림은 **검증 대상인 제안 구조**이며 성능 개선이 입증되었다는 의미는 아닙니다. 후속 Context·과거 이력 실험은 별도 프로토콜로 관리합니다.

## 생성

```bash
python3 figures/soh_adaptive_framework_v3/build_connected.py
```

Python 표준 라이브러리로 같은 폴더에 draw.io 편집 원본, SVG와 미리보기 HTML을 생성합니다.
PNG/PDF는 draw.io 등의 렌더러에서 내보낼 수 있습니다.
GitHub에는 생성 코드와 이 설명만 포함하며, 이미지와 편집 원본은 로컬에 보관합니다.

## 파일

- `.drawio`: 편집 원본
- `.png`: 발표·문서용 이미지
- `.pdf`, `.svg`: 확대 가능한 벡터 그림
- `build_connected.py`, `drawing_utils.py`: 편집 원본 및 SVG 생성 코드 (Python 표준 라이브러리만 사용)

## 영문 캡션 초안

**Proposed context-aware adaptive temporal selection framework.** Battery context guides a selector to choose a lookback of 10, 20 or 50 cycles. The corresponding history window is passed to its selected SOH predictor. Dashed arrows denote training-only supervision derived from seed-averaged out-of-fold prediction errors. Final evaluation on unseen battery cells considers SOH accuracy and total inference cost.
