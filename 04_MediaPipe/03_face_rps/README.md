# MediaPipe 프로젝트 1 — 얼굴 표정 가위바위보

## 1. 목표

웹캠 얼굴을 다음처럼 게임 입력으로 사용한다.

| 얼굴 입력 | 게임 입력 |
|---|---|
| 입 닫기 | ROCK(주먹) |
| 입 열기 | PAPER(보자기) |
| 한쪽 눈 윙크 | SCISSORS(가위) |

전체 흐름은 다음과 같다.

```text
웹캠 → Face Landmarker → jawOpen과 eyeBlinkLeft/Right
      → ROCK/PAPER/SCISSORS → 컴퓨터와 승패 비교
```

## 2. 꼭 알아둘 점

MediaPipe Face Landmarker의 blendshape에서 다음 세 값을 사용한다.

- 입 열기/닫기: `jawOpen`
- 왼쪽/오른쪽 눈 감김: `eyeBlinkLeft`, `eyeBlinkRight`

한쪽 눈 점수는 높고 반대쪽 눈 점수는 낮을 때만 윙크로 판정한다. 따라서 두 눈을
동시에 감는 평범한 눈 깜박임은 가위가 되지 않는다.

## 3. 준비

권장 환경은 Windows와 Python 3.11이다. PowerShell에서 실행한다.

```powershell
cd D:\AI_Labs\AI_Vision_Lab\04_MediaPipe
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

모델을 한 번 내려받는다.

```powershell
cd .\03_face_rps
.\download_model.ps1
```

브라우저에서 직접 다운로드하려면 Google 공식 모델 주소를 사용한다.

<https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task>

직접 받은 파일은 이름을 `face_landmarker.task`로 유지하고 다음 위치에 저장한다.

```text
D:\AI_Labs\AI_Vision_Lab\04_MediaPipe\03_face_rps\models\face_landmarker.task
```

PowerShell 실행 정책 때문에 막히면 다음처럼 실행할 수 있다.

```powershell
powershell -ExecutionPolicy Bypass -File .\download_model.ps1
```

폴더 구조는 다음과 같아진다.

```text
03_face_rps/
├─ face_rps.py
├─ download_model.ps1
├─ README.md
└─ models/
   └─ face_landmarker.task
```

## 4. 실행

가상환경이 활성화된 상태에서:

```powershell
python .\face_rps.py
```

화면 조작:

| 키 | 기능 |
|---|---|
| `Space` | 현재 표정으로 한 판 진행 |
| `W` / `S` | 입 열림 기준을 올림 / 내림 |
| `E` / `D` | 윙크(눈 감김) 기준을 올림 / 내림 |
| `Q` | 종료 |

## 5. 인식 순서

분류 순서가 중요하다.

```python
if max(blink_left, blink_right) > wink_threshold and \
   abs(blink_left - blink_right) > wink_difference_threshold:
    move = "SCISSORS"
elif jaw_open > mouth_threshold:
    move = "PAPER"
else:
    move = "ROCK"
```

윙크하면서 입을 벌릴 수도 있으므로 SCISSORS를 PAPER보다 먼저 검사한다. 최근 7프레임의
최빈값을 사용해 한 프레임의 흔들림이 게임 입력을 바로 바꾸지 않도록 했다.

## 6. 나에게 맞게 보정하기

1. 입을 편하게 닫고 `jawOpen` 값을 본다.
2. 입을 충분히 열고 값을 본다.
3. 두 값의 중간이 되도록 `W/S`로 threshold를 조정한다.
4. 두 눈을 뜬 상태와 한쪽 눈을 감은 상태의 `blink L/R` 값을 비교한다.
5. 윙크가 잘 잡히는 지점으로 `E/D`를 조정한다.

예: 뜬 눈이 0.05이고 감은 눈이 0.85라면 0.55가 좋은 시작점이다. 윙크 시 좌우
점수 차이도 0.35보다 커야 한다. 조명은 정면에서 밝고 일정하게 유지한다.

## 7. 코드에서 배우는 핵심

- `FaceLandmarkerOptions`: 모델과 VIDEO 모드를 설정한다.
- `output_face_blendshapes=True`: `jawOpen` 같은 표정 계수를 요청한다.
- `blendshape_scores()`: MediaPipe 결과를 이름→점수 딕셔너리로 바꾼다.
- `classify()`: 숫자 두 개와 임계값으로 표정에 의미를 붙인다.
- `winner()`: 플레이어와 컴퓨터 입력을 비교한다.

핵심 학습 과정은 다음과 같다.

```text
랜드마크/표정 계수 → 측정값 → 임계값 → 분류 → 게임 동작
```

## 8. 문제 해결

- 모델을 찾을 수 없음: `download_model.ps1`을 먼저 실행한다.
- 카메라를 열 수 없음: 다른 카메라 앱을 닫거나 `CAMERA_INDEX = 1`로 바꾼다.
- 평범한 눈 깜박임이 SCISSORS: 윙크할 때 좌우 점수 차이가 충분히 나도록 한쪽 눈만 감는다.
- 윙크가 인식되지 않음: `D`로 윙크 기준을 내린다.
- 닫은 입이 PAPER로 나옴: `W`로 입 기준을 올린다.
- 열린 입이 ROCK으로 나옴: `S`로 입 기준을 내린다.

## 9. 다음 단계

다음 프로젝트에서는 3초 카운트다운, 점수판, 효과음, Arduino LED/서보 출력을
추가할 수 있다. 여러 사람에게 같은 기준값을 적용하려면 시작 시 자동 보정 단계도
만들어 볼 수 있다.
