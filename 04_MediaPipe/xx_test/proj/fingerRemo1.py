import cv2
import math
import time
import mediapipe as mp

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==================================================
# 경로 설정
# ==================================================

BASE_DIR = Path(__file__).resolve().parent 
# Path(__file__).resolve(): 절대 경로
# partent: 현재 파일명 제외

MODEL = BASE_DIR / "hand_landmarker.task"

print("Model path:", MODEL)


# 모델 존재 확인
if not MODEL.exists():
    raise FileNotFoundError(
        f"모델 파일을 찾을 수 없습니다:\n{MODEL}"
    )


# ==================================================
# 거리 설정: 0 ~ 100
# ==================================================

MIN_DIST = 20    # 손가락 붙임 20px까지 인정 = 0 
MAX_DIST = 180   # 손가락 최대 벌림 180px까지 인정 = 100


# ==================================================
# MediaPipe 설정: HandLandmarker 동작 기준 설정
# ==================================================

options = vision.HandLandmarkerOptions(
    # 모델 지정
    base_options=python.BaseOptions(
        model_asset_path=str(MODEL)
    ),
    # Video 모드
    running_mode=vision.RunningMode.VIDEO,
    # 한 손만
    num_hands=1
)

# HandLandmarker 위 옵션대로 생성
landmarker = vision.HandLandmarker.create_from_options(
    options
)


# ==================================================
# 카메라
# ==================================================

cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        break


    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape  # (480, 640, 3)


    # OpenCV BGR → RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # OpenCV image → MediaPipe image 변환
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # 손 검출
    result = landmarker.detect_for_video(
        mp_image,
        int(time.monotonic() * 1000) # 타임스탬프: 시간 단순 증가(monotoic).
        # Video 모드에서는 MediaPipe가 아래처럼 프레임 순서를 알아야 함.
        # Frame 1 → 시간 1000  : 123.456 * 1000 = int 123456 ms
        # Frame 2 → 시간 1033
        # Frame 3 → 시간 1066
    )


    if result.hand_landmarks:

        hand = result.hand_landmarks[0] # 첫 째 손: 여기서는 손 하나뿐임.


        # 엄지 끝 = 4
        thumb = hand[4]

        # 검지 끝 = 8
        index = hand[8]


        # MediaPipe 좌표 → OpenCV 좌표
        # MP 좌표는 픽셀이 아니라 0.0 ~ 1.0 정규화 좌표임
        # 픽셀 좌표로 변환 예: 0.5 * 640 = 320
        x1 = int(thumb.x * w)  
        y1 = int(thumb.y * h)

        x2 = int(index.x * w)
        y2 = int(index.y * h)


        # 두 점 사이 거리
        # √(a²+b²) = √((x2-x1)² + (y2-y1)²)
        distance = math.hypot(
            x2 - x1,
            y2 - y1
        )


        # distance → 0~100
        value = int(
            (distance - MIN_DIST)
            / (MAX_DIST - MIN_DIST)
            * 100
        )

        # (0 ~ (100 또는 value 중 작은 값)) 중 큰 값
        value = max(
            0,
            min(100, value)
        )


        # 엄지
        cv2.circle(
            frame,
            (x1, y1),
            10,
            (0, 0, 255),
            -1
        )


        # 검지
        cv2.circle(
            frame,
            (x2, y2),
            10,
            (0, 0, 255),
            -1
        )


        # 연결선
        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )


        # 실제 거리
        cv2.putText(
            frame,
            f"Distance: {int(distance)}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )


        # 0~100 값
        cv2.putText(
            frame,
            f"Value: {value}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    cv2.imshow(
        "Finger Distance",
        frame
    )


    if cv2.waitKey(1) & 0xff == ord('q'):
        break


cap.release()

landmarker.close()

cv2.destroyAllWindows()