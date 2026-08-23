import cv2
import math
import time
import threading
import serial
import mediapipe as mp

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==================================================
# 설정
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL = BASE_DIR / "hand_landmarker.task"

PORT = "COM4"
BAUD = 115200

MIN_DIST = 20
MAX_DIST = 180


# ==================================================
# 모델 확인
# ==================================================

if not MODEL.exists():
    raise FileNotFoundError(
        f"모델 파일을 찾을 수 없습니다:\n{MODEL}"
    )


# ==================================================
# Thread 공유 변수
# ==================================================

value = 0

lock = threading.Lock()
stop_event = threading.Event()


# ==================================================
# Arduino Serial Thread
# ==================================================

def send_arduino(ser):

    while not stop_event.is_set():

        with lock:
            send_value = value

        # 예: "57\n"
        ser.write(
            f"{send_value}\n".encode()
        )

        # 초당 약 20회 전송
        time.sleep(0.05)


# ==================================================
# MediaPipe 설정
# ==================================================

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(MODEL)
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

landmarker = vision.HandLandmarker.create_from_options(
    options
)


# ==================================================
# Camera
# ==================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    landmarker.close()

    raise RuntimeError(
        "카메라를 열 수 없습니다."
    )


# ==================================================
# Arduino Serial 연결
# ==================================================

try:

    ser = serial.Serial(
        PORT,
        BAUD,
        timeout=1
    )

except serial.SerialException as e:

    cap.release()
    landmarker.close()

    raise RuntimeError(
        f"{PORT} 포트를 열 수 없습니다.\n"
        f"Arduino Serial Monitor가 열려 있는지 확인하세요.\n\n"
        f"{e}"
    )


# Arduino reset 대기
time.sleep(2)


# ==================================================
# Serial Thread 시작
# ==================================================

thread = threading.Thread(
    target=send_arduino,
    args=(ser,),
    daemon=True
)

thread.start()


# ==================================================
# MediaPipe Timestamp
# ==================================================

last_timestamp = 0


# ==================================================
# Main Thread
# ==================================================

try:

    while True:

        # ------------------------------------------
        # 1. Camera
        # ------------------------------------------

        ret, frame = cap.read()

        if not ret:
            break


        # 거울 화면
        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape


        # ------------------------------------------
        # 2. BGR → RGB
        # ------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ------------------------------------------
        # 3. MediaPipe Image
        # ------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # ------------------------------------------
        # 4. Timestamp
        # ------------------------------------------

        timestamp = (
            time.monotonic_ns()
            // 1_000_000
        )

        if timestamp <= last_timestamp:
            timestamp = last_timestamp + 1

        last_timestamp = timestamp


        # ------------------------------------------
        # 5. Hand detection
        # ------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )


        # 기본값
        current_value = 0


        # ------------------------------------------
        # 6. 손 검출 성공
        # ------------------------------------------

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # Thumb Tip = 4
            thumb = hand[4]

            # Index Tip = 8
            index = hand[8]


            # --------------------------------------
            # 7. 정규화 좌표 → 픽셀 좌표
            # --------------------------------------

            x1 = int(thumb.x * w)
            y1 = int(thumb.y * h)

            x2 = int(index.x * w)
            y2 = int(index.y * h)


            # --------------------------------------
            # 8. 두 손가락 거리
            # --------------------------------------

            distance = math.hypot(
                x2 - x1,
                y2 - y1
            )


            # --------------------------------------
            # 9. Distance → 0~100
            # --------------------------------------

            current_value = int(
                (distance - MIN_DIST)
                / (MAX_DIST - MIN_DIST)
                * 100
            )


            current_value = max(
                0,
                min(100, current_value)
            )


            # --------------------------------------
            # 10. 손가락 표시
            # --------------------------------------

            cv2.circle(
                frame,
                (x1, y1),
                10,
                (0, 0, 255),
                -1
            )

            cv2.circle(
                frame,
                (x2, y2),
                10,
                (0, 0, 255),
                -1
            )

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )


            # Distance 표시
            cv2.putText(
                frame,
                f"Distance: {int(distance)}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )


        # ==================================================
        # Arduino Thread에 값 전달
        # ==================================================

        with lock:
            value = current_value


        # ==================================================
        # 오른쪽 수직 상태 Bar
        # ==================================================

        top = 50
        bottom = h - 50

        left = w - 70
        right = w - 30

        bar_height = bottom - top


        # 0이면 아래
        # 100이면 위
        fill_y = int(
            bottom
            - bar_height
            * current_value / 100
        )


        # Bar 테두리
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (255, 255, 255),
            2
        )


        # 초록색 Bar
        cv2.rectangle(
            frame,
            (left, fill_y),
            (right, bottom),
            (0, 255, 0),
            -1
        )


        # ==================================================
        # 0~100 값 표시
        # ==================================================

        cv2.putText(
            frame,
            f"Value: {current_value}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # ==================================================
        # 화면 출력
        # ==================================================

        cv2.imshow(
            "Finger LED Controller",
            frame
        )


        # q → 종료
        if cv2.waitKey(1) & 0xff == ord('q'):
            break


# ==================================================
# 종료
# ==================================================

finally:

    stop_event.set()

    thread.join()

    # LED OFF 전송
    try:
        ser.write(b"0\n")
        time.sleep(0.1)
    except:
        pass

    ser.close()

    cap.release()

    landmarker.close()

    cv2.destroyAllWindows()