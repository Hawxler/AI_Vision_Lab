# 클래스 1이면 led 켜기
# 클래스 2이면 led 끄기 및 각도 정보 a숫자sb숫자sc숫자sd숫자se0s

from keras.models import load_model
import cv2
import numpy as np

import serial
import threading
import time


# ==========================================================
# 1. 설정
# ==========================================================

# Arduino Serial 설정
PORT = "COM4"          # 자신의 Arduino COM 포트로 변경
BAUD = 115200

# Serial 송신 주기
SEND_INTERVAL = 0.05   # 0.05초 = 최대 약 20Hz


# ----------------------------------------------------------
# Class 2일 때 전송할 모터 각도
# ----------------------------------------------------------
angle_a = 90
angle_b = 45
angle_c = 120
angle_d = 30

# LED 밝기
LED_ON = 255
LED_OFF = 0


# ==========================================================
# 2. Teachable Machine 모델
# ==========================================================

np.set_printoptions(suppress=True)

model = load_model(
    r"05_TeachableMachine\keras_model.h5",
    compile=False
)

class_names = open(
    r"05_TeachableMachine\labels.txt",
    "r"
).readlines()


# ==========================================================
# 3. Arduino Serial 연결
# ==========================================================

try:
    ser = serial.Serial(
        PORT,
        BAUD,
        timeout=1
    )

    # Arduino는 Serial 연결 순간 Reset되는 경우가 있으므로 잠깐 대기
    time.sleep(2)

    print(f"Arduino 연결 성공: {PORT}")

except serial.SerialException as e:
    print("Arduino Serial 연결 실패")
    print(e)
    raise SystemExit


# ==========================================================
# 4. Thread에서 공유할 변수
# ==========================================================

latest_packet = None

# latest_packet에 메인 스레드와 Serial 스레드가
# 동시에 접근하지 못하게 보호
packet_lock = threading.Lock()

# Serial Thread 종료용 플래그
stop_event = threading.Event()


# ==========================================================
# 5. Arduino에 보낼 패킷 갱신 함수
# ==========================================================

def update_packet(packet):
    global latest_packet

    with packet_lock:
        latest_packet = packet


# ==========================================================
# 6. Serial 송신 Thread
# ==========================================================

def serial_worker():

    last_sent = None

    while not stop_event.is_set():

        # -----------------------------
        # 최신 패킷 안전하게 복사
        # -----------------------------
        with packet_lock:
            packet = latest_packet

        # -----------------------------
        # 새로운 패킷이 있는 경우 송신
        # -----------------------------
        if packet is not None and packet != last_sent:

            try:
                ser.write(
                    packet.encode("utf-8")
                )

                print(
                    "[Arduino TX]",
                    packet.strip()
                )

                last_sent = packet

            except serial.SerialException as e:
                print("Serial 송신 오류:", e)
                break

        time.sleep(SEND_INTERVAL)


# ==========================================================
# 7. Serial Thread 시작
# ==========================================================

serial_thread = threading.Thread(
    target=serial_worker,
    daemon=True
)

serial_thread.start()


# ==========================================================
# 8. Camera 시작
# ==========================================================

camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)


# ==========================================================
# 9. Main Loop
# ==========================================================

try:

    while True:

        # --------------------------------------------------
        # 카메라 한 프레임 읽기
        # --------------------------------------------------
        ret, image = camera.read()

        if not ret:
            print("카메라 영상을 읽을 수 없음.")
            break


        # --------------------------------------------------
        # 224 x 224
        # --------------------------------------------------
        display_image = cv2.resize(
            image,
            (224, 224),
            interpolation=cv2.INTER_AREA
        )


        # --------------------------------------------------
        # 모델 입력용 이미지
        # --------------------------------------------------
        model_image = np.asarray(
            display_image,
            dtype=np.float32
        ).reshape(
            1, 224, 224, 3
        )


        # --------------------------------------------------
        # -1 ~ +1 정규화
        # --------------------------------------------------
        model_image = (
            model_image / 127.5
        ) - 1


        # ==================================================
        # Teachable Machine 추론
        # ==================================================

        prediction = model.predict(
            model_image,
            verbose=0
        )

        # 예:
        #
        # prediction =
        # [[0.91, 0.09]]
        #
        # prediction[0] =
        # [0.91, 0.09]

        index = np.argmax(
            prediction[0]
        )

        confidence_score = prediction[0][index]

        class_name = (
            class_names[index]
            .strip()
            .split(" ", 1)[1]
        )


        # ==================================================
        # Class에 따라 Arduino 명령 생성
        # ==================================================

        # --------------------------------------------------
        # Class 1
        # index = 0
        #
        # LED ON
        # --------------------------------------------------
        if index == 0:

            packet = f"e{LED_ON}s\n"

            update_packet(packet)


        # --------------------------------------------------
        # Class 2
        # index = 1
        #
        # LED OFF
        # + Motor 각도
        #
        # a90sb45sc120sd30se0s
        # --------------------------------------------------
        elif index == 1:

            packet = (
                f"a{angle_a}s"
                f"b{angle_b}s"
                f"c{angle_c}s"
                f"d{angle_d}s"
                f"e{LED_OFF}s\n"
            )

            update_packet(packet)


        # ==================================================
        # 화면 표시
        # ==================================================

        cv2.putText(
            display_image,
            f"{class_name}: {confidence_score * 100:.1f}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Webcam Image",
            display_image
        )


        # ==================================================
        # 결과 출력
        # ==================================================

        print(
            f"Class: {class_name}, "
            f"Confidence: {confidence_score * 100:.1f}%"
        )


        # ==================================================
        # ESC 종료
        # ==================================================

        keyboard_input = cv2.waitKey(1)

        if keyboard_input == 27:
            break


# ==========================================================
# 10. 프로그램 종료 처리
# ==========================================================

finally:

    # Serial thread 종료 요청
    stop_event.set()

    # Thread가 종료할 때까지 잠깐 기다림
    serial_thread.join(timeout=1)

    # Camera 반환
    camera.release()

    # OpenCV 창 닫기
    cv2.destroyAllWindows()

    # Serial Port 닫기
    if ser.is_open:
        ser.close()

    print("프로그램 종료")