import cv2

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 카메라 영상 포맷 지정
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# 일반적인 해상도로 지정
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

camera.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = camera.read()

    if not ret:
        print("카메라 읽기 실패")
        break

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) == 27:   # ESC
        break

camera.release()
cv2.destroyAllWindows()