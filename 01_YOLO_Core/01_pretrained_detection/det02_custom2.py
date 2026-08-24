from ultralytics import YOLO
import cv2

# 커스텀 학습 모델
model = YOLO(r"01_YOLO_Core\01_pretrained_detection\predict\weights\best.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 커스텀 모델로 탐지
    results = model.predict(
        frame,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    # Bounding Box 표시
    result_frame = results[0].plot()

    cv2.imshow("Custom YOLO", result_frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()