# 웹캠 + 탐지 영상 저장
from ultralytics import YOLO
import cv2

#1. 사전 학습된 YOLOv11 모델 불러오기
model = YOLO("yolo11n.pt")

#2. 카메라 열고 화면 설정
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)

# 카메라에 따라 FPS를 제대로 반환하지 않는 경우가 있어서
# 0 이하면 30으로 설정
if fps <= 0:
    fps = 30.0
print("화면:", width, "x", height)
print("FPS:", fps)

#3. VideoWriter 설정
# mp4 코덱 설정
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# 저장할 영상 파일
toMp4 = cv2.VideoWriter(
    "yolo_detect1.mp4",
    fourcc,
    fps,
    (width, height)
)

if not toMp4.isOpened():
    print("동영상 저장 실패")
    cap.release()
    exit()

print("탐지 시작")

# 4. 실시간 촬영/녹화
while True:
    ret, frame = cap.read()

    if not ret:
        print("카메라 읽기 실패")
        break

    # 5. YOLO 탐지
    results = model.predict(
        source=frame,
        conf=0.5,
        imgsz=640,
        verbose=False
    )

    # 6. 박스 치기
    result_frame = results[0].plot()

    # 7. 동영상 저장
    toMp4.write(result_frame)

    # 8. 화면 출력
    cv2.imshow("YOLO11", result_frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
toMp4.release()
cv2.destroyAllWindows()