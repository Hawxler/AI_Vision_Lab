# seg + webcam
import cv2
from ultralytics import YOLO

# 1. 커스텀 세그먼트 모델 불러오기
model = YOLO(r"01_YOLO_Core\xx_segTest_handtools\runs\handtools_seg\weights\best.pt")

# 2. 웹켐 열기
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 3. 웹캠 영상 처리
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break
    
    # 4. YOLO Segmentation
    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.05, # 0.25가 좋지만 학습 데이터가 많지 않았으므로 확 낮춰야 인식이 됨.
        verbose=False,
    )

    # 5. [옵션] 검출 결과 확인
    result = results[0]
    print("boxes:", len(result.boxes))

    if result.masks is not None:
        print("masks:", len(result.masks.data))
    else:
        print("mask 없음")

    # 6. Segmentation 결과 그리기
    annotated_frame = results[0].plot()

    # 7. 화면 출력
    cv2.imshow("YOLO11 Seg", annotated_frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# 8. 종료
cap.release()
cv2.destroyAllWindows()