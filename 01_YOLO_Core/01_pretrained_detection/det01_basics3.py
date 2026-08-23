# 웹캠
from ultralytics import YOLO
import cv2

#1. 사전 학습된 YOLOv11 모델 불러오기
model = YOLO("yolo11n.pt")

#2. 노트북 카메라 열기
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # 3. 카메라 한 프레임 읽기
    ret, frame = cap.read()

    if not ret:
        break

    # 4. YOLO Detect
    results = model.predict(
        source=frame,
        conf=0.5,
        imgsz=640,
        verbose=False,
        #############
        # 저장은 다음 장에서 동영상으로.
    )

    # 5. YOLO가 탐지 결과를 그려준 이미지
    result_frame = results[0].plot()

    # 6. 화면 출력
    cv2.imshow("YOLO11", result_frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

################################
# result = results[01]
# print(result)
"""
ultralytics.engine.results.Results object with attributes:

boxes: ultralytics.engine.results.Boxes object
keypoints: None
masks: None
names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: ......}
obb: None
orig_img: array([[[215, 212, 214],
        [215, 212, 214],
        [214, 211, 213],
        ...,
        [ 22,  28,  23],
        [ 27,  31,  26],
        [ 30,  34,  29]],

       [[134, 130, 135],
        [134, 130, 135],
        [134, 130, 135],
        ...,
        [132, 127, 129],
        [133, 128, 130],
        [133, 128, 130]]], dtype=uint8)
orig_shape: (480, 640)
path: 'D:\\AI_Labs\\AI_Vision_Lab\\01_YOLO_Core\\01_pretrained_detection\\dataset\\4.jpg'
probs: None
save_dir: '01_YOLO_Core\\01_pretrained_detection\\predict'
speed: {'preprocess': 1.4169999994919635, 'inference': 33.41989999898942, 'postprocess': 72.97449999896344}  
"""