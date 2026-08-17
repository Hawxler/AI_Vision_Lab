from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# 현재 py 파일이 있는 폴더
BASE_DIR = Path(__file__).resolve().parent

# 테스트 이미지
IMAGE_PATH = BASE_DIR / "1.jpg"

# 1. YOLO11 pretrained 모델 불러오기
model = YOLO("yolo11n.pt")

# 2. 이미지 객체 탐지
results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.25  # 신뢰도 임계값: 이하는 버림. 보통 0.5
)

# 3. 첫 번째 이미지의 결과 가져오기
result = results[0]

# 4. 탐지 결과 출력
print("탐지된 객체 수:", len(result.boxes))

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    print(
        "class:",
        result.names[class_id],
        "confidence:",
        round(confidence, 3)
    )

# 5. Bounding Box 그려진 이미지 만들기
annotated = result.plot()

# 6. 화면 출력
cv2.imshow("YOLO11 Test", annotated)

cv2.waitKey(0)
cv2.destroyAllWindows()