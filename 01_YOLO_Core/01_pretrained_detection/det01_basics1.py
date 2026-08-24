# 기본 검출 코드: 사진 파일
from ultralytics import YOLO

# 1. COCO 사전 학습 모델: 기본 제공
model = YOLO("yolo11n.pt")

# 2. Detect
results = model.predict(
    source=r"01_YOLO_Core\01_pretrained_detection\dataset\1.jpg",
    conf=0.5, # 50% 이상 확실해야 바운딩 박스 치고 결과 반환.
    save=True,  # 예즉 결과를 저장함. [디폴트 위치] 최상위 폴더 > runs > detect > predict > 1.jpg
    workers=0, # cpu 코어 하나만 사용(멀티 프로세싱 금지)
    project=r"01_YOLO_Core\01_pretrained_detection", # [옵션] 디폴트 위치 말고 결과 생성 위치 지정
    name="detect_01", # [옵션] 결과 폴더명. 디폴트는 predict
    exist_ok=True  #[옵션] 기존 폴더에 predict1, predict2... 추가되지 않고 덮어쓰게 함.
)

# 3. 결과 확인
print(results)