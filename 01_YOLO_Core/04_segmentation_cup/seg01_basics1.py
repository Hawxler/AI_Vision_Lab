from ultralytics import YOLO

# 1. YOLO11 Segmentation 사전학습 모델 불러오기
model = YOLO("yolo11n-seg.pt")

# 2. 이미지에서 세그멘테이션 수행
results = model.predict(
    source=r"images\test2\3.jpg",
    conf=0.5,
    imgsz=640,
    device=0,   # "cpu"
    save=True,
    project=r"runs\segment",
    name="predict1",
    exist_ok=True    
)

print("Done!")