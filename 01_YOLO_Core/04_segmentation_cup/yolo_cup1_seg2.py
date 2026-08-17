# yolo_cup1.py로 학습한 가중치로 웹캠 실시간 세그멘테이션 해보기
# 초간단 한 줄 테스트
from ultralytics import YOLO

YOLO(r"test5_1\yolo_test5_seg\weights\best.pt").predict(
    source=0, 
    show=True, 
    conf=0.5, 
    imgsz=640, 
    device=0
)

