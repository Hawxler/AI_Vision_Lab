# 커스템 데이터로 학습 모델 만들기
from ultralytics import YOLO

# def main():
data_yaml = r"01_YOLO_Core\xx_segTest_handtools\dataset\data.yaml"

# 1. 사전 학습 모델 불러오기
model = YOLO("yolo11n-seg.pt")

# 2. 내 데이터로 학습
model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    device=0,
    workers=0,  # cpu 코어 하나만 사용(멀티 프로세싱 금지)
    project=r"01_YOLO_Core\xx_segTest_handtools\runs",
    name="handtools_seg",
    exist_ok=True
)

# 멀티 프로세싱으로 인한 런타임 에러 방지를 위해 
# 새 프로세스를 못만들게 하기 위해 "__main__" 하나만 실행
# if __name__=='__main__':
#     main()