# 강화 학습: 커스텀 데이터로 커스텀 모델(best.pt) 생성하기
from ultralytics import YOLO

# yaml 파일의 경로 넣기 -> yaml이 자기 위치에서 데이터 찾아 경로 넣기
data_yaml = r"01_YOLO_Core\01_pretrained_detection\dataset\data.yaml"

def main():
    # 1. Pretrained Model
    model = YOLO("yolo11n.pt")

    # 2. 내 데이터셋으로 추가 학습
    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        device=0,   # 또는 "cpu"

        project=r"01_YOLO_Core\01_pretrained_detection",
        name="predict",
        exist_ok=True
    )

if __name__=='__main__':
    main()