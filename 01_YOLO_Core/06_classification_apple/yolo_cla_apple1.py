# 분류: 빨간 사과, 파란 사과
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import torch

def trainer_evaluator():
    # 1. 사전 학습된 모델 로드
    model = YOLO("yolo11n-cls.pt")  # 'n': nano
    
    # 2. 학습
    results = model.train(
        data="images/test7", #분류에서는 data.yaml 뺀 폴더 경로만!!
        epochs=10,
        imgsz=224,
        batch=32,  # -1: 메모리 최대치로 자동 조절
        project="test7_1", # 결과 저장 폴더
        name="yolo_test7_cla",
        exist_ok=True
        # workers=0  #멀티 끄고 단일 프로세스로 DataLoader 진행
    )
    
    # 3. 검증
    metrics = model.val(data="images/test7") # 경로 명시 권장
    print(f"Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"Top-5 Accuracy: {metrics.top5:.4f}")

    # 4. 검증 결과 저장 확인
    df = pd.read_csv("test7_1/yolo_test7_cla/results.csv")
    print(df.tail())

def predict_all_test_images():
    # 1. 학습시킨 모델 로드
    mymodel = YOLO("test7_1/yolo_test7_cla/weights/best.pt")
    mymodel.to("cpu") # (CPU 강제 사용: GPU 메모리 문제 회피)

    # 2. 테스트 이미지 경로
    test_dir = Path("images/test7/test")
    image_paths = list(test_dir.rglob("*.jpg")) # rglob: 하위 폴더에 있는 jpg까지

    # 배치 단위 추론
    batch_size = 8
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        myresults = mymodel(batch)

        for r in myresults:
            if r.probs is not None:
                top1 = r.probs.top1
                class_name = r.names[top1]
                confidence = r.probs.max().item()
                print(f"{r.path} -> {class_name} ({confidence:.2f})")
            else:
                print(f"{r.path} -> 추론 실패")

if __name__ == "__main__":
    trainer_evaluator()
    predict_all_test_images()
