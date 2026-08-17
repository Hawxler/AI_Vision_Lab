# cat segmentation
from ultralytics import YOLO
import pandas as pd

cat_best = r"test5_1\yolo_test5_seg\weights\best.pt"
cat_yaml = r"images\test6\data.yaml"
pretrained = "yolo11n-seg.pt"

def main():
    # 1. 사전 학습된 모델 로드
    model = YOLO(pretrained)

    # 2. 모델 훈련
    model.train(
        data=cat_yaml,
        epochs=50,
        imgsz=640,
        batch=-1,
        device=0,
        workers=8,
        lr0=1e-3,
        patience=20,
        project="test6_1",
        name="yolo_test6_seg",
        exist_ok=True  # 덮어쓰기
    )
    
    # 3. 검증
    metrics = model.val(split="val")
    print(f'검증 결과: {metrics}')
    df = pd.read_csv('test6_1/yolo_test6_seg/results.csv')
    print(df.tail())

    # 4. 테스트 (선택 - 모델 성능 확인)
    results = model.predict(
        source="./images/test6/test/images",
        save=True
    )
    
    print("훈련 완료! 결과는 'test6_1/yolo_test6_model' 폴더에 저장되었습니다.")

if __name__ == '__main__':
    main()