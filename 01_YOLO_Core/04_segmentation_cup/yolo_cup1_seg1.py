# 내 데이터: 컵 _ 훈련/검증/테스트
from ultralytics import YOLO
import pandas as pd

DATA_YAML = r"01_YOLO_Core\04_segmentation_cup\dataset\data.yaml"
PRETRAIN = "yolo11n-seg.pt"  # 또는 yolo11s/m/L/x-seg.pt

# GPU: device=0 / 여러 GPU: device=[0,1] / CPU: device='cpu'
def main():
    # 1. 사전 학습된 모델 로드
    model = YOLO(PRETRAIN)

    # 2. 모델 훈련
    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,      # 640으로 resize
        batch=-1,       # 자동 배치(메로리 맞춰 조절)
        device=0,       # GPU 없으면 'cpu'
        workers=8,      # DataLoader 병렬
        lr0=1e-3,       # 초기 학습률
        patience=20,    # 초기 종료(개선 없으면)
        project="test5_1",  # 결과 저장 폴더
        name="yolo_test5_seg3"
    )
    
    # 3. 검증
    metrics = model.val(data=DATA_YAML, split='val') # split='val'은 없어도 되지만 validation을 명확히 하는 것도 좋음.
    print(f"검증 결과: {metrics}")
    df = pd.read_csv(r'test5_1\yolo_test5_seg3\results.csv')
    print(df)

    # 4. 테스트 (선택 - 모델 성능 확인)
    results = model.predict(
        source=r"images\test5\test\images",
        save=True
    )
    
    print("훈련 완료! 결과는 'test5_1/yolo_test5_model' 폴더에 저장되었습니다.")

if __name__ == '__main__':
    main()