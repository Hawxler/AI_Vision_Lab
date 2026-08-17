from ultralytics import YOLO
import pandas as pd

def main():
    # 1. 사전 학습된 YOLOv11 모델 로드
    model = YOLO('yolo11n.pt') # 또는 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'

    # 2. 모델 훈련
    model.train(
        data='./images/test1/data.yaml',  # 데이터셋 설정 파일 경로
        epochs=50,
        imgsz=640,  # 이미지 크기
        batch=16,  # 배치 크기
        # device='0',  # 사용할 GPU 장치 (0은 첫 번째 GPU)
        project='test1_1',  # 결과 저장 폴더
        name='yolo_test1_model',  # 실행 이름
        exist_ok=True,  # 이전 학습 모델 폴더 덮어쓰기 허용. 아니면 매 학습마다 새 폴더 생성됨.
    )

    # 3. 검증 (선택)
    metrics = model.val()
    print(f"검증 결과: {metrics}")
    df = pd.read_csv('test1_1/yolo_test1_model/results.csv')
    print(df)

    # 4. 테스트 (선택 - 모델 성능 확인)
    results = model.predict(
        source='./images/test1/test/images', 
        save=True
    )

    print("훈련 완료! 결과는 'test1_1/yolo_test1_model' 폴더에 저장되었습니다.")

if __name__ == '__main__':
    main()