# yolo_3Train1.py에서 훈련한 모델을 사용해 테스트 해봄.
# 테스트 결과를 예측/정답 비교표로 출력하기
from ultralytics import YOLO
import os
import yaml
import glob

def main():
    # 1. 사전 훈련된 YOLOv11 모델 로드
    model = YOLO(r'test1_1\yolo_test1_model\weights\best.pt')  # 훈련된 모델 경로

    # 간단 버전 테스트 예측(코드 구조 강의용)
    """
    # 검증만 실행
    model.val()

    # 테스트(예측)만 실행
    model.predict(
        source='./images/test1/test/images',  # 테스트 이미지 경로
        save=True  # 예측 결과 이미지 저장
    )
    """

    # 2. 예측(테스트)
    results = model.predict(
        source='./images/test1/test/images',  # 테스트 이미지 경로
        save=True,  # 예측 결과 이미지 저장. 여기서는 변수만 리턴하니 False도 됨
        conf=0.25,  # 신뢰도 임계값
        iou=0.45,  # IoU 기본 임계값. 그 이상은 중복으로 간주해서 제거함
    )

    # 3. 클래스 이름 로드
    with open('./images/test1/data.yaml', 'r') as f:
        data_yaml = yaml.safe_load(f)
        class_names = data_yaml['names'] # ['wheel', 'esp32']

    # 4. 라벨(.txt) 로드
    label_dir = './images/test1/test/labels'
    label_map = {} # {파일명: [클래스ID, x_center, y_center, width, height]}

    for path in glob.glob(os.path.join(label_dir, '*.txt')):
        filename = os.path.basename(path).replace('.txt', '')
        with open(path, 'r') as f:
            ids = [int(line.split()[0]) for line in f.readlines()]
            label_map[filename] = ids

    # 5. 예측/실제 비교표 생성
    comparison = []

    for result in results:
        filename = os.path.splitext(os.path.basename(result.path))[0] # 파일명 (확장자 제거) 3.jpg -> 3
        gt_ids = label_map.get(filename, [])  # 실제 라벨 클래스 ID
        pred_ids = result.boxes.cls.int().tolist()  # 예측된 클래스 ID
        
        # 클래스 이름으로 변환
        gt_names = [class_names[i] for i in gt_ids] if gt_ids else ["없음"]
        pred_names = [class_names[i] for i in pred_ids] if pred_ids else ["없음"]
        
        comparison.append({
            '파일명': os.path.basename(result.path),
            '정답 클래스': ', '.join(gt_names),
            '예측 클래스': ', '.join(pred_names)
        })
        
    # 6. 결과 출력
    from pandas import DataFrame
    df = DataFrame(comparison)
    print(df.to_string(index=False))

    # 7. 결과를 CSV 파일로 저장 (선택)
    df.to_csv('test1_1/yolo_test1_comparison.csv', index=False, encoding='utf-8-sig')

    print("테스트 완료! 결과는 'test1_1/yolo_test1_comparison.csv' 파일에 저장되었습니다.")

if __name__ == '__main__':
    main()