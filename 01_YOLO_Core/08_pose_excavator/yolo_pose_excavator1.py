# 굴삭기 관절 각도 계산
# test 이미지는 구글, 영상은 유투브에서 받음
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd

# 각도 계산 함수 (3점: a-b-c)
def angle_3pts(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    print(a, b, c)
    v1, v2 = a - b, c - b
    print(v1, v2)
    cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
    return angle

# 1. 학습
def train_model():
    model = YOLO("yolo11n-pose.pt")
    model.train(
        data=r"images\test8\data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project="test8_1",
        name="yolo_pose_excavator",
        verbose=True,
        exist_ok=True
    )
    return model

# 2. 검증
def validate_model(model_path="test8_1/yolo_pose_excavator/weights/best.pt"):
    model = YOLO(model_path)
    metrics = model.val()
    print("\n 검증 결과 요약:")
    print(f"mAP@0.5: {metrics.box.map50:.3f}")
    print(f"전체 평가: {metrics.results_dict}")
    df = pd.read_csv("test8_1/yolo_pose_excavator/results.csv")
    print(df.tail())
    return metrics

# 3. 추론 + 각도 시각화
def test_and_visualize(model_path, image_path):
    model = YOLO(model_path)
    result = model(image_path)[0]
    kpts = result.keypoints.xy[0].cpu().numpy()
    
    # 관절 인덱스 예시(데이터셋에 따라 조정)
    BOOM = (5, 6, 7)
    ARM = (6, 7, 8)
    BUCKET = (7, 8, 9)

    boom_angle = angle_3pts(kpts[BOOM[0]], kpts[BOOM[1]],
                            kpts[BOOM[2]])
    arm_angle = angle_3pts(kpts[ARM[0]], kpts[ARM[1]],
                           kpts[ARM[2]])
    bucket_angle = angle_3pts(kpts[BUCKET[0]],
                    kpts[BUCKET[1]], kpts[BUCKET[2]])
    
    img = cv2.imread(image_path)
    for i, (x, y) in enumerate(kpts):
        cv2.circle(img, (int(x), int(y)), 5, (0,255,0), -1)
        cv2.putText(img, str(i), (int(x)+5, int(y)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    (255, 255, 255), 1)
    cv2.putText(img, f"Boom Angle: {boom_angle:.1f}",
                (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 255), 2)
    cv2.putText(img, f"Arm Angle: {arm_angle:.1f}", 
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)
    cv2.putText(img, f"Bucket Angle: {bucket_angle:.1f}",
                (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)
    
    cv2.imshow("Excavator Pose", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# 4. 실행
if __name__ == '__main__':
    train_model()
    
    validate_model(model_path="test8_1/yolo_pose_excavator/weights/best.pt")

    test_and_visualize(
        model_path="test8_1/yolo_pose_excavator/weights/best.pt",
        image_path="images/test8/test/111.jpg"
    )