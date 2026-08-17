# yolo_pose_excavator1.py을 클래스 파일로 만들어서 활용
# 이미지 분석
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd

class Pose_Angles1():
    def __init__(self, data_yaml: str = "images/test8/data.yaml", base_model: str = "yolo11n-pose.pt"):
        self.data_yaml = data_yaml
        self.base_model = base_model
        self.model = None
        
    # 각도 계산 (3점: a-b-c): 내부함수
    def angle_3pts(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        print(a, b, c)
        v1, v2 = a - b, c - b
        print(v1, v2)
        cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
        return angle

    # 1. 학습 함수
    def train(self, epochs=50, batch=16, imgsz=640, device=0, project="test8_1", name="yolo_pose_excavator"):
        self.model = YOLO(self.base_model)
        self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            verbose=True,
            exist_ok=True
        )
        
    # 2. 검증 함수
    def validate(self, model_path: str):
        model = YOLO(model_path)
        metrics = model.val(
            data=self.data_yaml,
            imgsz=640,
            batch=16,
            device=0,
            verbose=True
        )
        print("\n검증 결과 요약:")
        print(f"mAP@0.5 (bbox): {metrics.box.map50:.3f}")
        print(f"mAP@0.5 (keypoints): {metrics.keypoints.map50:.3f}")
        return metrics
    
    # 3. 추론 + 시각화
    def predict_and_visualize(self, model_path: str, image_path: str):
        model = YOLO(model_path)
        result = model(image_path)[0]
        kpts = result.keypoints.xy[0].cpu().numpy()
        
        # 예시 인덱스(데이터셋 구조에 따라 수정 가능)
        BOOM = (5, 6, 7)
        ARM = (6, 7, 8)
        BUCKET = (7, 8, 9)

        boom_angle = self.angle_3pts(kpts[BOOM[0]], kpts[BOOM[1]], kpts[BOOM[2]])
        arm_angle = self.angle_3pts(kpts[ARM[0]], kpts[ARM[1]], kpts[ARM[2]])
        bucket_angle = self.angle_3pts(kpts[BUCKET[0]], kpts[BUCKET[1]], kpts[BUCKET[2]])
        
        img = cv2.imread(image_path)
        for i, (x, y) in enumerate(kpts):
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(img, str(i), (int(x)+5, int(y)+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(img, f"Boom angle: {boom_angle:.1f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(img, f"Arm angle: {arm_angle:.1f}", (30, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(img, f"Bucket angle: {bucket_angle:.1f}", (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("Excavator Pose", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    # 클래스 인스턴스 생성
    pose = Pose_Angles1(
        data_yaml=r"images\test8\data.yaml",
        base_model="yolo11n-pose.pt"
    )
    
    # 모델 학습
    pose.train()
    
    # 검증
    pose.validate(model_path="test8_1/yolo_pose_excavator/weights/best.pt")
    
    # 추론 + 각도 계산
    pose.predict_and_visualize(
        model_path="test8_1/yolo_pose_excavator/weights/best.pt",
        image_path="images/test8/test/111.jpg"
    )
    
####다른 파일에서 불러 쓸 경우####################
exit()
"""
from yolo_pose_excavator2 import Pose_Angles1

# 클래스 인스턴스 생성
pose = Pose_Angles1(
    data_yaml=r"images\test8\data.yaml",
    base_model="yolo11n-pose.pt"
)

# 모델 학습
pose.train()

# 검증
pose.validate(model_path="test8_1/yolo_pose_excavator/weights/best.pt")

# 추론 + 각도 계산
pose.predict_and_visualize(
    model_path="test8_1/yolo_pose_excavator/weights/best.pt",
    image_path="images/test8/test/111.jpg"

"""