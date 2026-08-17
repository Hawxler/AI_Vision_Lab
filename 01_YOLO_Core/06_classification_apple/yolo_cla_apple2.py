# 기본 분류 코드(yolo_cla_apple1.py)를 Class 파일로 만듦.
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import torch
import cv2

class AppleClassifier:
    def __init__(self, 
        model_path="test7_1/yolo_test7_cla/weights/best.pt", 
        device="cuda"):
        self.model = YOLO(model_path)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def train_and_validate(self, data_dir="images/test7",
                           epochs=10, imgsz=224, batch=32):
        # 1. 사전 학습된 분류 모델 로드
        model = YOLO("yolo11n-cls.pt")
        # 2. 훈련
        model.train(
            data=data_dir,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project="test7_1",
            name="yolo_test7_cla",
            exist_ok=True,
            # workers=0  # 다른 프로세스 끄고 단일 프로세스로 로딩
        )
        
        metrics = model.val(data=data_dir)
        print(f"Top-1 Accuracy: {metrics.top1:.4f}")
        print(f"Top-5 Accuracy: {metrics.top5:.4f}")

        df = pd.read_csv("test7_1/yolo_test7_cla/results.csv")
        print(df.tail())
    
    def predict_folder(self, folder_path="images/test7/test",
                       batch_size=8):
        test_dir = Path(folder_path)
        image_paths = (
            list(test_dir.rglob("*.jpg")) + 
            list(test_dir.rglob("*.png")) +
            list(test_dir.rglob("*.jpeg"))
        )

        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            results = self.model(batch)
            
            for r in results:
                if r.probs is not None:
                    top1 = r.probs.top1
                    class_name = r.names[top1]
                    confidence = r.probs.max().item()
                    print(f"{r.path} -> {class_name} ({confidence:.2f})")
                else:
                    print(f"{r.path} -> 추론 실패")
    
    def predict_webcam(self, cam_id=0):
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            return
        print("웹캠을 시작합니다. 종료 = 'q'")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 가져올 수 없습니다.")
                break
            
            results = self.model(frame) # 내 모델
            r = results[0]  # 첫번째 이미지의 인식 결과
            if r.probs is not None:
                top1 = r.probs.top1  # 확률이 가장 높은 클래스의 인덱스
                class_name = r.names[top1] 
                confidence = r.probs.top1conf.item()  # top1 클래스의 신뢰도
                label = f"{class_name} ({confidence:.2f})"
                cv2.putText(frame, label, (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "추론 실패", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 2)
            # cv2.putText(img, text, org(시작점), font, fontScale, color, thickness)
            
            cv2.imshow("Apple Classificatiopn", frame)
            
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
    
        cap.release()
        cv2.destroyAllWindows()

#########################################
'''
from yolo_cla_apple2 import AppleClassifier

if __name__ == "__main__":
    clf = AppleClassifier(device="cuda")  # 또는 device="cpu"            
    clf.predict_webcam()
'''
##########################################