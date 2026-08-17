# yolo_pose_excavator1.py을 클래스 파일로 만들어서 활용
# 동영상 분석
import cv2
import numpy as np
from ultralytics import YOLO

class Pose_Angles2:
    def __init__(self, model_path: str, boom=(5,6,7), 
                 arm=(6,7,8), bucket=(7,8,9), conf=0.12,
                 imgsz=960, device=0):
        self.model = YOLO(model_path)
        self.boom = boom
        self.arm = arm
        self.bucket = bucket
        self.conf, self.imgsz, self.device = conf, imgsz, device
        
    def angle_3pts(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        v1, v2 = a - b, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return None
        cos_ang = np.dot(v1, v2) / (n1 * n2 + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))
    
    def _pick_instance(self, r):
        # keypoints가 없거나 0건이면 None
        if r.keypoints is None or r.keypoints.xy is None or len(r.keypoints.xy) == 0:
            return None
        
        # 여러 개면 가장 큰 박스(면적) 선택
        if r.boxes is not None and getattr(r.boxes, "xywh", None) is not None:
            wh = r.boxes.xywh[:, 2:4].cpu().numpy()
            idx = int(np.argmax(wh[:, 0] * wh[:, 1]))
        else:
            idx = 0
        return idx

    def analyze_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"파일 없음: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = self.model(frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
            
            idx = self._pick_instance(result)
            
            if idx is None:
                cv2.putText(frame, "No detection", (20, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                         (0, 0, 255), 2)
                cv2.imshow("Excavator Video", frame)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
                continue
            
            kpts = result.keypoints.xy[idx].cpu().numpy()
            
            def s(triple):
                ang = self.angle_3pts(kpts[triple[0]], kpts[triple[1]], kpts[triple[2]])
                return ang
            
            boom_angle, arm_angle, bucket_angle = s(self.boom), s(self.arm), s(self.bucket)
            
            # 출력 포멧 잡기
            def fmt(x): return "-" if x is None else f"{x:.1f}°"
            
            vis = result.plot()  # 스켈레톤 오버레이
            frame = vis if vis.shape[:2]==frame.shape[:2] else frame
                        
            # 시각화
            for i, (x, y) in enumerate(kpts):
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (int(x)+4, int(y)-6), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            cv2.putText(frame, f"Boom: {boom_angle:.1f}°", (20, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, f"Arm: {arm_angle:.1f}°", (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,255), 2)
            cv2.putText(frame, f"Bucket: {bucket_angle:.1f}°", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Excavator Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    analyzer = Pose_Angles2(
        model_path="test8_1/yolo_pose_excavator/weights/best.pt"
    )
    
    analyzer.analyze_video("images/test8/test/excavator_1.mp4")
    
    