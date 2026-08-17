# 웹캠 포즈 + (옵션) 트래킹 + 1Hz 각도 출력
import time
import math
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# CoCo 17 Keypoints
KPT = {
    0:"nose",1:"left_eye",2:"right_eye",3:"left_ear",4:"right_ear",
    5:"left_shoulder",6:"right_shoulder",7:"left_elbow",8:"right_elbow",
    9:"left_wrist",10:"right_wrist",11:"left_hip",12:"right_hip",
    13:"left_knee",14:"right_knee",15:"left_ankle",16:"right_ankle"
}

def angle_3pts(a, b, c):
    """점 a-b-c에서 b를 꼭짓점으로 하는 각도(degree)를 계산"""
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    # print(a, b, c)
    v1, v2 = a - b, c - b
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosang = np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

class PoseEstimator:
    def __init__(self,
                 model_path="yolo11n-pose.pt",
                 cam_index=0,
                 imgsz=640,
                 conf=0.5,
                 use_tracker=False, #True면 BOT-SORT/ByteTrack으로 ID 유지
                 tracker_yaml="bytetrack.yaml"): # uralytics 내장 yaml
        self.model = YOLO(model_path)
        self.cam_index = cam_index
        self.imgsz = imgsz
        self.conf = conf
        self.use_tracker = use_tracker
        self.tracker_yaml = tracker_yaml
        self.last_print_t = 0
    
    def _to_np(self, tensor_or_nd):
        if isinstance(tensor_or_nd, torch.Tensor):
            return tensor_or_nd.detach().cpu().numpy()
        return np.asarray(tensor_or_nd)
    
    def run(self):
        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            raise RuntimeError(f"카메라 {self.cam_index}를 열 수 없습니다.")

        print("웹켐 시작(q 종료). CUDA:", torch.cuda.is_available())

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 추론 (트레커 on/off)
            if self.use_tracker:
                results = self.model.track(
                    frame, imgsz=self.imgsz, conf=self.conf,
                    persist=True, tracker=self.tracker_yaml, verbose=False
                )
            else:
                results = self.model(
                    frame, imgsz=self.imgsz, conf=self.conf, verbose=False
                )

            r = results[0]
            annotated = r.plot() # 키포인트/스켈레톤이 그려진 프레임
            
            # 키포이트 처리
            if r.keypoints is not None:
                kxy = self._to_np(r.keypoints.xy)  #(N, K, 2)
                ids = None
                if getattr(r.boxes, "id", None) is not None:
                    ids = self._to_np(r.boxes.id).astype(int) # (N,)

                # 1초에 1번만 각도 출력
                now = time.time()
                if now - self.last_print_t >= 1.0:
                    self.last_print_t = now
                    for i, kpts in enumerate(kxy):
                        # 안전 체크; 필요한 키가 모두 있는가?
                        def ok_idx(idx):
                            x, y = kpts[idx]
                            return (x > 0) and (y > 0) # 둘 다 양수면 True

                        # 좌/우 팔꿈치 각도 (Shoulder-elbow-wrist)
                        L = (5, 7, 9); R = (6, 8, 10)
                        left_angle = right_angle = None # 값 없음. "0=팔 펴짐" 아님.
                        if all(ok_idx(j) for j in L): # 반복값이 모두 참이면 True
                            left_angle = angle_3pts(kpts[L[0]], 
                                                    kpts[L[1]], kpts[L[2]])
                        if all(ok_idx(j) for j in R):
                            right_angle = angle_3pts(kpts[R[0]],
                                                     kpts[R[1]], kpts[R[2]])

                        # 좌/우 무릎 각도(hip-knee-ankle)
                        LK = (11, 13, 15); RK = (12, 14, 16)
                        left_knee = right_knee = None # 값 없음
                        if all(ok_idx(j) for j in LK):
                            left_knee = angle_3pts(kpts[LK[0]],
                                                   kpts[LK[1]], kpts[LK[2]])
                        if all(ok_idx(j) for j in RK):
                            right_knee = angle_3pts(kpts[RK[0]], 
                                                    kpts[RK[1]], kpts[RK[2]])
                        
                        pid = int(ids[i]) if ids is not None else i
                        print(f"[ID {pid}] L-팔꿈치: {left_angle:.1f}° R-팔꿈치: {right_angle:.1f}°"
                              f"L-무릎: {left_knee:.1f}° R-무릎: {right_knee:.1f}°")
        
            cv2.imshow("Pose Estimation", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    app = PoseEstimator(
        model_path="yolo11n-pose.pt", 
        cam_index=0,
        imgsz=640,
        conf=0.5,
        use_tracker=True,              # ID 유지하려면 True
        tracker_yaml="bytetrack.yaml"  # 또는 botsort.yaml
    )
    app.run()