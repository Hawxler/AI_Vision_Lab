import cv2
import time
import numpy as np
import serial  # pip install pyserial
from ultralytics import YOLO

class YoloSegStreamer:
    def __init__(self,
                 model_path,
                 serial_port=None,
                 baudrate=115200,
                 camera_index=0,
                 conf_threshold=0.2,
                 img_size=640):
        
        self.model = YOLO(model_path)
        self.serial_enabled = serial_port is not None
        self.serial = serial.Serial(serial_port, baudrate) if self.serial_enabled else None
        self.cap = cv2.VideoCapture(camera_index)
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        
    def get_frame_with_overlay(self):
        success, frame = self.cap.read()
        if not success:
            return None, None
        
        results = self.model.predict(source=frame,
                                     conf=self.conf_threshold,
                                     imgsz=self.img_size,
                                     verbose=False)
        r = results[0]
        annotated = r.plot()
        
        centers = []
        if r.masks is not None:
            for i, mask in enumerate(r.masks.data):
                mask_np = mask.cpu().numpy().astype(np.uint8)
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours: # findContours로 얻은 윤곽선 리스트
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        area = cv2.contourArea(cnt)

                        # 텍스트 및 시각화
                        cv2.circle(annotated, (cx, cy), 5, (0,0,255), -1)
                        cv2.putText(annotated, f"({cx},{cy})", (cx+10, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 0, 0), 1)
                        cv2.putText(annotated, f"area: {int(area)}", 
                                    (cx + 10, cy + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 0), 1)
                        
                        centers.append((cx, cy, area))
                        
                        # Serial 전송
                        if self.serial_enabled:
                            msg = f"{cx},{cy},{int(area)}\n"
                            self.serial.write(msg.encode())
        
        return annotated, centers
    
    def release(self):
        self.cap.release()
        if self.serial_enabled:
            self.serial.close()
            
    def run_stream(self, show_fps=True):
        prev_t = time.time()
        while True:
            frame, centers = self.get_frame_with_overlay()
            if frame is None:
                print("카메라 프레임 실패")
                break
            
            if show_fps:
                now = time.time()
                fps = 1.0 / max(1e-6, now - prev_t)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 0), 2)
            
            cv2.imshow("YoloSegStreamer", frame)
            if cv2.waitKey(1) & 0xff in (ord('q'), 27):
                break
            
        self.release()
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    # 예시 실행: 직렬포트 없니 테스트
    streamer = YoloSegStreamer(
        model_path=r'test6_1\yolo_test6_seg\weights\best.pt',
        serial_port='COM4',
        baudrate=115200,
        camera_index=0
    )
    streamer.run_stream