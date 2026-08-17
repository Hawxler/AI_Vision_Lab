# Seg 이미지를 Flask로 스트리밍하기 
# + 카메라에 비친 고양이 인식하여 segmentation 하고 마스크를 그림
# + 중심좌표 + 면적 계산
from flask import Flask, Response, render_template_string
import cv2
import time
from ultralytics import YOLO
import numpy as np

# 1. 앞서 학습한 best.pt 모델 사용
model = YOLO(r'test6_1\yolo_test6_seg\weights\best.pt')

# 2. Flask 앱 생성
app = Flask(__name__)

# 3. 간단한 HTML 페이지
HTML_PAGE = """
<html>
<head><title>YOLO11 Segmentation Stream</title></head>
<body>
<h2>실시간 YOLOv11 세그멘테이션</h2>
<img src="{{ url_for('video_feed') }}" width="720">
</body>
</html>
"""

# 메인 페이지: 루트("/")에 접근 시 HTML 랜더링해줌.
# <img> 태그로 /video_feed 연결해줌
@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

# 추론 > 프레임을 JPEG로 인코딩 > 실시간으로 브라우저에 전달
def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없음")
    
    prev_t = time.time()
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # YOLO 추론
        results = model.predict(source=frame, conf=0.2, imgsz=640, verbose=False)
        r = results[0]
        annotated = r.plot() #[0](첫번째 이미지)에 plot()하기=마스킹하기
        
        #################추가###########################
        # 세그멘테이션 마스크에서 중심좌표/면적 추출
        if r.masks is not None:
            for i, mask in enumerate(r.masks.data):
                mask_np = mask.cpu().numpy().astype(np.uint8)
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        area = cv2.contourArea(cnt)

                        # 시각화: 중심점, 좌표 텍스트, 면적 텍스트
                        # 중심점
                        cv2.circle(annotated, (cx, cy), 5, (0,0,255), -1)
                        # 좌표 텍스트
                        cv2.putText(annotated, f"({cx},{cy})", (cx + 10, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255,0,0), 1)
                        cv2.putText(annotated, f"area: {int(area)}",
                                    (cx + 10, cy + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 0), 1)
        ############################################################
        # FPS 측정 및 표시
        now = time.time()
        fps = 1.0 / max(1e-6, now - prev_t) # 1초/(한 프레임 처리 소요 시간), 최소 1e-6이라도 넣어서 0으로 나누기 방지.
        prev_t = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        # (10, 30): 텍스트의 좌하단 좌표.
        # 1: 폰트 크기, 2: 두께
        
        # 이미지 -> JPEG 인코딩 -> yield
        _, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' 
               + frame_bytes + b'\r\n')
    cap.release()

# MJPEG 스트리밍 경로(/video_feed)
@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 서버 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
