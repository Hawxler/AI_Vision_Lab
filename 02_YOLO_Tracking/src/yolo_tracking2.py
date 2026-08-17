# Tracking + DeepSORT: 
# 동일 객체가 화면 밖으로 나갔다 와도 객체 특징 기억  + ID 유지
# pip install deep-sort-realtime
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def main():
    # 1. YOLO 모델 로드
    model = YOLO("yolo11n.pt")

    # 2. DeepSORT 초기화
    tracker = DeepSort(max_age=60) # 60프레임까지 사라져도 허용. 시간이 길어지면 유령 ID 유지 현상으로 객체가 계속 남아있다고 인식함.

    # 3. 웹캠 열기
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 4. YOLO 객체 탐지
        results = model(frame, imgsz=640, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf.cpu().numpy())
            cls_id = int(box.cls.cpu().numpy())
            w, h = x2 - x1, y2 - y1 # width, height
            detections.append(([x1, y1, w, h], conf, cls_id))

        # 5. DeepSORT로 추적기 업데이트
        tracks = tracker.update_tracks(detections, frame=frame)

        # 6. 트래킹 결과 시각화
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            # B box 좌표(ltrb: left,top,right,bottom)
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(frame, (x1,y1), (x2,y2),
                        (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}",
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        
        # 7. 결과 출력
        cv2.imshow("YOLOv11 + DeepSORT", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    print(__name__)