# 동일객체 추적 > 특정 ID 입력받기 > 그 객체만 추적 
# > 그 객체 Seg > 중심좌표, 면적 > 매초 1회 터미널 출력
import cv2
import time
import numpy as np
import keyboard  # pip install keyboard
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# IOU 계산 함수: Intersection Over Union (교집합 영역)
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def main():
    # 1. Seg 모델 로드
    model = YOLO("yolo11n-seg.pt")
    tracker = DeepSort(max_age=100) # 최대 100 프레임까지는 사라져도 ID 유지

    # 2. 초기화
    cap = cv2.VideoCapture(0)
    last_print_time = 0
    target_id = None
    seen_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 3. 좌우 반전
        frame = cv2.flip(frame, 1)
        
        # 4. 객체 탐지
        results = model(frame, imgsz=640, verbose=False)[0]
        
        # 5. DeepSORT용 객체 리스트 생성
        detections = []
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, cls_id))

        # 6. DeepSORT로 트래킹
        tracks = tracker.update_tracks(detections, frame=frame)

        # 7. 현재 프레임에 등장한 ID 모으기
        current_ids = []
        for track in tracks:
            if track.is_confirmed():
                tid = int(track.track_id)
                current_ids.append(tid) # 지금 보이는 것들
                seen_ids.add(tid) # 이제까지 보였던 것들
        
        # 추적 ID가 현재 프레임에 없으면 경고 출력
        if target_id is not None and target_id not in current_ids:
            print(f"[경고] ID {target_id}가 현재 화면에 없습니다.")
        
        # 8. ID 선택 대기 (한 번만)
        if target_id is None:
            cv2.putText(frame, "Press 's' to select target ID", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (255, 255, 0), 2)
            # 객체마다 박스치고 ID 표시
            for track in tracks:
                if not track.is_confirmed():
                    continue
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                cv2.rectangle(frame, (x1, y1), (x2, y2), 
                              (255,255,255), 2)
                cv2.putText(frame, f"ID {track.track_id}", (x1, y1 -10),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (255, 255, 255), 2)
            
            if keyboard.is_pressed("s"):
                print(f"현재 추적 가능한 ID들: {sorted(set(current_ids))}")
                try:
                    target_id = int(input("추적할 객체의 ID를 입력하시오: "))
                except:
                    print("잘못된 입력입니다. 다시 시도하세요.")
                    target_id = None
                    time.sleep(1)
        
        else:
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                if results.masks is None:
                    print("Segmentation 마스크 없음")
                    continue
                
                track_box = np.array(track.to_ltrb()) # 좌상우하
                best_iou, best_mask = 0, None
                for mask_idx, box in enumerate(results.boxes.xyxy):
                    box = box.cpu().numpy()
                    iou = compute_iou(track_box, box)
                    
                    # IOU가 일정 기준 이상일 때만 mask 선잭
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_mask = results.masks.data[mask_idx].cpu().numpy()
                # 마스크 매칭 실패 시
                if best_mask is None:
                    continue
                
                contours, _ = cv2.findContours((best_mask * 255).astype(np.uint8), 
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                
                contour = max(contours, key=cv2.contourArea) # 면적 최고인 것
                M = cv2.moments(contour)  # 윤곽선의 모멘트 값
                if M["m00"] != 0:
                    # center x, center y, 면적
                    cx = int(M["m10"] / M["m00"])  # x 합 / 전체 면적 픽셀 수 
                    cy = int(M["m01"] / M["m00"])  # y 합 / 전체 면적 픽셀 수
                    area = int(cv2.contourArea(contour))

                    now = time.time()
                    if now - last_print_time > 1.0:
                        print(f"[ID {target_id}] 마스크 매칭 (IOU={best_iou:.2f})")
                        print(f"[ID {target_id}] 중심: ({cx}, {cy}), 면적: {area}")
                        last_print_time = now
                    
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID {track.track_id}", 
                                (cx, cy - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 0, 0), 2)
        cv2.imshow("YOLOv11 + DeepSORT + Seg", frame)                            
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()