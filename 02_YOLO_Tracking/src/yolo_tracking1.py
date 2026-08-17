# 웹캠: CoCo 80개 객체 탐지 및 추적 + ID 부여
# pip install supervision lap cython_bbox
import cv2
from ultralytics import YOLO
import supervision as sv

def main():
    # 1. 모델 불러오기
    model = YOLO("yolo11n.pt")

    # 2. 추적기 초기화
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # 3. 웹캠 열기
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        
        labels = [f"ID {tid}" for tid in detections.tracker_id.tolist()] # 오류 아님. VSCode가 런타임 항목은 현재 인식 못함.
        annotated = box_annotator.annotate(scene=frame.copy(),
                                        detections=detections)
        annotated = label_annotator.annotate(scene=annotated,
                            detections=detections, labels=labels)

        cv2.imshow("Supervision + ByteTrack", annotated)
        
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()