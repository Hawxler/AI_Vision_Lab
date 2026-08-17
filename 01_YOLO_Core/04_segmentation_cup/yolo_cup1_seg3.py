# 사전 학습 가중치 > 실시간 웹캠 적용
import cv2
import time
import argparse
from ultralytics import YOLO

best_pt_path = "test5_1/yolo_test5_seg/weights/best.pt"

def main(weights=best_pt_path, cam=0, conf=0.5, imgsz=640, device=0, width=1280, height=720):
    # 1. 모델 로드
    model = YOLO(weights)
    
    # 2. 갬쳐 열기
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise RuntimeError(f"{cam}번 카메라 불능")
    
    # # [옵션]. 해상도 설정 및 지연 줄이기(가능한 경우)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # if width: cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    prev_t = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패")
            break
        
        # 3. 예측
        results = model.predict(
            source=frame,
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False
        )
        r = results[0]
        # print(r)
        
        # 4. 주석/마스크/박스/라벨된 프레임
        annotated = r.plot()  # OpenCV(BGR) 이미지 변환
        
        # 5. FPS 표시
        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev_t)) # 1/0.000001 ?
        prev_t = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),
                    2, cv2.LINE_AA)

        cv2.imshow("Seg Webcam", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): # q or ESC
            break
        elif key == ord('s'):
            path = f"frame_{int(time.time())}.jpg"
            cv2.imwrite(path, annotated)
            print(f"저장 위치: {path}")

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # 인자 지정해서 실행해보기
    # 실행 예시: python yolo_cup1_seg3.py --cam 0 --device 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=r"test5_1/yolo_test5_seg/weights/best.pt")
    ap.add_argument("--cam", type=int, default=0, help="웹캠 인덱스 (내장=0, 외장=1 등)")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=0, help="0 또는 'cpu'")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()
    main(**vars(args))

    # # 기존처럼 그냥 실행해보기
    # main()