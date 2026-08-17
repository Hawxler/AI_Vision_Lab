# 한글 번호판 인식할 때는 ["en", "ko"] -> ["ko", "en"] 
# 그래도 잘 안 됨. 나중에 한글 번호판만으로 학습하고 
# 정규표현식으로 숫자+한글만 남기는 패턴을 써야 함. (노션 참조)
# OCR 인식 GPU로 할 때 asyocr.Reader(['ko', 'en'], gpu=True)
# 학습 GPU로 할 때 DEVICE = "0"
import os
import csv
import cv2
import glob
import time
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO
import easyocr

# 1. 설정
DATA_YAML = "./images/test4/data.yaml"
PROJECT = "test4_1"
RUN_NAME = "yolo_test4_model"
EPOCHS = 50
IMGSZ = 640
DEVICE = "0"  # GPU: 0, CPU: -1 또는 "cpu"
CONF_TH = 0.5  # 번호판 탐지 신뢰도 Threshold
IO_DIR = "./images/test4/test/images"  # 테스트할 이미지 폴더
OUT_DIR = f"./{PROJECT}/{RUN_NAME}_inference"  # 결과 저장 폴더
CSV_PATH = os.path.join(OUT_DIR, "recognized.csv") # OCR 결과 저장

# 2. 학습
def train_detector():
    model = YOLO("yolo11n.pt")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        device=DEVICE,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
    )
    # best 가중치 경로 반환
    return f"./{PROJECT}/{RUN_NAME}/weights/best.pt"

# 3. 인식: 탐지 -> 크롭 -> OCR
def load_images(input_path: str) -> List[str]:
    """
    폴더면 jpg/png 전체, 파일이면 그 파일 하나만 처리
    """
    if os.path.isdir(input_path):
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            imgs.extend(glob.glob(os.path.join(input_path, ext)))
        return sorted(imgs)
    else:
        return [input_path]

def preprocess_plate(plate_bgr: np.ndarray) -> np.ndarray:
    """
    OCR 성능 향상을 위한 간단 전처리(Gray scale, 대비/이진화 등),
    필요 시 더 조정해볼 것.
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    # 히스토리 평활화로 대비 향상
    gray = cv2.equalizeHist(gray)
    # 가벼운 Blur 후 Otsu 이진화
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def ocr_text(reader: easyocr.Reader, plate_img: np.ndarray) -> str:
    """
    EasyOCR로 문자 추출, 결과 중 confidence가 높은 조각들을 이어붙임.
    """
    result = reader.readtext(plate_img)
    # result: [[bbox, text, conf], [[x1,y1,x2,y2], "19거9334", 0.87신뢰도]...] 
    # conf 높은 순으로 정렬 후, 짧은 조각 여러 개를 붙여 최종 문자열 구성
    if not result:
        return ""
    
    # bbox, text, conf로 풀어서 받기
    parsed = []
    for (bbox, text, conf) in result:
        parsed.append((bbox, text, conf))
        
    # conf 기준으로 정렬
    parsed.sort(key=lambda x: x[2], reverse=True)
    
    # conf > 0.3인 것만 모아 문자열 합치기
    texts = [text for (_, text, conf) in parsed if conf > 0.3]
    out = "".join(texts).replace(" ", "").strip()
    return out

def recognize_from_images(weights_path: str, input_path: str, out_dir: str):
    """
    학습된 탐지 모델로 번호판 탐지 -> OCR -> 결과 저장(이미지, CSV)
    """
    os.makedirs(out_dir, exist_ok=True)
    model = YOLO(weights_path)

    # EASYOCR Reader (한/영/숫자 가능, 상황에 따라 ['en']만도 충분, 한글 번호판이면 'ko', 'en' 순으로)
    reader = easyocr.Reader(['en', 'ko'], gpu=True) # gpu 쓰려면 True 

    images = load_images(input_path)
    
    rows: List[Tuple[str, str, float]] = [] # (filename, text, conf_max)

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] 이미지를 읽지 못했습니다: {img_path}")
            continue
        
        # YOLO 추론
        results = model.predict(img, imgsz=IMGSZ, conf=CONF_TH, verbose=False)
        
        recog_texts = []
        max_conf = 0.0
        
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            # 각 박스별로 번호판 영역 크롭 후 OCR
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            # 가장 높은 conf를 기억(리포팅용)
            if len(confs) > 0:
                max_conf = float(np.max(confs))
            
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img.shape[1] - 1, x2); y2 = min(img.shape[0] - 1, y2)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                proc = preprocess_plate(crop)
                text = ocr_text(reader, proc)
                if text:
                    recog_texts.append(text)
                    
                # 결과 이미지(번호판 박스만 저장할 경우 아래 주석 해제)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)

        # 결과 텍스트 합치기(번호판 여러 개면 쉼표로)
        final_text = ",".join(recog_texts) if recog_texts else ""
        rows.append((os.path.basename(img_path), final_text, max_conf))

        # 원본 이미지를 결과 폴더에 그대로 복사 저장(원하면 OCR 텍스트를 파일명에 넘)
        out_path = os.path.join(out_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, img)

    # CSV 저장
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["filename", "recognized_text", "max_det_conf"])
        wr.writerows(rows)
    
    print(f"[완료] 결과 이미지: {out_dir}")
    print(f"[완료] 인식 결과 CSV: {CSV_PATH}")

# 엔트리 포인트
def main():
    t0 = time.time()
    
    # a. 학습
    print("[1/2] YOLO 번호판 탐지 모델 학습 시작")
    weights = train_detector()
    print(f"[학습완료] best weights: {weights}")

    # b. 인식
    print("[2/2] 테스트 이미지에서 번호 인식 시작")
    recognize_from_images(weights, IO_DIR, OUT_DIR)
    
    print(f"[끝] 전체 소요: {time.time() - t0:.1f}s")

if __name__ == '__main__':
    main()