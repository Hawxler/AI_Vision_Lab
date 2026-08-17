import ultralytics
from ultralytics import YOLO

###################################
# 파일 준비
###################################
"""
import shutil
import os
import zipfile

# 1. 내컴 파일 가져오기
def copy_file(src, dest):
    os.makedirs(dest, exist_ok=True)  # 폴더 없으면 생성
    shutil.copy(src, dest)

# 2. 압축 풀기
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# 수업용 파일을 만들었으면 사용
src = 'C:/Users/AI06/Downloads/coco_dataset1.zip'
dest = './images'

# 수업용 파일이 있으면 사용
copy_file(src, dest)

unzip_file('./images/coco_dataset1.zip', './images/test2')

"""
###################################### 
# YOLOv11 모델 훈련 스크립트 
######################################
# 1. YOLOv11 모델 로드
model = YOLO('yolo11n.pt') # 또는 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'

# print(len(model.names))  # 클래스 개수 확인
# print(model.names)  # 클래스 이름 출력

"""
# 2. 모델 훈련
model.train()

# 3. 검증 (선택)
model.val()
"""

# 4. 예측 (선택)
results = model.predict(source='images/test2', save=True, save_txt=True)

