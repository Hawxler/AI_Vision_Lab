# 라벨링(Bounding box 좌표 생성) 및 데이터 준비
# Roboflow, LabelImg, Label-studio, CVAT 등 사용
# 라벨링 방법 정리: https://www.notion.so/labelImg-label-studio-256a5c22d89180b78fcef565bb71df69?source=copy_link

import zipfile

labelled_zip = r"images\test1.v2-test1_1.yolov11.zip"
dest = r"images\test1"

# 1. 라벨링 된 파일 압축 풀기(images\test1.v2-test1_1.yolov11.zip)
def unzipper(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

unzipper(labelled_zip, dest)

