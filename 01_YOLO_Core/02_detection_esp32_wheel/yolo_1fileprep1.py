# 1. 이미지 준비 (images/train, images/val)
""" 파일 가져와서 압축풀기
1. 내컴 파일 가져오기
1.1 cmd 명령어 사용
mkdir ./폴더명
cp ~/Downloads/image.zip ./폴더명/
1.2 파이썬 코드 사용
# 파일 하나 복사
shutil.copy('~/Downloads/image.zip', './폴더명/')
# 폴더 전체 복사 (덮어쓰기 허용: dirs_exist_ok=True)
shutil.copytree('~/Downloads/폴더명', './폴더명/', dirs_exist_ok=True)

2. 인터넷 파일 가져오기
2.1. wget 명령어 사용
wget -P ./폴더명 https://example.com/image.zip
2.2. curl 명령어 사용
curl -L https://example.com/image.zip -o ./폴더명/image.zip

3. 파일 압축 해제
unzip ./폴더명/image.zip
"""

import shutil
import os
import zipfile

# 1. 내컴 파일 가져오기
def copy_file_to_folder(src, dest):
    os.makedirs(dest, exist_ok=True) # 폴더 없으면 생성
    shutil.copy(src, dest)

# 2. 압축 풀기
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# 수업용 라벨링 파일을 만들었으면 사용
# src1 = r"C:\Users\AI06\Downloads\wheel-samples.zip"
# src2 = r"C:\Users\AI06\Downloads\esp32-samples.zip"

dest = "./images"

# 수업용 라벨링 파일이 있으면 사용
# copy_file_to_folder(src1, dest)
# copy_file_to_folder(src2, dest)

unzip_file('./images/wheel-samples.zip', './images/wheel')
unzip_file('./images/esp32-samples.zip', './images/esp32')

