# 클래스 목록을 뽑아서 YOLO용 YAML 자동 생성
# 1. Classification용 Yaml 파일 만들기

from pathlib import Path
import yaml

# (1) 데이터셋 경로(압축을 풀어둔 최상위 디렉터리)
DATA_ROOT = Path("images/test7")

# (2) train 아래 폴더명으로 클래스 추출(val/test도 동일 구조일것!)
class_dirs = [p.name for p in (DATA_ROOT/"train").iterdir() if p.is_dir()]
class_names = sorted(class_dirs)

# (3) yaml 딕셔너리 만들기
data_yaml = {
    "path": str(DATA_ROOT),
    "train": "train",
    "val": "val",
    "test": "test",
    "names": class_names,  # ["짜장", "짬뽕", "볶음밥", ...]
    "nc": len(class_names)
}

# (4) yaml 파일 저장
yaml_path = DATA_ROOT / "data.yaml" 
# "/": 경로 이어주기. pathlib 모듈에서 오버로딩한 객체.

with open(yaml_path, "w", encoding="utf_8") as f:
    yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

print(f"Saved: {yaml_path}")
print("Classes:", class_names)
print("nc:", len(class_names))


##########################################
# 2. 검출/세그용 YAML 파일 만들기

'''
import yaml

# (1) 경로 및 이름 지정
root_path = "images/test1"  # 최상위 디렉터리
train_path = "../train/images"
val_path = "../valid/images"
test_path = "../test/images"
class_names = ['esp32', 'wheel']  # 수동으로 입력(라벨링 때 정한 이름으로 입력)

data_yaml = {
    'train': train_path,
    'val': val_path,
    'test': test_path,
    'nc': len(class_names),
    'names': class_names
}

# 저장
yaml_path = f"{root_path}/data1.yaml"
# yaml_str = yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True)
with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    # f.write(yaml_str)
    
print(f"Saved: {yaml_path}")
print("data_yaml 내용:", data_yaml)
'''