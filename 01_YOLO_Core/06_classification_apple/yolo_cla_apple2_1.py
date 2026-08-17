# yolo_cla_apple2.py를 불러와 사용함.
from yolo_cla_apple2 import AppleClassifier as ac

if __name__ == "__main__":
    clf = ac(device="cuda")
    clf.predict_webcam()