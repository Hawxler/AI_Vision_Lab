# 사진: 분석 결과 리턴값 이해하기
from ultralytics import YOLO

# 1. 사전 학습된 모델 불러오기
model = YOLO("yolo11n.pt")

# 2. 이미지 탐지
results = model.predict(
    source=r"01_YOLO_Core\01_pretrained_detection\dataset\4.jpg",
    conf=0.5,
    save=True,
    project=r"01_YOLO_Core\01_pretrained_detection",
    exist_ok=True
)

print("--------------결과 자료--------------")
# results = [사진1 결과, 사진2 결과, ...]
# 사진1에 사람, 말, 트럭이 있다면
# 3. 첫 번째 이미지 결과
result = results[0]
print(result)

# 4. Bounding Box
boxes = result.boxes

print("----------전체 boxes ---------")
print(boxes)

print("----------박스 좌표 ---------")
print(boxes.xyxy) # 각 물체의 [좌상단_x, 좌상단_y, 우하단_x, 우하단_y] 절대 좌표
print(boxes.xywh) # 각 물체의 [중심x, 중심y, 너비, 높이]
print(boxes.xyxyn) # 정규화된 xyxy: 좌상단 x좌표 / x축 픽셀수(w), 좌상단 y좌표 / y축 픽셀수(h)
print(boxes.xywhn) # 정규화된 xywn: 중앙 x좌표 / w, 중앙 y좌표 /h

print("----------Confidence ---------")
print(boxes.conf)

print("---------Class ID ---------")
print(boxes.cls)

print("----------Shape ---------")
print(boxes.xyxy.shape)

print("----------클래스 이름 ---------")
print(model.names)

##########################################
# print(boxes)의 출력 결과
"""
cls: tensor([ 0., 17.,  7.], device='cuda:0')
conf: tensor([0.8726, 0.8726, 0.6157], device='cuda:0')        
data: tensor([[4.2564e+02, 9.5549e+01, 5.3480e+02, 4.1503e+02, 8.7264e-01, 0.0000e+00],
        [2.4628e+02, 2.3182e+02, 6.3087e+02, 4.7999e+02, 8.7257e-01, 1.7000e+01],
        [7.6831e-01, 1.5229e+02, 2.7706e+02, 3.8194e+02, 6.1566e-01, 7.0000e+00]], device='cuda:0')
id: None
is_track: False
orig_shape: (480, 640)
shape: torch.Size([3, 6])
xywh: tensor([[480.2181, 255.2882, 109.1642, 319.4788],        
        [438.5787, 355.9055, 384.5877, 248.1739],
        [138.9166, 267.1134, 276.2966, 229.6508]], device='cuda:0')
xywhn: tensor([[0.7503, 0.5319, 0.1706, 0.6656],
        [0.6853, 0.7415, 0.6009, 0.5170],
        [0.2171, 0.5565, 0.4317, 0.4784]], device='cuda:0')    
xyxy: tensor([[425.6360,  95.5488, 534.8002, 415.0276],        
        [246.2849, 231.8185, 630.8726, 479.9924],
        [  0.7683, 152.2880, 277.0649, 381.9388]], device='cuda:0')
xyxyn: tensor([[0.6651, 0.1991, 0.8356, 0.8646],
        [0.3848, 0.4830, 0.9857, 1.0000],
        [0.0012, 0.3173, 0.4329, 0.7957]], device='cuda:0') 
"""
###########################################