# 테스트 코드: GPU 사용 여부 확인
import torch
print("CUDA 가능:", torch.cuda.is_available())
print("GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


# from ultralytics import YOLO
# import torch

# def main():
#     assert torch.cuda.is_available(), "GPU 사용 불가"

#     model = YOLO('yolo11n.pt').to('cuda')  # GPU로 모델 로드
#     model.train(data=r"images\test1\data.yaml", epochs=10)

# if __name__ == '__main__':
#     main()
