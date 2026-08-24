# 카메라를 다른 앱(Teachable Machine)이 잡고 있으면 안 됨!
# 카메라 모드를 Windows에서는 아래와 같이 잡는 것이 좋음.
# camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model(r"05_TeachableMachine\keras_model.h5", compile=False)

# Load the labels
class_names = open(r"05_TeachableMachine\labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    if not ret:
        print("카메라 영상을 읽을 수 없음.")
        break

    # TM이 학습한 이미지가 224x224x3 != 웹캠으로 찍은 이미지는 640x480x3 등등 -> 480x640x3 -> 224x224x3로 맞춤
    # 추론할 때의 입력 크기를 학습할 때의 입력 크기와 동일하게 만들어야 함!! + 가로/세로 -> 세로/가로
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Keras 신경망은 아래 정규화처럼 소수 계산이 필요해서 주로 float32를 사용함.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # TM은 학습 시 전처리에서 -1 ~ +1 정규화를 사용하므로 /127.5 -1을 해줌.
    # 보통의 정규화: 원본 0~255 -> /255 -> 0 ~ 1
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    # [[이미지1의 클라1 확률 0.27, 이미지1의 클라2 확률 0.95],
    #  [이미지2의 클라1 확률 0.83, 이미지2의 클라2 확률 0.12],
    #  [이미지2의 클라1 확률 0.21, 이미지2의 클라2 확률 0.00],  ...]
    index = np.argmax(prediction) # 예측값 중 가장 확률 높은 놈
    # 여러 이미지면 열 비교: np.argmax(prediction, axis=1) 
    # 0.27 < 0.95이므로 1번 이미지는 클라1 = 0, 클라2 = 1
    class_name = class_names[index] # [[Filers Screwdriver]] -> "0 Filers\n"
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="") # "0 Filers\n" -> "Filers"
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
