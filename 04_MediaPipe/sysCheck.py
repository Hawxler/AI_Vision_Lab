import sys
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


print("Python    :", sys.version)
print("NumPy     :", np.__version__)
print("OpenCV    :", cv2.__version__)
print("MediaPipe :", mp.__version__)

print()
print("Tasks API :", vision.HandLandmarker)
print("Solutions :", hasattr(mp, "solutions"))