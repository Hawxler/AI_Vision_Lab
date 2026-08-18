# MediaPipe로 구현해보기: CPU 초경량
# pip install mediapipe opencv-python
import cv2, time, numpy as np, math
import mediapipe as mp
mp_pose = mp.solutions.pose

def angle(a,b,c):
    a,b,c = np.array(a),np.array(b),np.array(c)
    v1, v2 = a-b, c-b
    cosang = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-9)
    return float(np.degrees(np.arccos(np.clip(cosang,-1,1))))

cap = cv2.VideoCapture(0)
with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
    t0=0
    while True:
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe는 RGB 사용함.
        res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            # 예: 좌 어깨-팔꿈치-손목 각
            L = (11,13,15)  # MediaPipe index는 다름
            pts = [(lm[i].x*frame.shape[1], lm[i].y*frame.shape[0]) for i in L]
            if time.time()-t0>=1.0:
                t0=time.time()
                print("L-팔꿈치:", angle(*pts))
            mp.solutions.drawing_utils.draw_landmarks(
                frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow("MediaPipe Pose", frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break
cap.release(); cv2.destroyAllWindows()
