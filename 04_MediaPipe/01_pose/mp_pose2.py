# 왼팔의 팔꿈치 각도 계산
import cv2
import time   # 1초마다 각도 출력
import numpy as np
import mediapipe as mp

# Pose 모듈 짧은 이름으로 준비
mp_pose = mp.solutions.pose

# 각도 계산: b의 각도 계산(왼팔: 11, 13, 15)
def angle(a,b,c):
    a, b, c = np.array(a), np.array(b), np.array(c) # np.array([200,150])
    # 두 개 벡터 생성
    v1, v2 = a-b, c-b
    # 벡터의 각도 공식
    cosang = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-9)
    # np.clip(): 코사인 값을 -1 <= cos(theta) <= 1로 제한
    # np.arccos(): "코사인 값 -> 각도"를 구함.(라디안)
    return float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))

cap = cv2.VideoCapture(0)
# Pose 분석기 객체 생성: 
# False: 매 프레임에서 사람탐지>Landmark 검출>추적을 할까? 
# 0: 가벼움/빠름, 1: 보통, 2: 복잡/무거움
with mp_pose.Pose(static_image_mode=False, model_complexity=1)as pose:
    t0=0  # 마지막 각도 출력 시간
    while True:
        ok, frame = cap.read()
        # frame = cv2.flip(frame, 1) # 안 하면 각도 인식 좋아짐.
        if not ok: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 자세 추정 수행
        res = pose.process(rgb)
        if res.pose_landmarks:  # 사람의 포즈가 검출되었다면,
            lm = res.pose_landmarks.landmark # landmark 리스트: 
            # [x,y,z,visibility]: lm[13].x, lm[13].visibility
            # 좌 어깨-팔꿈치-손목 각도
            L = (11, 13, 15)
            pts = [(lm[i].x*frame.shape[1], lm[i].y*frame.shape[0]) for i in L] # 각 관절의 높이, 너비
            if time.time()-t0 >= 1.0: # 1초마다=30프레임마다 출력
                t0 = time.time()
                print("L-팔꿈치:", angle(*pts)) # points 리스트 언패킹: == angle(pts[0], pts[1], pts[2])
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
        cv2.imshow("MP Pose", frame)
        if cv2.waitKey(1) & 0xff == ord('q'): break
cap.release(); cv2.destroyAllWindows()