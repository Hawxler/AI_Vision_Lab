import cv2
import mediapipe as mp

def main():
    # 초기화(hands, drawing_utils는 런타임에 로드되는 동적 속성이라 코드 작성 시에는 인식이 안 됨. 실행해야 로드되어 인식됨.)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(  # mediapipe 내부 키워드 인자이므로 인자명 변경 불허!
        static_image_mode=False, # 실시간 영상이므로 False
        max_num_hands=2,         # 최대 감지 손 개수
        min_detection_confidence=0.5, # 최하 탐지 신뢰도 0.5 이상
        min_tracking_confidence=0.5   # 최하 추적 신뢰도 0.5 이상
    )

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 이미지 전처리
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # 관절 그리기
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
                
                # 각 관절 좌표 출력 예시: 손가락 관절 번호(0~20)
                for idx, lm in enumerate(hand_landmarks.landmark):
                    h, w = frame.shape[:2]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.putText(frame, str(idx), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 0, 0), 1            
                    )
                    
        cv2.imshow("Hand Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()