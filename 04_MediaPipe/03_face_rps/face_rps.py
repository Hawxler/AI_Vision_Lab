"""Face Rock-Paper-Scissors: closed mouth / open mouth / one-eye wink."""

from __future__ import annotations

import random
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp


MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
CAMERA_INDEX = 0

# Start values. Adjust while running: W/S = mouth, E/D = wink.
MOUTH_OPEN_THRESHOLD = 0.48
WINK_CLOSED_THRESHOLD = 0.55
WINK_DIFFERENCE_THRESHOLD = 0.35
SMOOTHING_FRAMES = 7

MOVES = ("ROCK", "PAPER", "SCISSORS")
GESTURE = {"ROCK": "MOUTH CLOSED", "PAPER": "MOUTH OPEN", "SCISSORS": "WINK"}


def blendshape_scores(face_blendshapes) -> dict[str, float]:
    """Convert MediaPipe's category list to a name -> score dictionary."""
    return {item.category_name: float(item.score) for item in face_blendshapes}


def classify(
    jaw_open: float,
    blink_left: float,
    blink_right: float,
    mouth_t: float,
    wink_t: float,
) -> str:
    """Turn blendshape scores into a move; wink has the highest priority."""
    wink_difference = abs(blink_left - blink_right)
    one_eye_closed = max(blink_left, blink_right) > wink_t
    if one_eye_closed and wink_difference > WINK_DIFFERENCE_THRESHOLD:
        return "SCISSORS"
    if jaw_open > mouth_t:
        return "PAPER"
    return "ROCK"


def winner(player: str, computer: str) -> str:
    if player == computer:
        return "DRAW"
    winning_pairs = {("ROCK", "SCISSORS"), ("PAPER", "ROCK"), ("SCISSORS", "PAPER")}
    return "YOU WIN" if (player, computer) in winning_pairs else "COMPUTER WINS"


def draw_text(frame, text, xy, scale=0.7, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run download_model.ps1 first, or place face_landmarker.task there."
        )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
    )

    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        landmarker.close()
        raise RuntimeError("Cannot open webcam. Try CAMERA_INDEX = 1.")

    history: deque[str] = deque(maxlen=SMOOTHING_FRAMES)
    mouth_t = MOUTH_OPEN_THRESHOLD
    wink_t = WINK_CLOSED_THRESHOLD
    last_timestamp = -1
    computer = "-"
    game_result = "Press SPACE to play"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(time.monotonic() * 1000)
            timestamp_ms = max(timestamp_ms, last_timestamp + 1)
            last_timestamp = timestamp_ms
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
            )

            raw_move = "NO FACE"
            jaw_open = 0.0
            blink_left = 0.0
            blink_right = 0.0
            if result.face_landmarks and result.face_blendshapes:
                scores = blendshape_scores(result.face_blendshapes[0])
                jaw_open = scores.get("jawOpen", 0.0)
                blink_left = scores.get("eyeBlinkLeft", 0.0)
                blink_right = scores.get("eyeBlinkRight", 0.0)
                raw_move = classify(jaw_open, blink_left, blink_right, mouth_t, wink_t)
                history.append(raw_move)
            else:
                history.clear()

            player = Counter(history).most_common(1)[0][0] if history else "NO FACE"
            color = {"ROCK": (0, 220, 255), "PAPER": (0, 255, 0), "SCISSORS": (255, 100, 255)}.get(player, (0, 0, 255))

            cv2.rectangle(frame, (12, 12), (600, 220), (25, 25, 25), -1)
            draw_text(frame, f"YOU: {player} ({GESTURE.get(player, '-')})", (28, 48), 0.9, color, 2)
            draw_text(frame, f"jawOpen: {jaw_open:.2f}   threshold: {mouth_t:.2f}", (28, 84))
            draw_text(frame, f"blink L/R: {blink_left:.2f} / {blink_right:.2f}", (28, 116))
            draw_text(frame, f"wink threshold: {wink_t:.2f}   difference: {abs(blink_left-blink_right):.2f}", (28, 148))
            draw_text(frame, f"CPU: {computer}   {game_result}", (28, 180), 0.7, (255, 220, 120))
            draw_text(frame, "SPACE play | W/S mouth | E/D wink | Q quit", (28, 210), 0.53)

            cv2.imshow("Face RPS - MediaPipe", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("w"):
                mouth_t = min(0.95, mouth_t + 0.02)
            elif key == ord("s"):
                mouth_t = max(0.05, mouth_t - 0.02)
            elif key == ord("e"):
                wink_t = min(0.95, wink_t + 0.02)
            elif key == ord("d"):
                wink_t = max(0.05, wink_t - 0.02)
            elif key == 32 and player in MOVES:
                computer = random.choice(MOVES)
                game_result = winner(player, computer)
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
