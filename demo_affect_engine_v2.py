"""
Test Affect Engine V2 on Mac (Local Camera)
AU-First Architecture with Feedback Learning

Keys:
  1-6: Correct detected emotion (trigger feedback learning)
  r: Reset feedback learner
  s: Save AU weights to file
  l: Load AU weights from file
  d: Toggle AU debug display
  q: Quit
"""

import cv2
import numpy as np
import os
from src.vision.face_mesh import FaceMeshDetector
from src.vision.affect_engine_v2 import AffectEngineV2, EMOTION_LABELS, AU_NAMES
from src.vision.au_feedback_learner import AUFeedbackLearner

# Emotion index mapping (6 emotions, disgust removed)
EMOTION_KEY_MAP = {
    ord('1'): 0,  # angry
    ord('2'): 1,  # fear
    ord('3'): 2,  # happy
    ord('4'): 3,  # neutral
    ord('5'): 4,  # sad
    ord('6'): 5,  # surprise
}


def draw_calibration_ui(frame, progress, h, w):
    """Draw calibration progress bar."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, "Please keep a NEUTRAL expression...",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, "This will be your baseline.",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

    bar_x, bar_y, bar_width, bar_height = 20, 100, w - 40, 30
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    fill_width = int(bar_width * progress)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)

    cv2.putText(frame, f"Calibrating: {int(progress * 100)}%",
                (bar_x + 10, bar_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def draw_au_debug(frame, au_raw, au_norm, x, y, max_height=200):
    """Draw AU debug information."""
    bar_width = 60
    bar_height = 8

    for i, (au_name, raw, norm) in enumerate(zip(AU_NAMES, au_raw, au_norm)):
        bar_y = y + i * (bar_height + 2)

        # AU name
        cv2.putText(frame, au_name, (x, bar_y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Raw value bar
        cv2.rectangle(frame, (x + 50, bar_y), (x + 50 + int(bar_width * raw), bar_y + bar_height), (150, 150, 150), -1)

        # Normalized value bar
        cv2.rectangle(frame, (x + 120, bar_y), (x + 120 + int(bar_width * norm), bar_y + bar_height), (0, 255, 255), -1)

    return frame


def main():
    # Initialize
    model_path = "models/face_landmarker.task"
    face_detector = FaceMeshDetector(model_path)
    affect_engine = AffectEngineV2(window_size=10, ema_alpha=0.3, calibration_frames=30)
    feedback_learner = AUFeedbackLearner(num_aus=15)

    # Connect feedback learner to affect engine
    affect_engine.feedback_learner = feedback_learner

    # Weights file path
    weights_dir = "data/au_weights"
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "au_weights.npz")

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print("Starting Affect Engine V2...")
    print("Keys: 1-6=correct emotion, r=reset, s=save, l=load, d=debug, q=quit")

    frame_count = 0
    show_debug = False
    last_prediction = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]

        # Detect landmarks (async)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        face_detector.detect_async(frame, timestamp_ms)
        face_detector.update_fps(1/30)

        result = face_detector.get_result(timeout_ms=100)

        if result is not None and result.face_landmarks:
            landmarks = face_detector.get_landmarks()

            if landmarks is not None:
                # Draw face bounding box
                x_coords = landmarks[:, 0] * w
                y_coords = landmarks[:, 1] * h
                x_min, x_max = int(x_coords.min()), int(x_coords.max())
                y_min, y_max = int(y_coords.min()), int(y_coords.max())
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Get predictions
                prediction = affect_engine.predict(landmarks, frame_size=(w, h))

                if prediction['emotion'] != 'calibrating':
                    emotion = prediction['emotion']
                    confidence = prediction['confidence']
                    probs = prediction['probabilities']
                    au_raw = prediction['au_raw']
                    au_norm = prediction['au_norm']
                    last_prediction = prediction

                    # Draw emotion
                    color = (0, 255, 0) if confidence > 0.3 else (0, 200, 200)
                    cv2.putText(frame, f"{emotion}: {confidence:.2f}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

                    # Draw all 7 emotion probabilities as bars
                    bar_x, bar_y = 10, 110
                    bar_max_width = 200
                    bar_height = 22

                    for i, prob in enumerate(probs):
                        label = EMOTION_LABELS[i]
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_width, bar_y + bar_height), (50, 50, 50), -1)
                        bar_w = int(bar_max_width * prob)
                        color = (0, 255, 0) if prob > 0.1 else (100, 100, 100)
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_height), color, -1)
                        cv2.putText(frame, f"{label}: {prob:.1%}", (bar_x + 5, bar_y + 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        bar_y += 27

                    # Draw AU debug if enabled
                    if show_debug:
                        frame = draw_au_debug(frame, au_raw, au_norm, w - 200, 100)

                    print(f"Frame {frame_count}: {emotion} ({confidence:.2f})")
                else:
                    # Calibration in progress
                    progress = affect_engine.calibration_count / affect_engine.calibration_frames
                    frame = draw_calibration_ui(frame, progress, h, w)

        # Draw FPS
        fps = face_detector.get_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw key hints
        cv2.putText(frame, "1-6:correct  r:reset  s:save  l:load  d:debug  q:quit",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Show frame
        cv2.imshow('Affect Engine V2', frame)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in EMOTION_KEY_MAP:
            # Correct detected emotion
            if last_prediction is not None:
                true_emotion = EMOTION_LABELS[EMOTION_KEY_MAP[key]]
                feedback_learner.receive_feedback(true_emotion, last_prediction['au_raw'])
                print(f"Feedback: corrected to {true_emotion}")
        elif key == ord('r'):
            # Reset learner
            feedback_learner.reset()
            print("Feedback learner reset")
        elif key == ord('s'):
            # Save weights
            feedback_learner.save(weights_path)
            print(f"Weights saved to {weights_path}")
        elif key == ord('l'):
            # Load weights
            if os.path.exists(weights_path):
                feedback_learner.load(weights_path)
                print(f"Weights loaded from {weights_path}")
            else:
                print(f"No weights file found at {weights_path}")
        elif key == ord('d'):
            # Toggle debug
            show_debug = not show_debug
            print(f"Debug display: {'ON' if show_debug else 'OFF'}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    face_detector.close()


if __name__ == '__main__':
    main()