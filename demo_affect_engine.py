"""
Test Affect Engine on Mac (Local Camera)
Phase 3: Real-time testing
"""

import cv2
import numpy as np
from src.vision.face_mesh import FaceMeshDetector
from src.vision.affect_engine import AffectEngine, EMOTION_LABELS


def draw_calibration_ui(frame, affect_engine, h, w):
    """Draw calibration prompt and progress bar on the frame."""
    # Get calibration progress
    current, total, progress = affect_engine.get_calibration_progress()

    # Dark overlay for text readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Prompt text
    cv2.putText(
        frame,
        "Please keep a NEUTRAL expression...",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 255), 2
    )

    cv2.putText(
        frame,
        "This will be used as your baseline.",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (200, 200, 200), 1
    )

    # Progress bar background
    bar_x = 20
    bar_y = 100
    bar_width = w - 40
    bar_height = 30

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

    # Progress bar fill
    fill_width = int(bar_width * progress)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)

    # Progress text
    cv2.putText(
        frame,
        f"Calibrating: {current}/{total} frames",
        (bar_x + 10, bar_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2
    )

    return frame


def draw_ready_ui(frame, h, w, countdown=60):
    """Draw 'Ready!' message briefly after calibration."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(
        frame,
        "Ready! Calibration complete.",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0), 2
    )

    return frame


def main():
    # Initialize
    model_path = "models/face_landmarker.task"
    face_detector = FaceMeshDetector(model_path)
    affect_engine = AffectEngine(
        window_size=10,
        calibration_frames=180,
        ema_alpha=0.3,
        fusion_weights=(0.6, 0.4)
    )

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print("Starting camera feed...")
    print("Press 'q' to quit")

    # Create controls window with trackbars
    cv2.namedWindow('Controls')

    for i, label in enumerate(EMOTION_LABELS):
        cv2.createTrackbar(label, 'Controls', 100, 200, lambda x: None)

    frame_count = 0
    ready_countdown = 0  # For "Ready!" message after calibration

    while True:
        # Get frame
        ret, frame = cap.read()
        if not ret:
            continue

        # Detect landmarks (async)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        face_detector.detect_async(frame, timestamp_ms)
        face_detector.update_fps(1/30)  # Update FPS

        # Get result
        result = face_detector.get_result(timeout_ms=100)

        if result is not None and result.face_landmarks:
            landmarks = face_detector.get_landmarks()

            if landmarks is not None:
                # Get frame dimensions
                h, w = frame.shape[:2]

                # Draw face bounding box
                x_coords = landmarks[:, 0] * w
                y_coords = landmarks[:, 1] * h
                x_min, x_max = int(x_coords.min()), int(x_coords.max())
                y_min, y_max = int(y_coords.min()), int(y_coords.max())
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Draw landmarks (simplified)
                for lm in landmarks[::10]:  # Every 10th landmark
                    px, py = int(lm[0] * w), int(lm[1] * h)
                    cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

                # Draw FPS
                fps = face_detector.get_fps()
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2
                )

                # Check calibration state
                if not affect_engine.is_calibrated:
                    # Update affect engine (this does calibration)
                    update_result = affect_engine.update(landmarks, frame_size=(w, h))

                    # Draw calibration UI
                    frame = draw_calibration_ui(frame, affect_engine, h, w)
                else:
                    # Show "Ready!" briefly after calibration completes
                    if ready_countdown > 0:
                        frame = draw_ready_ui(frame, h, w, ready_countdown)
                        ready_countdown -= 1

                    # Get emotion weights from trackbars
                    emotion_weights = np.array([cv2.getTrackbarPos(EMOTION_LABELS[i], 'Controls') for i in range(10)])

                    # Get predictions
                    prediction = affect_engine.predict(landmarks, frame_size=(w, h), emotion_weights=emotion_weights)

                    # Draw emotion
                    emotion = prediction['emotion']
                    confidence = prediction['confidence']
                    cv2.putText(
                        frame,
                        f"{emotion}: {confidence:.2f}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2
                    )

                    # Draw all 10 emotion probabilities as bars
                    probs = prediction['probabilities']
                    bar_x = 10
                    bar_y = 120
                    bar_height = 20
                    bar_max_width = 200

                    for i, prob in enumerate(probs):
                        label = EMOTION_LABELS[i]

                        # Draw bar background (gray)
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_width, bar_y + bar_height), (50, 50, 50), -1)

                        # Draw bar fill (green based on probability)
                        bar_width = int(bar_max_width * prob)
                        color = (0, 255, 0) if prob > 0.1 else (100, 100, 100)
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), color, -1)

                        # Draw label
                        cv2.putText(
                            frame,
                            f"{label}: {prob:.1%}",
                            (bar_x + 5, bar_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1
                        )

                        bar_y += 25

                    print(f"Frame {frame_count}: {emotion} ({confidence:.2f})")

        # Show frame
        cv2.imshow('Affect Engine Test', frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

        # Start ready countdown after first calibration complete
        if affect_engine.is_calibrated and ready_countdown == 0:
            ready_countdown = 60  # Show "Ready!" for ~1 second

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    face_detector.close()


if __name__ == '__main__':
    main()
