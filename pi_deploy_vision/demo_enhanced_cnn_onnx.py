"""
Test EnhancedCNN-ONNX for Real-time Emotion Detection
Pi deployment version with headless support.

Uses FaceMeshDetector for face detection and ONNX Runtime for inference.
Set DEBUG_DISPLAY=False for SSH/headless mode.
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
import sys

from face_mesh import FaceMeshDetector
from camera import CameraReader


# === CONFIGURATION ===
DEBUG_DISPLAY = os.environ.get('DEBUG_DISPLAY', '1') == '1'  # Set to '0' for headless
HEADLESS_FRAME_SAVE = os.environ.get('HEADLESS_FRAME_SAVE', None)  # Path to save frames in headless mode
# ====================


# Emotion labels (matching FER2013 dataset)
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def preprocess_face_onnx(face_crop: np.ndarray) -> np.ndarray:
    """
    Preprocess face crop for EnhancedCNN ONNX input using pure OpenCV/NumPy.

    Args:
        face_crop: BGR face crop from OpenCV

    Returns:
        numpy array of shape (1, 1, 48, 48) with values normalized to [-1, 1]
    """
    # Resize to 48x48
    face_resized = cv2.resize(face_crop, (48, 48))

    # Convert to grayscale
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

    # ToTensor equivalent: [0, 255] -> [0, 1] as float32
    face_tensor = face_gray.astype(np.float32) / 255.0

    # Normalize: [0, 1] -> [-1, 1] using (img - mean) / std
    # mean=0.5, std=0.5 -> (img - 0.5) / 0.5 = img * 2 - 1
    face_tensor = (face_tensor - 0.5) / 0.5

    # Add batch and channel dimensions: (48, 48) -> (1, 1, 48, 48)
    face_tensor = face_tensor[np.newaxis, np.newaxis, :, :]

    return face_tensor


def softmax(logits: np.ndarray) -> np.ndarray:
    """Apply softmax to logits."""
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def headless_frame_save(frame, count):
    """Save frame to file in headless mode."""
    if HEADLESS_FRAME_SAVE:
        path = os.path.join(HEADLESS_FRAME_SAVE, f"frame_{count:06d}.jpg")
        cv2.imwrite(path, frame)


def main():
    # Model paths
    onnx_model_path = "models/enhanced_cnn_fer2013.onnx"
    face_model_path = "models/face_landmarker.task"

    # Check headless mode
    if not DEBUG_DISPLAY:
        print("=== HEADLESS MODE ===")
        if HEADLESS_FRAME_SAVE:
            os.makedirs(HEADLESS_FRAME_SAVE, exist_ok=True)
            print(f"Will save frames to: {HEADLESS_FRAME_SAVE}")
        print("====================")

    # Load ONNX model
    print(f"Loading ONNX model from {onnx_model_path}")
    session = ort.InferenceSession(
        onnx_model_path,
        providers=['CPUExecutionProvider']
    )
    print(f"ONNX Runtime providers: {session.get_providers()}")

    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Input: {input_name}, Output: {output_name}")

    # Initialize FaceMeshDetector
    face_detector = FaceMeshDetector(face_model_path)
    print(f"Loaded FaceMeshDetector from {face_model_path}")

    # Open camera with probing
    cam = CameraReader()
    if not cam.is_opened():
        print("FATAL: Cannot open any camera")
        sys.exit(1)

    print("\nStarting EnhancedCNN-ONNX emotion detection...")
    print("Press 'q' to quit (or Ctrl+C)\n")

    frame_count = 0

    while True:
        ret, frame = cam.read()
        if cam.failed():
            break
        if not ret:
            continue

        # Detect landmarks (async)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        face_detector.detect_async(frame, timestamp_ms)
        face_detector.update_fps(1/30)

        # Get result
        result = face_detector.get_result(timeout_ms=100)

        emotion = None
        confidence = 0.0
        probs = None

        if result is not None and result.face_landmarks:
            landmarks = face_detector.get_landmarks()

            if landmarks is not None:
                h, w = frame.shape[:2]

                # Get face bounding box
                x_coords = landmarks[:, 0] * w
                y_coords = landmarks[:, 1] * h
                x_min, x_max = int(x_coords.min()), int(x_coords.max())
                y_min, y_max = int(y_coords.min()), int(y_coords.max())

                # Draw face bounding box (green)
                if DEBUG_DISPLAY:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Draw landmarks (every 10th)
                    for lm in landmarks[::10]:
                        px, py = int(lm[0] * w), int(lm[1] * h)
                        cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

                # Crop face region
                face_crop = frame[y_min:y_max, x_min:x_max]

                if face_crop.size > 0:
                    # Preprocess and run ONNX inference
                    face_tensor = preprocess_face_onnx(face_crop)

                    # ONNX inference
                    logits = session.run([output_name], {input_name: face_tensor})[0]
                    probs = softmax(logits)[0]

                    conf, pred_idx = np.max(probs), np.argmax(probs)

                    emotion = EMOTION_LABELS[pred_idx]
                    confidence = conf

                    if DEBUG_DISPLAY:
                        # Draw FPS
                        fps = face_detector.get_fps()
                        cv2.putText(
                            frame,
                            f"FPS: {fps:.1f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2
                        )

                        # Draw emotion result
                        cv2.putText(
                            frame,
                            f"ONNX: {emotion} ({confidence:.2f})",
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2
                        )

                        # Draw all emotion probabilities as bars
                        bar_x = 10
                        bar_y = 120
                        bar_height = 20
                        bar_max_width = 200

                        for i, prob in enumerate(probs):
                            label = EMOTION_LABELS[i]

                            # Draw bar background (gray)
                            cv2.rectangle(
                                frame,
                                (bar_x, bar_y),
                                (bar_x + bar_max_width, bar_y + bar_height),
                                (50, 50, 50),
                                -1
                            )

                            # Draw bar fill (green based on probability)
                            bar_width = int(bar_max_width * prob)
                            color = (0, 255, 0) if prob > 0.1 else (100, 100, 100)
                            cv2.rectangle(
                                frame,
                                (bar_x, bar_y),
                                (bar_x + bar_width, bar_y + bar_height),
                                color,
                                -1
                            )

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

        # Headless mode: print status every 30 frames
        if not DEBUG_DISPLAY and frame_count % 30 == 0 and frame_count > 0:
            fps = face_detector.get_fps()
            status = f"{emotion} ({confidence:.2f})" if emotion else "no face"
            print(f"[{frame_count}] FPS: {fps:.1f}, Emotion: {status}")

        # Display or save
        if DEBUG_DISPLAY:
            cv2.imshow('EnhancedCNN-ONNX Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # Headless: save frame periodically
            if HEADLESS_FRAME_SAVE and frame_count % 10 == 0:
                headless_frame_save(frame, frame_count)

        frame_count += 1

    # Cleanup
    print(f"\nExiting... (processed {frame_count} frames)")
    cam.release()
    if DEBUG_DISPLAY:
        cv2.destroyAllWindows()
    face_detector.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
