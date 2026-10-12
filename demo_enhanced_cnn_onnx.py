"""
Test EnhancedCNN-ONNX for Real-time Emotion Detection on Mac (Local Camera)

Uses FaceMeshDetector for face detection and ONNX Runtime for inference.
This is the ONNX version of test_enhanced_cnn.py for Pi deployment testing.
"""

import cv2
import numpy as np
import onnxruntime as ort

from src.vision.face_mesh import FaceMeshDetector


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


def main():
    # Model paths
    onnx_model_path = "models/enhanced_cnn_fer2013.onnx"
    face_model_path = "models/face_landmarker.task"

    # Load ONNX model
    print(f"Loading ONNX model from {onnx_model_path}")
    session = ort.InferenceSession(
        onnx_model_path,
        providers=['CPUExecutionProvider', 'CUDAExecutionProvider']
    )
    print(f"ONNX Runtime providers: {session.get_providers()}")

    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Input: {input_name}, Output: {output_name}")

    # Initialize FaceMeshDetector
    face_detector = FaceMeshDetector(face_model_path)
    print(f"Loaded FaceMeshDetector from {face_model_path}")

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print("\nStarting EnhancedCNN-ONNX emotion detection...")
    print("Press 'q' to quit\n")

    frame_count = 0

    while True:
        # Get frame
        ret, frame = cap.read()
        if not ret:
            continue

        # Detect landmarks (async)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        face_detector.detect_async(frame, timestamp_ms)
        face_detector.update_fps(1/30)

        # Get result
        result = face_detector.get_result(timeout_ms=100)

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
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Draw landmarks (every 10th)
                for lm in landmarks[::10]:
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

        # Show frame
        cv2.imshow('EnhancedCNN-ONNX Emotion Detection', frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    face_detector.close()


if __name__ == '__main__':
    main()