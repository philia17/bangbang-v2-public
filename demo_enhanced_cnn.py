"""
Test EnhancedCNN-V2 for Real-time Emotion Detection on Mac (Local Camera)

Uses FaceMeshDetector for face detection and EnhancedCNN for emotion classification.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.vision.face_mesh import FaceMeshDetector
from src.models.enhanced_cnn import EnhancedCNN


# Emotion labels (matching FER2013 dataset)
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def preprocess_face(face_crop: np.ndarray) -> torch.Tensor:
    """
    Preprocess face crop for EnhancedCNN input.

    Args:
        face_crop: BGR face crop from OpenCV

    Returns:
        Tensor of shape (1, 1, 48, 48) with values normalized to [-1, 1]
    """
    # Resize to 48x48
    face_resized = cv2.resize(face_crop, (48, 48))

    # Convert to grayscale
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

    # Convert to tensor and normalize: [0, 255] -> [0, 1] -> [-1, 1]
    # ToTensor normalizes to [0, 1], then Normalize with mean=0.5, std=0.5 gives [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    face_tensor = transform(face_gray).unsqueeze(0)  # (1, 1, 48, 48)
    return face_tensor


def main():
    # Model paths
    model_path = "models/enhanced_cnn_fer2013.pth"
    face_model_path = "models/face_landmarker.task"

    # Load EnhancedCNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = EnhancedCNN(num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded EnhancedCNN-V2 from {model_path}")

    # Initialize FaceMeshDetector
    face_detector = FaceMeshDetector(face_model_path)
    print(f"Loaded FaceMeshDetector from {face_model_path}")

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print("\nStarting EnhancedCNN-V2 emotion detection...")
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
                    # Preprocess and run CNN inference
                    face_tensor = preprocess_face(face_crop).to(device)

                    with torch.no_grad():
                        logits = model(face_tensor)
                        probs = F.softmax(logits, dim=1)
                        conf, pred_idx = torch.max(probs, dim=1)

                    emotion = EMOTION_LABELS[pred_idx.item()]
                    confidence = conf.item()
                    probabilities = probs[0].cpu().numpy()

                    # Draw emotion result
                    cv2.putText(
                        frame,
                        f"CNN: {emotion} ({confidence:.2f})",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2
                    )

                    # Draw all emotion probabilities as bars
                    bar_x = 10
                    bar_y = 120
                    bar_height = 20
                    bar_max_width = 200

                    for i, prob in enumerate(probabilities):
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
        cv2.imshow('EnhancedCNN-V2 Emotion Detection', frame)

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