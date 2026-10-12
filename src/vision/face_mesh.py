"""
FaceMeshDetector: MediaPipe V2 Face Landmarker wrapper for Bangbang V2 Emotion Engine.

This module provides a thread-safe interface to the MediaPipe FaceLandmarker API,
extracting 478 3D facial landmarks for emotion detection.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Optional, List, Tuple
import threading


class FaceMeshDetector:
    """
    Wrapper class for MediaPipe FaceLandmarker V2 API.

    Provides real-time 3D facial landmark detection with spatial normalization
    support for distance-invariant emotion detection.
    """

    # Key landmark indices for AU detection (MediaPipe V2)
    # Reference: https://face-api.net/landmarks
    LANDMARK_LEFT_EYE_OUTER = 33   # L_ref left corner
    LANDMARK_RIGHT_EYE_OUTER = 263 # L_ref right corner
    LANDMARK_LEFT_EYE_INNER = 133   # AU4 anchor (inner corner - new)
    LANDMARK_RIGHT_EYE_INNER = 362   # AU4 anchor (inner corner - new)
    LANDMARK_LEFT_BROW_INNER = 107   # AU4 (Brow Lowerer)
    LANDMARK_RIGHT_BROW_INNER = 336   # AU4 (Brow Lowerer)
    LANDMARK_LEFT_CHEEK = 205            # AU6 (Cheek Raiser)
    LANDMARK_RIGHT_CHEEK = 425         # AU6 (Cheek Raiser)
    LANDMARK_LEFT_EYE_BOTTOM = 145      # AU6 anchor (lower eyelid)
    LANDMARK_RIGHT_EYE_BOTTOM = 374     # AU6 anchor (lower eyelid)
    LANDMARK_LEFT_HIGH_CHEEK = 116   # Upper cheek (new - larger range)
    LANDMARK_RIGHT_HIGH_CHEEK = 345   # Upper cheek (new - larger range)

    # EAR - Eye Aspect Ratio landmarks (MediaPipe viewer's perspective, mirrored)
    # Viewer's right = Person's LEFT eye
    LANDMARK_RIGHT_EYE_UPPER = 159    # Person's LEFT eye upper lid
    LANDMARK_RIGHT_EYE_LOWER = 145  # Person's LEFT eye lower lid
    # Viewer's left = Person's RIGHT eye
    LANDMARK_LEFT_EYE_UPPER = 386     # Person's RIGHT eye upper lid
    LANDMARK_LEFT_EYE_LOWER = 374   # Person's RIGHT eye lower lid

    # MAR - Mouth Aspect Ratio landmarks
    LANDMARK_UPPER_LIP = 13        # Upper lip mid
    LANDMARK_LOWER_LIP = 14         # Lower lip mid

    def __init__(
        self,
        model_path: str,
        num_faces: int = 1,
        min_confidence: float = 0.5,
        output_blendshapes: bool = False
    ):
        """
        Initialize the FaceMeshDetector.

        Args:
            model_path: Path to the MediaPipe .task model file
            num_faces: Maximum number of faces to detect
            min_confidence: Minimum face detection confidence threshold
            output_blendshapes: Whether to output facial blendshapes
        """
        self.model_path = model_path
        self.num_faces = num_faces
        self.min_confidence = min_confidence
        self.output_blendshapes = output_blendshapes

        # Initialize the landmarker
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=num_faces,
            min_face_detection_confidence=min_confidence,
            min_tracking_confidence=0.5,
            output_face_blendshapes=output_blendshapes,
            output_facial_transformation_matrixes=True,
            result_callback=self._on_result
        )

        self._lock = threading.Lock()
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

        # Neutral baseline for relative calibration
        self._baseline_lock = threading.Lock()
        self._neutral_baseline: Optional[dict] = None

        # Result storage
        self._latest_result: Optional[vision.FaceLandmarkerResult] = None
        self._result_ready = threading.Event()

        # FPS tracking
        self._frame_count = 0
        self._fps = 0.0
        self._fps_lock = threading.Lock()

    def _on_result(
        self,
        result: vision.FaceLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int
    ):
        """Callback for async landmark detection."""
        with self._lock:
            self._latest_result = result
        self._result_ready.set()

        # Update FPS
        with self._fps_lock:
            self._frame_count += 1

    def detect_async(self, frame: np.ndarray, timestamp_ms: int) -> None:
        """
        Detect landmarks asynchronously from a camera frame.

        Args:
            frame: BGR frame from OpenCV (H, W, 3)
            timestamp_ms: Timestamp in milliseconds
        """
        # Convert BGR to RGB
        if frame is None or frame.size == 0:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self._result_ready.clear()
        self._landmarker.detect_async(mp_image, timestamp_ms)

    def get_result(
        self,
        timeout_ms: int = 100
    ) -> Optional[vision.FaceLandmarkerResult]:
        """
        Get the latest detection result.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            FaceLandmarkerResult or None if not available
        """
        if self._result_ready.wait(timeout=timeout_ms / 1000.0):
            with self._lock:
                return self._latest_result
        return None

    def get_landmarks(
        self,
        face_idx: int = 0
    ) -> Optional[np.ndarray]:
        """
        Get 3D landmarks for a specific face.

        Args:
            face_idx: Index of the face (0 for first face)

        Returns:
            numpy array of shape (478, 3) containing x, y, z coordinates,
            or None if no face detected
        """
        result = self.get_result()
        if result is None or not result.face_landmarks:
            return None

        if face_idx >= len(result.face_landmarks):
            return None

        landmarks = result.face_landmarks[face_idx]
        # Convert to numpy array
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        return coords

    def get_scale_reference(self, landmarks: np.ndarray) -> float:
        """
        Calculate the inter-ocular distance (L_ref) for normalization.

        Uses only X,Y to avoid Z-axis drift from MediaPipe's estimated depth.

        Args:
            landmarks: Array of shape (478, 3)

        Returns:
            Euclidean distance between landmark 33 and 263 (X,Y only)
        """
        if landmarks is None or len(landmarks) < 384:
            return 0.0

        p_left = landmarks[self.LANDMARK_LEFT_EYE_OUTER][:2]  # X,Y only - no Z
        p_right = landmarks[self.LANDMARK_RIGHT_EYE_OUTER][:2]  # X,Y only - no Z

        return np.linalg.norm(p_right - p_left)

    def normalize_distance(
        self,
        distance: float,
        scale_ref: float
    ) -> float:
        """
        Apply distance-invariant normalization.

        Args:
            distance: Raw Euclidean distance between two landmarks
            scale_ref: L_ref (inter-ocular distance)

        Returns:
            Normalized distance D_norm = distance / L_ref
        """
        if scale_ref <= 0:
            return 0.0
        return distance / scale_ref

    def get_au_distances(
        self,
        landmarks: np.ndarray
    ) -> dict:
        """
        Calculate AU-relevant distances for emotion detection.

        Args:
            landmarks: Array of shape (478, 3)

        Returns:
            Dictionary with AU-specific normalized distances:
            - 'au4_brow': AU4 Brow Lowerer (inner brow to inner eye)
            - 'au6_cheek': AU6 Cheek Raiser (high cheek to lower eyelid)
        """
        if landmarks is None or len(landmarks) < 426:
            return {}

        scale_ref = self.get_scale_reference(landmarks)
        if scale_ref <= 0:
            return {}

        au_distances = {}

        # AU4: Brow Lowerer (Y-axis displacement from inner brow to inner eye)
        # Uses inner eye (133/362) instead of outer for higher sensitivity
        p_brow_left = landmarks[self.LANDMARK_LEFT_BROW_INNER]
        p_eye_left = landmarks[self.LANDMARK_LEFT_EYE_INNER]
        au4_left = abs(p_brow_left[1] - p_eye_left[1])  # Y-axis difference

        p_brow_right = landmarks[self.LANDMARK_RIGHT_BROW_INNER]
        p_eye_right = landmarks[self.LANDMARK_RIGHT_EYE_INNER]
        au4_right = abs(p_brow_right[1] - p_eye_right[1])  # Y-axis difference

        # Average AU4 score normalized by L_ref
        au4_score = ((au4_left + au4_right) / 2) / scale_ref
        au_distances['au4_brow'] = au4_score

        # AU6: Cheek Raiser (Y-axis displacement of high cheek relative to eyelid)
        # Uses high cheek (116/345) for larger vertical range
        p_cheek_left = landmarks[self.LANDMARK_LEFT_HIGH_CHEEK]
        p_eyelid_left = landmarks[self.LANDMARK_LEFT_EYE_BOTTOM]
        au6_left = abs(p_cheek_left[1] - p_eyelid_left[1])  # Y-axis difference

        p_cheek_right = landmarks[self.LANDMARK_RIGHT_HIGH_CHEEK]
        p_eyelid_right = landmarks[self.LANDMARK_RIGHT_EYE_BOTTOM]
        au6_right = abs(p_cheek_right[1] - p_eyelid_right[1])  # Y-axis difference

        # Average AU6 score normalized by L_ref
        au6_score = ((au6_left + au6_right) / 2) / scale_ref
        au_distances['au6_cheek'] = au6_score

        return au_distances

    def get_ear(self, landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for detecting eye squinting and blink.

        Uses vertical eye opening normalized by full eye width.
        EAR = ((|upper - lower|) / |outer_corner_left - outer_corner_right|)

        Typical values:
        - Idle (eyes open): EAR ~ 0.25-0.35
        - Blink: EAR → 0
        - Squint: EAR < 0.15

        Args:
            landmarks: Array of shape (478, 3)

        Returns:
            Eye Aspect Ratio (0 = fully closed, ~0.3 = fully open)
        """
        if landmarks is None or len(landmarks) < 387:
            return 0.0

        # Get scale reference (full eye width)
        eye_width = self.get_scale_reference(landmarks)
        if eye_width <= 0:
            return 0.0

        # Get eye lid positions (Y-axis only - vertical opening)
        # Using viewer's perspective: 159/145 = viewer's right eye (person's left)
        p_upper_right = landmarks[self.LANDMARK_RIGHT_EYE_UPPER][1]
        p_lower_right = landmarks[self.LANDMARK_RIGHT_EYE_LOWER][1]
        right_eye_opening = abs(p_upper_right - p_lower_right)

        p_upper_left = landmarks[self.LANDMARK_LEFT_EYE_UPPER][1]
        p_lower_left = landmarks[self.LANDMARK_LEFT_EYE_LOWER][1]
        left_eye_opening = abs(p_upper_left - p_lower_left)

        # Average of both eyes, normalized by eye width
        avg_opening = (right_eye_opening + left_eye_opening) / 2
        ear = avg_opening / eye_width

        return ear

    def get_mar(self, landmarks: np.ndarray) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR) for detecting jaw drop.

        Uses vertical mouth opening normalized by L_ref.
        MAR = |upper_lip - lower_lip| / L_ref

        Typical values:
        - Idle (mouth closed): MAR ~ 0.02-0.04
        - Talking: MAR varies 0.05-0.10
        - Yawn/Gasp: MAR > 0.10

        Args:
            landmarks: Array of shape (478, 3)

        Returns:
            Mouth Aspect Ratio (0 = closed, ~0.1+ = open)
        """
        if landmarks is None or len(landmarks) < 15:
            return 0.0

        # Get scale reference
        scale_ref = self.get_scale_reference(landmarks)
        if scale_ref <= 0:
            return 0.0

        # Get lip positions (Y-axis only - vertical opening)
        p_upper_lip = landmarks[self.LANDMARK_UPPER_LIP][1]
        p_lower_lip = landmarks[self.LANDMARK_LOWER_LIP][1]
        mouth_opening = abs(p_lower_lip - p_upper_lip)

        # Normalize by scale reference
        mar = mouth_opening / scale_ref

        return mar

    def get_fps(self) -> float:
        """Get current detection FPS."""
        with self._fps_lock:
            return self._fps

    def update_fps(self, dt: float) -> None:
        """
        Update FPS counter.

        Args:
            dt: Time delta in seconds since last update
        """
        with self._fps_lock:
            if dt > 0:
                self._fps = 1.0 / dt

    def set_neutral_baseline(self, landmarks: np.ndarray) -> None:
        """
        Capture neutral face baseline for relative calibration.

        Records the current facial expression as the baseline (neutral) reference.
        Subsequent calls to get_relative_features() will return values relative to this baseline.
        Should be called at program start with the user in a neutral expression.

        Args:
            landmarks: Array of shape (478, 3) from a neutral expression
        """
        au_distances = self.get_au_distances(landmarks)
        with self._baseline_lock:
            self._neutral_baseline = {
                'au4_brow': au_distances.get('au4_brow', 0.0),
                'au6_cheek': au_distances.get('au6_cheek', 0.0),
                'ear': self.get_ear(landmarks),
                'mar': self.get_mar(landmarks),
                'lref': self.get_scale_reference(landmarks),
            }

    def get_relative_features(self, landmarks: np.ndarray) -> dict:
        """
        Return features relative to the neutral baseline.

        Calculates the ratio of current features to baseline features.
        Delta = (current / baseline) - 1

        Values:
        - 0 or near 0 = same as baseline (neutral)
        - Positive = feature increased (e.g., brow lowered, jaw dropped)
        - Negative = feature decreased (e.g., squinting)

        Args:
            landmarks: Array of shape (478, 3)

        Returns:
            Dictionary with relative feature values:
            - 'delta_au4': AU4 relative change
            - 'delta_au6': AU6 relative change
            - 'delta_ear': EAR relative change (negative = squinting)
            - 'delta_mar': MAR relative change (positive = jaw dropped)
            Returns absolute values if no baseline set.
        """
        features = self.get_all_features(landmarks)
        with self._baseline_lock:
            if self._neutral_baseline is None:
                return features

            baseline = self._neutral_baseline
            return {
                'delta_au4': self._safe_ratio(
                    features['au4_brow'], baseline['au4_brow']
                ) - 1,
                'delta_au6': self._safe_ratio(
                    features['au6_cheek'], baseline['au6_cheek']
                ) - 1,
                'delta_ear': self._safe_ratio(
                    features['ear'], baseline['ear']
                ) - 1,
                'delta_mar': self._safe_ratio(
                    features['mar'], baseline['mar']
                ) - 1,
            }

    def get_all_features(self, landmarks: np.ndarray) -> dict:
        """
        Get all facial features in a single dictionary.

        Combines AU distances and EAR/MAR into one aggregation.

        Args:
            landmarks: Array of shape (478, 3)

        Returns:
            Dictionary with all features:
            - 'au4_brow': AU4 Brow Lowerer score
            - 'au6_cheek': AU6 Cheek Raiser score
            - 'ear': Eye Aspect Ratio
            - 'mar': Mouth Aspect Ratio
            - 'lref': Scale reference (inter-ocular distance)
        """
        au_dist = self.get_au_distances(landmarks)
        return {
            'au4_brow': au_dist.get('au4_brow', 0.0),
            'au6_cheek': au_dist.get('au6_cheek', 0.0),
            'ear': self.get_ear(landmarks),
            'mar': self.get_mar(landmarks),
            'lref': self.get_scale_reference(landmarks),
        }

    def reset_neutral_baseline(self) -> None:
        """
        Clear the neutral baseline to return to absolute values.

        After clearing, get_relative_features() will return absolute values.
        """
        with self._baseline_lock:
            self._neutral_baseline = None

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        """Safe division to avoid division by zero."""
        if denominator <= 0 or numerator <= 0:
            return 0.0
        return numerator / denominator

    def close(self) -> None:
        """Release resources."""
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def draw_landmarks(
    image: np.ndarray,
    landmarks: np.ndarray,
    highlight_indices: Optional[List[int]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 2
) -> np.ndarray:
    """
    Draw facial landmarks on an image for visualization.

    Args:
        image: BGR image from OpenCV
        landmarks: Array of shape (478, 3)
        highlight_indices: Specific landmark indices to highlight
        color: BGR color tuple
        radius: Point radius

    Returns:
        Image with drawn landmarks
    """
    if landmarks is None:
        return image

    h, w = image.shape[:2]
    output = image.copy()

    # Draw all landmarks
    for lm in landmarks:
        x, y = int(lm[0] * w), int(lm[1] * h)
        cv2.circle(output, (x, y), radius, color, -1)

    # Highlight specific indices
    if highlight_indices:
        for idx in highlight_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                x, y = int(lm[0] * w), int(lm[1] * h)
                cv2.circle(output, (x, y), radius * 2, (0, 0, 255), -1)

    return output


# Key landmark indices for reference
KEY_LANDMARKS = {
    'left_eye_outer': 33,
    'right_eye_outer': 263,
    'left_eye_inner': 133,
    'right_eye_inner': 362,
    'left_brow_inner': 107,
    'right_brow_inner': 336,
    'left_cheek': 205,
    'right_cheek': 425,
    'left_high_cheek': 116,
    'right_high_cheek': 345,
    # EAR landmarks
    'right_eye_upper': 159,  # Viewer's right = Person's left
    'right_eye_lower': 145,
    'left_eye_upper': 386,   # Viewer's left = Person's right
    'left_eye_lower': 374,
    # MAR landmarks
    'upper_lip': 13,
    'lower_lip': 14,
}