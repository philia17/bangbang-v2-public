"""
Affect Engine: Feature Extraction + Temporal Buffer + Welford + Hybrid Fusion
Phase 3: Temporal Feature-Based Pipeline
"""

import numpy as np
from collections import deque
from typing import Optional


# Emotion labels (10 classes: 7 FER2013 + pain + confusion + tired)
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise', 'pain', 'confusion', 'tired']

# Landmark indices
LANDMARKS = {
    # Eye corners (for L_ref)
    'left_eye_left': 33,
    'left_eye_right': 263,
    'right_eye_left': 362,
    'right_eye_right': 130,
    # Inner brow
    'brow_left_inner': 107,
    'brow_right_inner': 336,
    # Nose bridge (for head tilt)
    'nose_bridge': 4,
    # Eye
    'eye_top_left': 159,
    'eye_bottom_left': 145,
    'eye_top_right': 386,
    'eye_bottom_right': 374,
    # Cheek
    'cheek_left': 116,
    'cheek_right': 345,
    # Mouth
    'mouth_upper': 13,
    'mouth_lower': 14,
    'mouth_left': 61,   # Mouth corner left
    'mouth_right': 291,  # Mouth corner right
}


class TemporalBuffer:
    """Temporal buffer for storing feature history."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def add(self, features: np.ndarray):
        """Add new features to buffer."""
        self.buffer.append(features)

    def get(self) -> np.ndarray:
        """Get buffered features as array."""
        if len(self.buffer) < self.window_size:
            # Pad with zeros if not enough frames
            padded = np.zeros((self.window_size, len(self.buffer[0]) if self.buffer else 12))
            for i, feat in enumerate(self.buffer):
                padded[i] = feat
            return padded
        return np.array(self.buffer)

    def is_full(self) -> bool:
        return len(self.buffer) >= self.window_size

    def reset(self):
        self.buffer.clear()


class WelfordCalibrator:
    """Welford's algorithm for online mean/std computation."""

    def __init__(self, window_size: int = 180):
        self.window_size = window_size
        self.count = 0
        self.mean = None
        self.M2 = None

    def update(self, value: float) -> dict:
        """Update with new value and return normalized features."""
        self.count += 1
        delta = value - self.mean if self.mean is not None else 0
        self.mean = (self.mean + delta / self.count) if self.mean is not None else value
        delta2 = value - self.mean
        self.M2 = self.M2 + delta * delta2 if self.M2 is not None else 0

        if self.count < 2:
            return {'z_score': 0, 'delta': 0}

        variance = self.M2 / (self.count - 1)
        std = np.sqrt(variance) if variance > 0 else 1

        return {
            'z_score': (value - self.mean) / std,
            'delta': value / self.mean - 1 if self.mean > 0 else 0
        }

    def reset(self):
        self.count = 0
        self.mean = None
        self.M2 = None


class AffectEngine:
    """
    Main affect engine for temporal emotion detection.

    Pipeline:
    Camera → MediaPipe → Features → Temporal Buffer → MLP+LSTM → Fusion → EMA → Emotion
    """

    def __init__(
        self,
        window_size: int = 10,
        calibration_frames: int = 180,
        ema_alpha: float = 0.3,
        fusion_weights: tuple = (0.6, 0.4)
    ):
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.fusion_weights = fusion_weights

        # Temporal buffer
        self.buffer = TemporalBuffer(window_size)

        # Welford calibrators (one per feature)
        self.calibrators = {
            'AU4': WelfordCalibrator(calibration_frames),
            'AU6': WelfordCalibrator(calibration_frames),
            'EAR': WelfordCalibrator(calibration_frames),
            'MAR': WelfordCalibrator(calibration_frames),
        }

        # EMA state
        self.ema_probs = None
        self.face_confidence = 1.0

        # Calibration state
        self.is_calibrated = False

        # yawn detection (tired counter)
        self.yawn_counter = 0
        self.yawn_threshold = 10  # Frames of yawn to detect tired

    def get_calibration_progress(self) -> tuple:
        """
        Get calibration progress.

        Returns:
            tuple: (current_frame, total_frames, progress_ratio)
        """
        current = self.calibrators['AU4'].count
        total = self.calibrators['AU4'].window_size
        return (current, total, min(current / total, 1.0))

    def compute_features(self, landmarks: np.ndarray, frame_size: tuple = None) -> dict:
        """
        Compute AU4, AU6, EAR, MAR from 478 landmarks.

        Args:
            landmarks: Array of shape (478, 3) with x, y, z coordinates (normalized 0-1)
            frame_size: (width, height) of the original frame (optional)

        Returns:
            dict with AU4, AU6, EAR, MAR values (normalized to ~0.05-0.3 range)
        """
        # If frame_size provided, convert to pixel coordinates
        if frame_size is not None:
            w, h = frame_size
            lm = landmarks.copy()
            lm[:, 0] *= w
            lm[:, 1] *= h
        else:
            lm = landmarks

        # Compute face width in pixels
        x_min, x_max = lm[:, 0].min(), lm[:, 0].max()
        face_width = x_max - x_min

        if face_width < 10:
            face_width = 100  # Default fallback

        # AU4: Brow lowerer - distance from brow to eye (normalized by face width)
        brow_left = lm[LANDMARKS['brow_left_inner']][1]
        eye_left = lm[LANDMARKS['left_eye_left']][1]
        brow_eye_dist = abs(brow_left - eye_left)
        AU4 = brow_eye_dist / face_width  # Already in 0-0.5 range

        # AU6: Cheek raiser - distance from cheek to eye corner
        cheek_left = lm[LANDMARKS['cheek_left']][1]
        eye_bottom = lm[LANDMARKS['eye_bottom_left']][1]
        cheek_eye_dist = abs(cheek_left - eye_bottom)
        AU6 = cheek_eye_dist / face_width

        # EAR: Eye aspect ratio (using eye corner distance as reference)
        left_eye_dist = np.linalg.norm(
            lm[LANDMARKS['left_eye_right']][:2] -
            lm[LANDMARKS['left_eye_left']][:2]
        )
        right_eye_dist = np.linalg.norm(
            lm[LANDMARKS['right_eye_right']][:2] -
            lm[LANDMARKS['right_eye_left']][:2]
        )
        eye_corner_dist = (left_eye_dist + right_eye_dist) / 2

        eye_top = lm[LANDMARKS['eye_top_left']][1]
        eye_bottom = lm[LANDMARKS['eye_bottom_left']][1]
        vertical = abs(eye_top - eye_bottom)

        EAR = vertical / (2 * eye_corner_dist + 1)

        # MAR: Mouth aspect ratio
        mouth_upper = lm[LANDMARKS['mouth_upper']][1]
        mouth_lower = lm[LANDMARKS['mouth_lower']][1]
        mouth_open = abs(mouth_upper - mouth_lower)
        MAR = mouth_open / face_width

        # Mouth corner direction (for sad/happy detection)
        mouth_left = lm[LANDMARKS['mouth_left']][1]  # y-coordinate
        mouth_right = lm[LANDMARKS['mouth_right']][1]  # y-coordinate
        mouth_corner_diff = (mouth_left - mouth_right) / face_width * 10  # Positive = left down (sad), Negative = left up (happy)

        # Head tilt: angle from nose bridge relative to vertical
        nose_x = lm[LANDMARKS['nose_bridge']][0]
        nose_left_x = lm[LANDMARKS['brow_left_inner']][0]  # proxy for left side
        nose_right_x = lm[LANDMARKS['brow_right_inner']][0]  # proxy for right side
        head_tilt = abs(nose_left_x - nose_right_x) / face_width  # 0 = upright, larger = tilted

        return {'AU4': AU4, 'AU6': AU6, 'EAR': EAR, 'MAR': MAR, 'mouth_corner_diff': mouth_corner_diff, 'head_tilt': head_tilt}

    def calibrate(self, features: dict):
        """Update calibrators with calibration frames."""
        for key in ['AU4', 'AU6', 'EAR', 'MAR']:
            self.calibrators[key].update(features[key])

    def get_normalized_features(self, features: dict) -> np.ndarray:
        """
        Get normalized + delta + z-score features.

        Returns:
            12-dimensional feature vector
        """
        feature_vector = []
        for key in ['AU4', 'AU6', 'EAR', 'MAR']:
            value = features[key]
            calib = self.calibrators[key]

            # Raw value
            feature_vector.append(value)

            # Delta and z-score (if calibrated)
            if self.is_calibrated:
                normalized = calib.update(value)
                feature_vector.append(normalized['delta'])
                feature_vector.append(normalized['z_score'])
            else:
                feature_vector.append(0.0)
                feature_vector.append(0.0)

        return np.array(feature_vector)

    def extract_features(self, landmarks: np.ndarray, frame_size: tuple = None) -> np.ndarray:
        """
        Main feature extraction pipeline.

        Args:
            landmarks: 478 landmark points
            frame_size: (width, height) of the original frame

        Returns:
            12-dimensional feature vector
        """
        # Compute raw features
        raw_features = self.compute_features(landmarks, frame_size)

        # Calibrate during warmup
        if not self.is_calibrated:
            self.calibrate(raw_features)
            if self.calibrators['AU4'].count >= self.calibrators['AU4'].window_size:
                self.is_calibrated = True
            return np.zeros(12)  # Return zeros during calibration

        # Get normalized features
        normalized = self.get_normalized_features(raw_features)
        return normalized

    def update(self, landmarks: np.ndarray, frame_size: tuple = None) -> dict:
        """
        Update engine with new landmarks.

        Args:
            landmarks: 478x3 numpy array
            frame_size: (width, height) of the original frame

        Returns:
            dict with: features, buffer, emotion_scores
        """
        # Extract features
        features = self.extract_features(landmarks, frame_size)

        if features.sum() == 0:
            return None  # Still calibrating

        # Add to temporal buffer
        self.buffer.add(features)

        return {
            'features': features,
            'buffer_full': self.buffer.is_full()
        }

    def get_buffer(self) -> np.ndarray:
        """Get temporal buffer as model input."""
        return self.buffer.get()

    def au_logic(self, features: dict) -> np.ndarray:
        """
        Rule-based AU detection with decision tree + weighted scoring.

        Returns:
            10-class probability scores (angry, disgust, fear, happy, neutral, sad, surprise, pain, confusion, tired)

        Strategy:
        1. Decision tree for strong signals (high confidence)
        2. Weighted scoring for ambiguous cases
        3. Neutral suppression when other emotions are active
        """
        probs = np.zeros(10)  # No base probability - start from 0

        AU4 = features['AU4']
        AU6 = features['AU6']
        EAR = features['EAR']
        MAR = features['MAR']
        mouth_corner = features.get('mouth_corner_diff', 0)
        head_tilt = features.get('head_tilt', 0)

        # Index mapping: 0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad, 6=surprise, 7=pain, 8=confusion, 9=tired

        # === Decision Tree: Strong signals take priority ===
        # Only triggers when signal is clear and unmistakable

        # 1. Clear happy:嘴角上扬 + 脸颊提升
        if mouth_corner < -0.025 and AU6 > 0.06:
            probs[3] = 0.75
            probs[6] = 0.1  # slight surprise component
            probs[4] = 0.1
            return probs / probs.sum()

        # 2. Clear surprise:眼睛大 + 嘴巴张开
        if EAR < 0.025 and MAR > 0.08:
            probs[6] = 0.7
            probs[2] = 0.15  # fear component
            probs[4] = 0.1
            return probs / probs.sum()

        # 3. Clear sad:嘴角下垂 + 眉毛降低
        if mouth_corner > 0.025 and AU4 < 0.11:
            probs[5] = 0.65
            probs[4] = 0.2
            probs[0] = 0.1  # slight angry component
            return probs / probs.sum()

        # 4. Clear tired:眼睛半闭 + 眉毛低 (not yawn-based)
        if EAR < 0.035 and AU4 > 0.10:
            probs[9] = 0.7
            probs[4] = 0.2
            return probs / probs.sum()

        # === Weighted Scoring: Ambiguous cases ===
        # Mouth corner detection for sad/happy
        if mouth_corner > 0.025:
            probs[5] += 0.3  # Sad
        elif mouth_corner < -0.025:
            probs[3] += 0.4  # Happy

        # AU4 (brow lowerer) - angry/fear/sad/neutral/pain
        if AU4 < 0.09:
            probs[0] += 0.45  # Angry (very strong)
            probs[5] += 0.1  # Sad (suppress slightly)
        elif AU4 < 0.11:
            probs[0] += 0.35  # Angry (medium)
            probs[5] += 0.15  # Sad (suppress)
        elif AU4 < 0.12:
            probs[0] += 0.25  # Angry (mild)
            probs[5] += 0.1  # Sad
        elif AU4 < 0.13:
            probs[4] += 0.2  # Neutral
        elif AU4 > 0.15:
            probs[7] += 0.3  # Pain

        # Angry combination: brow lowerer + mouth corner down
        if AU4 < 0.11 and mouth_corner > 0.015:
            probs[0] += 0.35  # Angry (combination)

        # AU6 (cheek raiser) - happy/neutral
        if AU6 > 0.08:
            probs[3] += 0.35  # Happy
        elif AU6 > 0.06:
            probs[4] += 0.15  # Neutral

        # EAR - surprise/fear/neutral/pain/confusion/tired
        if EAR < 0.025:
            probs[6] += 0.65  # Surprise (strengthened)
            probs[2] += 0.25  # Fear
        elif EAR < 0.03:
            probs[7] += 0.4  # Pain
        elif EAR < 0.035:
            probs[9] += 0.35  # Tired
        elif EAR > 0.05:
            probs[4] += 0.2  # Neutral
        elif EAR > 0.04:
            probs[8] += 0.3  # Confusion (slight squint)
        else:
            probs[8] += 0.35  # Confusion (more squint)

        # Confusion combination: squinting + brows up
        if EAR < 0.035 and AU4 > 0.14:
            probs[8] += 0.4  # Confusion (squinting + brows up)

        # Confusion: head tilt (curious/uncertain)
        if head_tilt > 0.08 and AU4 > 0.12:
            probs[8] += 0.35  # Confusion (head tilt + brows up)

        # MAR - surprise/happy/neutral/disgust/confusion
        if MAR > 0.12:
            probs[6] += 0.35  # Surprise
            probs[3] += 0.2  # Happy
        elif MAR > 0.08:
            probs[6] += 0.2  # Surprise (medium)
        elif MAR > 0.06:
            probs[3] += 0.25  # Happy
        elif MAR > 0.03:
            probs[4] += 0.2  # Neutral
        elif MAR > 0.01:
            probs[8] += 0.25  # Confusion
        else:
            probs[6] += 0.1  # Surprise (tight mouth)

        # Disgust detection (combination)
        if AU4 > 0.14 and MAR > 0.04:
            probs[1] += 0.3  # Disgust

        # Fear detection
        if AU4 < 0.10 and EAR < 0.03:
            probs[2] += 0.3  # Fear

        # Sad detection (brows down + not angry)
        if AU4 < 0.11 and MAR < 0.05:
            probs[5] += 0.25  # Sad

        # Pain detection (combination)
        if AU4 > 0.13 and EAR < 0.028:
            probs[7] += 0.5  # Pain

        # Confusion detection
        if abs(AU4 - 0.13) < 0.02 and abs(MAR - 0.03) < 0.015:
            probs[8] += 0.35  # Confusion

        # Tired/Yawn detection (over time)
        if MAR > 0.12:
            self.yawn_counter += 1
        else:
            self.yawn_counter = max(0, self.yawn_counter - 1)

        if self.yawn_counter >= self.yawn_threshold:
            probs[9] += 0.45  # Tired (sustained yawn)
        elif EAR < 0.035 and AU4 > 0.10:
            probs[9] += 0.3  # Tired (eyes droopy)

        # === Neutral Suppression ===
        # When other emotions are active, reduce neutral
        other_max = max(probs[i] for i in range(10) if i != 4)
        if other_max > 0.35:
            probs[4] *= 0.5  # Suppress neutral

        # === Happy-Sad Ambivalence → Neutral ===
        # When happy and sad are similar in score, interpret as neutral
        # (face not clearly expressing one or the other)
        happy_sad_diff = abs(probs[3] - probs[5])
        happy_sad_avg = (probs[3] + probs[5]) / 2
        if happy_sad_diff < 0.15 and happy_sad_avg > 0.1:
            # Happy and sad are close and both present → leaning neutral
            boost = happy_sad_avg * 0.4
            probs[4] += boost
            # Reduce the dominant one slightly
            if probs[3] > probs[5]:
                probs[3] *= 0.85
            else:
                probs[5] *= 0.85

        # If no emotion is strong enough, boost neutral as fallback
        if probs.max() < 0.15:
            probs[4] = 0.5  # Default to neutral
            return probs / probs.sum()

        # Normalize
        probs = probs / probs.sum()
        return probs

    def hybrid_fusion(
        self,
        au_probs: np.ndarray,
        ml_probs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Hybrid fusion of AU logic + ML model.

        Args:
            au_probs: Rule-based probabilities
            ml_probs: ML model probabilities (optional)

        Returns:
            Fused probabilities
        """
        if ml_probs is None:
            return au_probs

        rule_weight, ml_weight = self.fusion_weights
        fused = rule_weight * au_probs + ml_weight * ml_probs
        return fused / fused.sum()

    def ema_smooth(self, probs: np.ndarray) -> np.ndarray:
        """
        EMA smoothing.

        Args:
            probs: Raw probabilities

        Returns:
            Smoothed probabilities
        """
        if self.ema_probs is None:
            self.ema_probs = probs
            return probs

        smoothed = self.ema_alpha * probs + (1 - self.ema_alpha) * self.ema_probs
        self.ema_probs = smoothed
        return smoothed

    def predict(
        self,
        landmarks: np.ndarray,
        ml_probs: Optional[np.ndarray] = None,
        frame_size: tuple = None,
        emotion_weights: Optional[np.ndarray] = None
    ) -> dict:
        """
        Main prediction pipeline.

        Args:
            landmarks: 478x3 numpy array
            ml_probs: ML model probabilities (optional)
            frame_size: (width, height) of the original frame

        Returns:
            dict with emotion, probabilities, confidence
        """
        # Update
        result = self.update(landmarks, frame_size)
        if result is None:
            return {
                'emotion': 'calibrating',
                'emotion_id': -1,
                'confidence': 0.0,
                'probabilities': np.zeros(10)
            }

        # AU logic
        raw_features = self.compute_features(landmarks, frame_size)
        au_probs = self.au_logic(raw_features)

        # Apply emotion weights (for real-time tuning via trackbars)
        if emotion_weights is not None:
            for i in range(10):
                au_probs[i] *= emotion_weights[i] / 100.0

        # Hybrid fusion
        fused = self.hybrid_fusion(au_probs, ml_probs)

        # EMA smoothing
        smoothed = self.ema_smooth(fused)

        # Get final prediction
        emotion_id = np.argmax(smoothed)
        emotion = EMOTION_LABELS[emotion_id]

        return {
            'emotion': emotion,
            'emotion_id': emotion_id,
            'confidence': smoothed[emotion_id],
            'probabilities': smoothed
        }

    def reset(self):
        """Reset calibrators and buffers."""
        self.buffer.reset()
        for calib in self.calibrators.values():
            calib.reset()
        self.ema_probs = None
        self.is_calibrated = False