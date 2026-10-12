"""
Affect Engine V2: AU-First Architecture
Standard FACS-based emotion detection with rule-based reasoning + feedback learning

Emotion Classes (7 standard):
  0: angry, 1: disgust, 2: fear, 3: happy, 4: neutral, 5: sad, 6: surprise
"""

import numpy as np
from collections import deque
from typing import Optional, Dict, List


# Emotion labels (6 standard FACS emotions, disgust removed)
EMOTION_LABELS = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# AU indices (standard FACS)
AU_NAMES = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10',
           'AU12', 'AU15', 'AU17', 'AU23', 'AU24', 'AU25', 'AU26']
NUM_AUS = len(AU_NAMES)  # 15
AUS = AU_NAMES  # Alias for convenience

# AU → Emotion mapping (positive correlations, higher AU → emotion)
# Based on FACS standard relationships
AU_EMOTION_MAP = {
    # AU1 + AU2: Inner/Outer brow raiser → fear, sad
    'AU1': {'fear': 0.7, 'sad': 0.5},
    'AU2': {'fear': 0.6, 'sad': 0.4},
    # AU4: Brow lowerer → angry, sad, fear
    'AU4': {'angry': 0.8, 'sad': 0.4, 'fear': 0.3},
    # AU5: Upper lid raiser → fear, surprise
    'AU5': {'fear': 0.5, 'surprise': 0.6},
    # AU6: Cheek raiser → happy, pain
    'AU6': {'happy': 0.7, 'pain': 0.4},
    # AU7: Lid tightener → angry, fear
    'AU7': {'angry': 0.5, 'fear': 0.4},
    # AU9: Nose wrinkler → (removed: disgust)
    # AU10: Upper lip raiser → happy
    'AU10': {'happy': 0.3},
    # AU12: Lip corner puller → happy
    'AU12': {'happy': 0.9},
    # AU15: Lip corner depressor → sad
    'AU15': {'sad': 0.8},
    # AU17: Chin raiser → sad
    'AU17': {'sad': 0.5},
    # AU23: Lip tightener → angry, fear
    'AU23': {'angry': 0.4, 'fear': 0.3},
    # AU24: Lip pressor → angry
    'AU24': {'angry': 0.6},
    # AU25: Lips part → surprise, happy, fear
    'AU25': {'surprise': 0.6, 'happy': 0.3, 'fear': 0.3},
    # AU26: Jaw drop → surprise, fear
    'AU26': {'surprise': 0.7, 'fear': 0.5},
}


class TemporalBuffer:
    """Temporal buffer for storing AU history."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def add(self, features: np.ndarray):
        self.buffer.append(features)

    def get(self) -> np.ndarray:
        if len(self.buffer) < self.window_size:
            padded = np.zeros((self.window_size, len(self.buffer[0]) if self.buffer else NUM_AUS))
            for i, feat in enumerate(self.buffer):
                padded[i] = feat
            return padded
        return np.array(self.buffer)

    def is_full(self) -> bool:
        return len(self.buffer) >= self.window_size

    def reset(self):
        self.buffer.clear()


class AffectEngineV2:
    """
    AU-First emotion detection engine.

    Pipeline:
    landmarks → AU extraction → rule-based reasoning → feedback-adjusted → EMA → emotion
    """

    def __init__(
        self,
        window_size: int = 10,
        ema_alpha: float = 0.3,
        calibration_frames: int = 30
    ):
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.calibration_frames = calibration_frames

        # Temporal buffer
        self.buffer = TemporalBuffer(window_size)

        # Calibration for normalization
        self.au_baseline = np.zeros(NUM_AUS)  # Neutral baseline
        self.au_range = np.ones(NUM_AUS)       # Max observed range
        self.calibration_count = 0
        self.is_calibrated = False

        # EMA state
        self.ema_probs = None

        # Feedback learner reference (set externally)
        self.feedback_learner = None

        # Landmark indices for AU extraction (MediaPipe FaceMesh 478)
        self._setup_landmarks()

    def _setup_landmarks(self):
        """Setup landmark indices for AU extraction."""
        # MediaPipe FaceMesh landmark indices
        self.LM = {
            # Brows
            'brow_left_inner': 107,
            'brow_left_outer': 337,
            'brow_right_inner': 336,
            'brow_right_outer': 227,
            # Eyes
            'eye_left_inner': 133,
            'eye_left_outer': 33,
            'eye_right_inner': 362,
            'eye_right_outer': 263,
            'eye_left_top': 159,
            'eye_left_bottom': 145,
            'eye_right_top': 386,
            'eye_right_bottom': 374,
            # Nose
            'nose_bridge': 4,
            'nose_tip': 1,
            # Mouth
            'mouth_upper': 13,
            'mouth_lower': 14,
            'mouth_left': 61,
            'mouth_right': 291,
            # Cheeks
            'cheek_left': 116,
            'cheek_right': 345,
            # Jaw
            'jaw_left': 132,
            'jaw_right': 93,
            'chin': 175,
        }

    def _normalize_landmarks(self, landmarks: np.ndarray, frame_size: tuple = None) -> np.ndarray:
        """Normalize landmarks by inter-eye distance."""
        lm = landmarks.copy()

        if frame_size is not None:
            w, h = frame_size
            lm[:, 0] *= w
            lm[:, 1] *= h

        # Inter-eye distance as reference
        left_eye = lm[self.LM['eye_left_inner']]
        right_eye = lm[self.LM['eye_right_inner']]
        eye_dist = np.linalg.norm(left_eye[:2] - right_eye[:2])

        if eye_dist < 1e-6:
            eye_dist = 100  # Fallback

        # Normalize
        norm_lm = lm / eye_dist
        return norm_lm

    def _compute_distance(self, lm: np.ndarray, idx1: int, idx2: int) -> float:
        """Compute normalized distance between two landmarks."""
        return np.linalg.norm(lm[idx1] - lm[idx2])

    def _compute_angle(self, lm: np.ndarray, p1: int, vertex: int, p2: int) -> float:
        """Compute angle at vertex formed by p1-vertex-p2."""
        v1 = lm[p1] - lm[vertex]
        v2 = lm[p2] - lm[vertex]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.clip(cos_angle, -1, 1)

    def extract_au(self, landmarks: np.ndarray, frame_size: tuple = None) -> np.ndarray:
        """
        Extract AU intensities from landmarks.

        Returns:
            au_vector: 15-dimensional AU intensity vector (normalized 0-1)
        """
        lm = self._normalize_landmarks(landmarks, frame_size)

        # Reference distances
        face_width = lm[:, 0].max() - lm[:, 0].min()
        if face_width < 1e-6:
            face_width = 10

        au = np.zeros(NUM_AUS)

        # AU1: Inner brow raiser (distance brow_inner - eye_top relative to face width)
        au[0] = np.clip((lm[self.LM['brow_left_inner']][1] - lm[self.LM['eye_left_top']][1]) / face_width, 0, 1)
        au[0] = np.clip((lm[self.LM['brow_right_inner']][1] - lm[self.LM['eye_right_top']][1]) / face_width, 0, 1)
        au[0] = (au[0] + au[0]) / 2  # Average both sides

        # AU2: Outer brow raiser
        au[1] = np.clip((lm[self.LM['brow_left_outer']][1] - lm[self.LM['eye_left_inner']][1]) / face_width, 0, 1)
        au[1] = np.clip((lm[self.LM['brow_right_outer']][1] - lm[self.LM['eye_right_inner']][1]) / face_width, 0, 1)
        au[1] = (au[1] + au[1]) / 2

        # AU4: Brow lowerer (distance brow - eye)
        brow_eye_left = lm[self.LM['brow_left_inner']][1] - lm[self.LM['eye_left_top']][1]
        brow_eye_right = lm[self.LM['brow_right_inner']][1] - lm[self.LM['eye_right_top']][1]
        au[2] = np.clip((abs(brow_eye_left) + abs(brow_eye_right)) / 2 / face_width * 10, 0, 1)

        # AU5: Upper lid raiser (eye openness)
        eye_open_left = lm[self.LM['eye_left_top']][1] - lm[self.LM['eye_left_bottom']][1]
        eye_open_right = lm[self.LM['eye_right_top']][1] - lm[self.LM['eye_right_bottom']][1]
        eye_height = (abs(eye_open_left) + abs(eye_open_right)) / 2
        eye_width = self._compute_distance(lm, self.LM['eye_left_inner'], self.LM['eye_left_outer'])
        au[3] = np.clip(eye_height / (2 * eye_width + 1e-6), 0, 1)

        # AU6: Cheek raiser (distance cheek - eye corner)
        cheek_eye_left = self._compute_distance(lm, self.LM['cheek_left'], self.LM['eye_left_bottom'])
        cheek_eye_right = self._compute_distance(lm, self.LM['cheek_right'], self.LM['eye_right_bottom'])
        au[4] = np.clip((cheek_eye_left + cheek_eye_right) / 2 / face_width * 5, 0, 1)

        # AU7: Lid tightener (squint)
        eye_width_left = self._compute_distance(lm, self.LM['eye_left_inner'], self.LM['eye_left_outer'])
        eye_height_left = abs(lm[self.LM['eye_left_top']][1] - lm[self.LM['eye_left_bottom']][1])
        squint_left = 1 - eye_height_left / (eye_width_left + 1e-6)
        eye_width_right = self._compute_distance(lm, self.LM['eye_right_inner'], self.LM['eye_right_outer'])
        eye_height_right = abs(lm[self.LM['eye_right_top']][1] - lm[self.LM['eye_right_bottom']][1])
        squint_right = 1 - eye_height_right / (eye_width_right + 1e-6)
        au[5] = np.clip((squint_left + squint_right) / 2, 0, 1)

        # AU9: Nose wrinkler (nose wrinkle intensity)
        nose_bridge = lm[self.LM['nose_bridge']]
        nose_tip = lm[self.LM['nose_tip']]
        nose_up = nose_bridge[1] - nose_tip[1]
        au[6] = np.clip(nose_up / face_width * 10, 0, 1)

        # AU10: Upper lip raiser
        mouth_open = abs(lm[self.LM['mouth_upper']][1] - lm[self.LM['mouth_lower']][1])
        upper_lip_raise = lm[self.LM['nose_tip']][1] - lm[self.LM['mouth_upper']][1]
        au[7] = np.clip(upper_lip_raise / face_width * 10, 0, 1)

        # AU12: Lip corner puller (smile)
        mouth_center_y = (lm[self.LM['mouth_upper']][1] + lm[self.LM['mouth_lower']][1]) / 2
        # In image coords, Y increases downward. Smile = corners go UP (lower Y).
        # So smile value is negative; negate to get positive for happy.
        smile_left = -(lm[self.LM['mouth_left']][1] - mouth_center_y)
        smile_right = -(lm[self.LM['mouth_right']][1] - mouth_center_y)
        mouth_width = self._compute_distance(lm, self.LM['mouth_left'], self.LM['mouth_right'])
        au[8] = np.clip((smile_left + smile_right) / 2 / mouth_width * 5, 0, 1)

        # AU15: Lip corner depressor - sad has corners DOWN (higher Y in image coords)
        # Positive value = corner below center = sad; Negative = corner above center = happy
        mouth_corner_left = lm[self.LM['mouth_left']][1] - mouth_center_y
        mouth_corner_right = lm[self.LM['mouth_right']][1] - mouth_center_y
        au[9] = np.clip((mouth_corner_left + mouth_corner_right) / 2 / face_width * 5, 0, 1)

        # AU17: Chin raiser
        chin_raise = lm[self.LM['chin']][1] - mouth_center_y
        au[10] = np.clip(chin_raise / face_width * 5, 0, 1)

        # AU23: Lip tightener
        lip_tightness = abs(lm[self.LM['mouth_upper']][1] - lm[self.LM['mouth_lower']][1])
        au[11] = np.clip(1 - lip_tightness / face_width * 20, 0, 1)

        # AU24: Lip pressor
        mouth_width_check = self._compute_distance(lm, self.LM['mouth_left'], self.LM['mouth_right'])
        au[12] = np.clip(1 - mouth_width_check / face_width * 2, 0, 1)

        # AU25: Lips part
        au[13] = np.clip(mouth_open / face_width * 10, 0, 1)

        # AU26: Jaw drop
        jaw_drop = lm[self.LM['chin']][1] - lm[self.LM['nose_tip']][1]
        au[14] = np.clip(jaw_drop / face_width * 5, 0, 1)

        return au

    def calibrate(self, au_vector: np.ndarray):
        """Update calibration with AU vector."""
        if self.calibration_count < self.calibration_frames:
            # Accumulate baseline
            if self.calibration_count == 0:
                self.au_baseline = au_vector.copy()
            else:
                self.au_baseline += (au_vector - self.au_baseline) / (self.calibration_count + 1)
            self.calibration_count += 1

            if self.calibration_count >= self.calibration_frames:
                self.is_calibrated = True

    def get_normalized_au(self, au_vector: np.ndarray) -> np.ndarray:
        """Get normalized AU vector relative to baseline."""
        if not self.is_calibrated:
            return au_vector

        # Deviation from baseline
        deviation = au_vector - self.au_baseline

        # Normalize by range
        normalized = np.clip(deviation / (self.au_range + 0.01), -1, 1)
        return normalized

    def rule_based_detection(self, au_norm: np.ndarray) -> np.ndarray:
        """
        Rule-based emotion detection from normalized AU vector.

        Returns:
            probs: 6-dimensional emotion probabilities
        """
        probs = np.zeros(6)

        # Map AU indices to emotion weights
        for au_idx, au_val in enumerate(au_norm):
            if au_val <= 0:
                continue

            au_name = AU_NAMES[au_idx]
            if au_name not in AU_EMOTION_MAP:
                continue

            # Apply AU → emotion mapping
            for emotion, weight in AU_EMOTION_MAP[au_name].items():
                emotion_idx = EMOTION_LABELS.index(emotion)
                probs[emotion_idx] += au_val * weight

        # Strong signal decision tree
        # Happy: AU12 (smile) + AU6 (cheek raiser)
        if au_norm[AUS.index('AU12')] > 0.3 and au_norm[AUS.index('AU6')] > 0.2:
            probs[EMOTION_LABELS.index('happy')] += 0.5

        # Surprise: AU1+AU2 (brows up) + AU26 (jaw drop)
        if au_norm[AUS.index('AU1')] > 0.3 and au_norm[AUS.index('AU26')] > 0.3:
            probs[EMOTION_LABELS.index('surprise')] += 0.5

        # Sad: AU15 (lip corner depressor) + AU4 (brow lowerer)
        if au_norm[AUS.index('AU15')] > 0.3 and au_norm[AUS.index('AU4')] > 0.2:
            probs[EMOTION_LABELS.index('sad')] += 0.4

        # Angry: AU4 (brow lowerer) + AU24 (lip pressor)
        if au_norm[AUS.index('AU4')] > 0.3 and au_norm[AUS.index('AU24')] > 0.3:
            probs[EMOTION_LABELS.index('angry')] += 0.5

        # Fear: AU1+AU2 (brows up) + AU5 (eye wide) + AU26 (jaw drop)
        if au_norm[AUS.index('AU1')] > 0.2 and au_norm[AUS.index('AU5')] > 0.3 and au_norm[AUS.index('AU26')] > 0.3:
            probs[EMOTION_LABELS.index('fear')] += 0.5

        # Normalize
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs[EMOTION_LABELS.index('neutral')] = 1.0

        # --- Neutral Progressive Enhancement ---
        # When max confidence is low, increase neutral probability
        max_prob = probs.max()
        neutral_idx = EMOTION_LABELS.index('neutral')

        if max_prob < 0.2:
            # Very low confidence → neutral dominates
            probs = np.zeros(6)
            probs[neutral_idx] = 1.0
        elif max_prob < 0.6:
            # Low confidence → neutral weighted
            probs[neutral_idx] += 0.3
            probs = probs / probs.sum()

        return probs

    def ema_smooth(self, probs: np.ndarray) -> np.ndarray:
        """EMA smoothing for temporal stability."""
        if self.ema_probs is None:
            self.ema_probs = probs
            return probs

        smoothed = self.ema_alpha * probs + (1 - self.ema_alpha) * self.ema_probs
        self.ema_probs = smoothed
        return smoothed

    def update(self, landmarks: np.ndarray, frame_size: tuple = None) -> Optional[Dict]:
        """Update engine with new landmarks."""
        # Extract AU
        au_raw = self.extract_au(landmarks, frame_size)

        # Calibrate during warmup
        if not self.is_calibrated:
            self.calibrate(au_raw)
            return None

        # Get normalized AU
        au_norm = self.get_normalized_au(au_raw)

        # Apply feedback adjustment if available
        if self.feedback_learner is not None:
            au_norm = self.feedback_learner.get_adjusted_au(au_norm)

        # Add to buffer
        self.buffer.add(au_norm)

        # Rule-based detection
        probs = self.rule_based_detection(au_norm)

        # EMA smoothing
        smoothed = self.ema_smooth(probs)

        return {
            'au_raw': au_raw,
            'au_norm': au_norm,
            'probabilities': smoothed
        }

    def predict(self, landmarks: np.ndarray, frame_size: tuple = None) -> Dict:
        """Main prediction method."""
        result = self.update(landmarks, frame_size)

        if result is None:
            return {
                'emotion': 'calibrating',
                'emotion_id': -1,
                'confidence': 0.0,
                'probabilities': np.zeros(6),
                'au_raw': np.zeros(NUM_AUS),
                'au_norm': np.zeros(NUM_AUS)
            }

        probs = result['probabilities']
        neutral_idx = EMOTION_LABELS.index('neutral')

        # --- Display Layer: Neutral Override ---
        # When neutral probability exceeds threshold, emit 'neutral' even if
        # other emotions have slight probability contributions from AU noise.
        if probs[neutral_idx] > 0.4:
            emotion_id = neutral_idx
            confidence = probs[neutral_idx]
        else:
            emotion_id = np.argmax(probs)
            confidence = probs[emotion_id]

        return {
            'emotion': EMOTION_LABELS[emotion_id],
            'emotion_id': emotion_id,
            'confidence': confidence,
            'probabilities': probs,
            'au_raw': result['au_raw'],
            'au_norm': result['au_norm']
        }

    def reset(self):
        """Reset engine state."""
        self.buffer.reset()
        self.au_baseline = np.zeros(NUM_AUS)
        self.au_range = np.ones(NUM_AUS)
        self.calibration_count = 0
        self.is_calibrated = False
        self.ema_probs = None


# Quick reference
AUS = AU_NAMES