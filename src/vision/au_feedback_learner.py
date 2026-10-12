"""
AU Feedback Learner
Adjusts AU weights and thresholds based on user corrections.
"""

import numpy as np
from typing import Optional


class AUFeedbackLearner:
    """
    Feedback-based AU weight learner.

    When user corrects the emotion, this learner adjusts AU weights
    to improve accuracy over time.
    """

    def __init__(self, num_aus: int = 15):
        self.num_aus = num_aus

        # AU weights (enhance or suppress specific AUs)
        self.au_weights = np.ones(num_aus)

        # AU thresholds (bias for specific emotions)
        self.au_thresholds = np.zeros(num_aus)

        # Feedback history for learning rate adaptation
        self.feedback_count = 0
        self.cumulative_corrections = np.zeros(num_aus)

        # Emotion-specific AU importance
        self.emotion_au_importance = {
            'angry': ['AU4', 'AU7', 'AU24'],
            'disgust': ['AU9', 'AU10', 'AU17'],
            'fear': ['AU1', 'AU2', 'AU5', 'AU26'],
            'happy': ['AU6', 'AU12', 'AU25'],
            'neutral': [],
            'sad': ['AU1', 'AU4', 'AU15', 'AU17'],
            'surprise': ['AU1', 'AU2', 'AU5', 'AU26'],
        }

    def _get_au_index(self, au_name: str) -> int:
        """Get AU index from name."""
        from .affect_engine_v2 import AU_NAMES
        return AU_NAMES.index(au_name)

    def receive_feedback(self, true_emotion: str, au_vector: np.ndarray):
        """
        Receive feedback correction from user.

        Args:
            true_emotion: The correct emotion label
            au_vector: Current AU intensity vector (15,)
        """
        if true_emotion not in self.emotion_au_importance:
            return

        self.feedback_count += 1

        # Get relevant AUs for this emotion
        relevant_aus = self.emotion_au_importance[true_emotion]

        # Learning rate (decreases with more feedback)
        lr = 1.0 / (1 + self.feedback_count * 0.1)

        for i in range(self.num_aus):
            au_name = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9',
                      'AU10', 'AU12', 'AU15', 'AU17', 'AU23', 'AU24', 'AU25', 'AU26'][i]

            if au_name in relevant_aus:
                # Boost AU weight (positive correlation)
                self.cumulative_corrections[i] += au_vector[i] * lr
                self.au_weights[i] = 1.0 + self.cumulative_corrections[i]
            else:
                # Reduce weight slightly
                self.cumulative_corrections[i] -= au_vector[i] * lr * 0.1
                self.au_weights[i] = 1.0 + self.cumulative_corrections[i]

        # Clip weights to reasonable range
        self.au_weights = np.clip(self.au_weights, 0.2, 3.0)

    def get_adjusted_au(self, au_vector: np.ndarray) -> np.ndarray:
        """
        Apply feedback adjustments to AU vector.

        Args:
            au_vector: Raw AU intensity vector (15,)

        Returns:
            Adjusted AU vector
        """
        # Apply weights
        adjusted = au_vector * self.au_weights

        # Apply thresholds
        adjusted = adjusted + self.au_thresholds

        # Normalize to 0-1
        adjusted = np.clip(adjusted, 0, 1)

        return adjusted

    def reset(self):
        """Reset learned weights."""
        self.au_weights = np.ones(self.num_aus)
        self.au_thresholds = np.zeros(self.num_aus)
        self.feedback_count = 0
        self.cumulative_corrections = np.zeros(self.num_aus)

    def save(self, path: str):
        """Save weights to file."""
        np.savez(
            path,
            weights=self.au_weights,
            thresholds=self.au_thresholds,
            feedback_count=self.feedback_count,
            corrections=self.cumulative_corrections
        )

    def load(self, path: str):
        """Load weights from file."""
        data = np.load(path)
        self.au_weights = data['weights']
        self.au_thresholds = data['thresholds']
        self.feedback_count = int(data['feedback_count'])
        self.cumulative_corrections = data['corrections']