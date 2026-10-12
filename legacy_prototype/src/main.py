#!/usr/bin/env python3
"""
Bangbang V2 - Main entry point for empathy-driven stationary assistant.

This module orchestrates concurrent vision and speech processing,
integrates distress detection and empathetic feedback.

Archived prototype: threshold labels and template responses are for operator review.
They are not validated measurements of pain, clinical advice, or autonomous actions.
Historical target: end-to-end latency <800 ms on Raspberry Pi 5 (not reproduced here).
Stationary deployment (no motors/mobility hardware).
"""

import threading
import queue
import time
import logging
import signal
import sys
import os
from pathlib import Path
from typing import Optional, Any

import cv2
import numpy as np
import yaml

# Try importing optional dependencies
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
CONFIG_PATH = Path(os.environ.get(
    "BANGBANG_CONFIG",
    str(Path(__file__).parent.parent / "config" / "default.example.yaml"),
))


def load_config() -> dict:
    """Load configuration from YAML file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f) or {}
        model_path = os.environ.get("BANGBANG_VOSK_MODEL_PATH")
        if model_path:
            config.setdefault("audio", {}).setdefault("vosk", {})["model_path"] = model_path
        return config
    return {}


CONFIG = load_config()

LOG_LEVEL = getattr(logging, CONFIG.get("logging", {}).get("level", "INFO"), logging.INFO)
VISION_FPS_TARGET = CONFIG.get("performance", {}).get("vision_fps_target", 30)
AUDIO_CHUNK_MS = CONFIG.get("performance", {}).get("audio_chunk_ms", 3000)
PAIN_LEVEL_UPDATE_INTERVAL = CONFIG.get("performance", {}).get("update_interval", 0.5)

# Vision config
VISION_CONFIG = CONFIG.get("vision", {})
EAR_THRESHOLD = VISION_CONFIG.get("distress", {}).get("low_pain_ear", 0.25)
NASOLABIAL_THRESHOLD = VISION_CONFIG.get("distress", {}).get("low_nasolabial", -0.05)

# Audio config
AUDIO_CONFIG = CONFIG.get("audio", {})
FUZZY_CONFIG = AUDIO_CONFIG.get("fuzzy", {})
FUZZY_DICTIONARY = FUZZY_CONFIG.get("dictionary", ["pain", "help", "water"])
MAX_EDIT_DISTANCE = FUZZY_CONFIG.get("max_edit_distance", 2)

# Empathy config
EMPATHY_CONFIG = CONFIG.get("empathy", {})
RESPONSES = EMPATHY_CONFIG.get("responses", {})

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Utility: Levenshtein Distance for Fuzzy Matching
# ----------------------------------------------------------------------
def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two strings.

    Time complexity: O(m*n) where m,n are string lengths
    Space complexity: O(min(m,n)) using rolling array optimization
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def fuzzy_match(transcript: str, dictionary: list[str], max_distance: int = 2) -> tuple[Optional[str], float]:
    """
    Find closest match in dictionary within edit distance threshold.

    Args:
        transcript: The transcribed text to match
        dictionary: List of valid words/phrases
        max_distance: Maximum allowed edit distance

    Returns:
        Tuple of (matched_word, confidence) or (None, 0.0) if no match
    """
    transcript_lower = transcript.lower().strip()
    best_match: Optional[str] = None
    best_distance = max_distance + 1

    for word in dictionary:
        distance = levenshtein_distance(transcript_lower, word.lower())
        if distance < best_distance:
            best_distance = distance
            best_match = word

    if best_match:
        max_len = max(len(transcript), len(best_match))
        confidence = 1.0 - (best_distance / max_len) if max_len > 0 else 0.0
        return best_match, confidence

    return None, 0.0


# ----------------------------------------------------------------------
# Vision Processor
# ----------------------------------------------------------------------
class VisionProcessor:
    """Process camera frames to compute distress metrics using MediaPipe Face Mesh."""

    # Landmark indices for EAR computation
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    # Landmark indices for nasolabial fold
    NASOLABIAL = [61, 291, 39, 269]

    # Landmark indices for eyebrow furrow
    EYEBROW = [65, 158, 295]

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.face_mesh = None
        self.face_mesh_config = self.config.get("face_mesh", {})
        self.ear_threshold = self.config.get("distress", {}).get("low_pain_ear", 0.25)
        self.nasolabial_threshold = self.config.get("distress", {}).get("low_nasolabial", -0.05)
        self.eye_close_start_time: Optional[float] = None
        self._initialized = False

    def _initialize(self) -> bool:
        """Initialize MediaPipe Face Mesh."""
        if not MEDIAPIPE_AVAILABLE:
            logger.warning("MediaPipe not available, vision processing disabled")
            return False

        try:
            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=self.face_mesh_config.get("static_image_mode", False),
                max_num_faces=self.face_mesh_config.get("max_num_faces", 1),
                refine_landmarks=self.face_mesh_config.get("refine_landmarks", True),
                min_detection_confidence=self.face_mesh_config.get("min_detection_confidence", 0.5),
                min_tracking_confidence=self.face_mesh_config.get("min_tracking_confidence", 0.5),
            )
            self._initialized = True
            logger.info("VisionProcessor initialized with MediaPipe Face Mesh")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Face Mesh: {e}")
            return False

    def _compute_ear(self, landmarks: list, eye_indices: list) -> float:
        """
        Compute Eye Aspect Ratio from eye landmarks.

        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

        Where p1-p6 are landmark points of the eye.
        """
        points = [landmarks[i] for i in eye_indices]
        if len(points) < 6:
            return 1.0

        # Vertical distances
        p1, p2, p3, p4, p5, p6 = points[:6]

        def euclidean_distance(a, b):
            return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

        vertical_1 = euclidean_distance(p2, p6)
        vertical_2 = euclidean_distance(p3, p5)
        horizontal = euclidean_distance(p1, p4)

        if horizontal == 0:
            return 1.0

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    def _compute_nasolabial_shift(self, landmarks: list) -> float:
        """
        Compute nasolabial fold vertical displacement.

        Negative shift indicates downward movement (frown/pain).
        Normalized to face bounding box scale.
        """
        if len(landmarks) < max(self.NASOLABIAL) + 1:
            return 0.0

        # Get y-coordinates of nasolabial landmarks
        nasolabial_points = [landmarks[i] for i in self.NASOLABIAL]

        # Compute average vertical position
        avg_y = np.mean([p.y for p in nasolabial_points])

        # Normalize relative to face height (approximate from landmark range)
        face_height = max(p.y for p in landmarks) - min(p.y for p in landmarks)
        if face_height > 0:
            return (avg_y - 0.5) / face_height  # Centered around 0

        return 0.0

    def _compute_eyebrow_lower(self, landmarks: list, face_height: float) -> float:
        """
        Detect eyebrow lowering (furrow).

        Returns normalized decrease in eyebrow-eye distance.
        """
        if len(landmarks) < max(self.EYEBROW + self.RIGHT_EYE) + 1:
            return 0.0

        # Average eyebrow y-position
        eyebrow_y = np.mean([landmarks[i].y for i in self.EYEBROW])

        # Average eye top y-position (lower is larger y)
        eye_y = np.mean([landmarks[i].y for i in self.RIGHT_EYE[:3]])

        if face_height > 0:
            return (eye_y - eyebrow_y) / face_height

        return 0.0

    def _classify_pain_level(self, ear: float, nasolabial: float, eyebrow: float) -> int:
        """
        Classify pain level based on facial metrics.

        Returns:
            0 = no pain, 1 = low, 2 = medium, 3 = high
        """
        distress = self.config.get("distress", {})
        high_ear = distress.get("high_pain_ear", 0.15)
        high_nasolabial = distress.get("high_nasolabial", -0.15)

        # High pain: EAR very low AND significant nasolabial downshift
        if ear < high_ear and nasolabial < high_nasolabial and eyebrow > 0.05:
            return 3

        # Medium pain: EAR low AND nasolabial downshift
        medium_ear = distress.get("medium_pain_ear", 0.20)
        medium_nasolabial = distress.get("medium_nasolabial", -0.10)
        if ear < medium_ear and nasolabial < medium_nasolabial:
            return 2

        # Low pain: EAR below threshold OR nasolabial downshift
        if ear < self.ear_threshold or nasolabial < self.nasolabial_threshold:
            return 1

        return 0

    def process_frame(self, frame: Any) -> dict:
        """
        Process a single frame and return distress metrics.

        Returns dict with:
            - pain_level: 0-3 (none, low, medium, high)
            - attention: bool (user looking at camera)
            - ear: float (eye aspect ratio)
            - nasolabial_shift: float
            - landmarks_detected: bool
        """
        if not self._initialized:
            if not self._initialize():
                return {
                    "pain_level": 0,
                    "attention": False,
                    "ear": 1.0,
                    "nasolabial_shift": 0.0,
                    "landmarks_detected": False,
                }

        if frame is None:
            return {
                "pain_level": 0,
                "attention": False,
                "ear": 1.0,
                "nasolabial_shift": 0.0,
                "landmarks_detected": False,
            }

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return {
                "pain_level": 0,
                "attention": False,
                "ear": 1.0,
                "nasolabial_shift": 0.0,
                "landmarks_detected": False,
            }

        # Process first face only
        landmarks = results.multi_face_landmarks[0].landmark
        num_landmarks = len(landmarks)

        if num_landmarks < max(self.LEFT_EYE + self.RIGHT_EYE + self.NASOLABIAL) + 1:
            return {
                "pain_level": 0,
                "attention": False,
                "ear": 1.0,
                "nasolabial_shift": 0.0,
                "landmarks_detected": False,
            }

        # Compute metrics
        left_ear = self._compute_ear(landmarks, self.LEFT_EYE)
        right_ear = self._compute_ear(landmarks, self.RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        nasolabial_shift = self._compute_nasolabial_shift(landmarks)

        face_height = max(l.y for l in landmarks) - min(l.y for l in landmarks)
        eyebrow_lower = self._compute_eyebrow_lower(landmarks, face_height)

        # Classify pain
        pain_level = self._classify_pain_level(avg_ear, nasolabial_shift, eyebrow_lower)

        # Attention detection (simplified: face centered and upright)
        # In production, use head pose estimation
        attention = True

        return {
            "pain_level": pain_level,
            "attention": attention,
            "ear": avg_ear,
            "nasolabial_shift": nasolabial_shift,
            "eyebrow_lower": eyebrow_lower,
            "landmarks_detected": True,
            "num_landmarks": num_landmarks,
        }


# ----------------------------------------------------------------------
# Audio Processor
# ----------------------------------------------------------------------
class AudioProcessor:
    """Process microphone audio to transcribe speech and detect distress keywords."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.vosk_model = None
        self.vosk_recognizer = None
        self.stream = None
        self.pyaudio = None
        self.fuzzy_dictionary = self.config.get("fuzzy", {}).get("dictionary", FUZZY_DICTIONARY)
        self.max_edit_distance = self.config.get("fuzzy", {}).get("max_edit_distance", MAX_EDIT_DISTANCE)
        self.vosk_config = self.config.get("vosk", {})
        self.mic_config = self.config.get("microphone", {})
        self._initialized = False

    def _initialize(self) -> bool:
        """Initialize Vosk STT and audio capture."""
        if not VOSK_AVAILABLE or not PYAUDIO_AVAILABLE:
            logger.warning("Vosk or PyAudio not available, audio processing disabled")
            return False

        try:
            # Load Vosk model
            model_path = self.vosk_config.get("model_path", "models/vosk-model-small-en-us-0.15")
            if Path(model_path).exists():
                self.vosk_model = vosk.Model(str(model_path))
                sample_rate = self.vosk_config.get("sample_rate", 16000)
                self.vosk_recognizer = vosk.KaldiRecognizer(self.vosk_model, sample_rate)
                logger.info(f"Vosk model loaded from {model_path}")
            else:
                logger.warning(f"Vosk model not found at {model_path}, audio STT disabled")
                return False

            # Initialize PyAudio
            self.pyaudio = pyaudio.PyAudio()
            chunk_size = self.mic_config.get("chunk_size", 2048)
            sample_rate = self.mic_config.get("sample_rate", 16000)

            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size,
            )
            logger.info("AudioProcessor initialized with Vosk STT")
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize AudioProcessor: {e}")
            return False

    def _detect_speech_acts(self, transcript: str) -> list[str]:
        """Detect speech acts from transcript keywords."""
        speech_acts_config = self.config.get("speech_acts", {})
        transcript_lower = transcript.lower()

        detected_acts = []
        for act, keywords in speech_acts_config.items():
            for keyword in keywords:
                if keyword in transcript_lower:
                    detected_acts.append(act)
                    break

        return detected_acts

    def process_audio(self, audio_chunk: bytes | None = None) -> dict:
        """
        Process audio chunk and return transcription with distress detection.

        Returns dict with:
            - transcript: str (recognized text)
            - keywords: list[str] (matched distress keywords)
            - speech_acts: list[str] (detected speech acts)
            - confidence: float
            - fuzzy_matches: dict[str, float] (word -> confidence)
        """
        if not self._initialized:
            if not self._initialize():
                return {
                    "transcript": "",
                    "keywords": [],
                    "speech_acts": [],
                    "confidence": 0.0,
                    "fuzzy_matches": {},
                }

        if audio_chunk is None or len(audio_chunk) == 0:
            return {
                "transcript": "",
                "keywords": [],
                "speech_acts": [],
                "confidence": 0.0,
                "fuzzy_matches": {},
            }

        # Process with Vosk
        if self.vosk_recognizer.AcceptWaveform(audio_chunk):
            result = self.vosk_recognizer.Result()
            import json
            result_dict = json.loads(result)
            transcript = result_dict.get("text", "")
        else:
            # Partial result
            partial = self.vosk_recognizer.PartialResult()
            import json
            partial_dict = json.loads(partial)
            transcript = partial_dict.get("partial", "")

        if not transcript:
            return {
                "transcript": "",
                "keywords": [],
                "speech_acts": [],
                "confidence": 0.0,
                "fuzzy_matches": {},
            }

        # Fuzzy matching against distress dictionary
        fuzzy_matches = {}
        keywords = []
        words = transcript.split()

        for word in words:
            match, confidence = fuzzy_match(word, self.fuzzy_dictionary, self.max_edit_distance)
            if match and confidence > 0.5:
                fuzzy_matches[word] = {"matched": match, "confidence": confidence}
                if match not in keywords:
                    keywords.append(match)

        # Speech act detection
        speech_acts = self._detect_speech_acts(transcript)

        # Average confidence
        confidence = np.mean([v["confidence"] for v in fuzzy_matches.values()]) if fuzzy_matches else 0.5

        return {
            "transcript": transcript,
            "keywords": keywords,
            "speech_acts": speech_acts,
            "confidence": confidence,
            "fuzzy_matches": fuzzy_matches,
        }

    def read_chunk(self) -> bytes | None:
        """Read next audio chunk from microphone."""
        if not self._initialized:
            return None

        if self.stream is None:
            return None

        try:
            chunk_size = self.mic_config.get("chunk_size", 2048)
            data = self.stream.read(chunk_size, exception_on_overflow=False)
            return data
        except Exception as e:
            logger.debug(f"Audio read error: {e}")
            return None

    def cleanup(self):
        """Release audio resources."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.pyaudio:
            self.pyaudio.terminate()


# ----------------------------------------------------------------------
# Empathy Engine
# ----------------------------------------------------------------------
class EmpathyEngine:
    """Fuse vision and audio results to generate empathetic responses."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.responses = self.config.get("responses", RESPONSES)
        self.urgency_config = self.config.get("urgency", {})
        self.last_pain_level = 0
        self.last_keywords: list[str] = []
        self.last_speech_acts: list[str] = []

    def _get_urgency(self, pain_level: int, speech_acts: list[str]) -> int:
        """Determine urgency level from pain and speech."""
        if "emergency" in speech_acts:
            return self.urgency_config.get("critical", 4)
        if pain_level >= 3:
            return self.urgency_config.get("high", 3)
        if pain_level >= 2 or "emotional_expression" in speech_acts:
            return self.urgency_config.get("medium", 2)
        if pain_level >= 1 or "direct_request" in speech_acts:
            return self.urgency_config.get("low", 1)
        return self.urgency_config.get("none", 0)

    def _select_response(self, pain_level: int, speech_acts: list[str]) -> str:
        """Select appropriate empathetic response."""
        # Emergency/high pain
        if pain_level >= 3:
            return self.responses.get("pain_high", "I detect significant distress. I'm calling for help immediately.")

        # Speech-act based responses
        if "greeting" in speech_acts and pain_level == 0:
            return self.responses.get("greeting", "Hello! I'm here to help. How are you feeling?")

        if "emotional_expression" in speech_acts:
            if pain_level >= 2:
                return self.responses.get("pain_medium", "You seem to be in pain. Would you like me to call someone?")
            return self.responses.get("emotional_distress", "I'm here for you. Would you like to talk about how you're feeling?")

        if "direct_request" in speech_acts:
            return self.responses.get("request", "I'll do my best to help. Let me see what I can do.")

        if "question" in speech_acts:
            return self.responses.get("question", "That's a good question. Let me help you find the answer.")

        # Pain-level based responses
        if pain_level >= 2:
            return self.responses.get("pain_medium", "You seem to be in pain. Would you like me to call someone?")
        if pain_level >= 1:
            return self.responses.get("pain_low", "I notice you might be uncomfortable. Can I get you something?")

        return self.responses.get("unknown", "I'm here to help. Please tell me what you need.")

    def decide_response(self, vision_result: dict, audio_result: dict) -> dict:
        """
        Fuse vision and audio results to generate empathetic response.

        Returns dict with:
            - verbal: str (response text)
            - urgency: int (0-4)
            - pain_level: int (0-3)
            - detected_keywords: list[str]
        """
        # Extract metrics
        pain_level = vision_result.get("pain_level", 0)
        keywords = audio_result.get("keywords", [])
        speech_acts = audio_result.get("speech_acts", [])

        # Update state
        self.last_pain_level = pain_level
        self.last_keywords = keywords
        self.last_speech_acts = speech_acts

        # Determine urgency
        urgency = self._get_urgency(pain_level, speech_acts)

        # Select response
        verbal = self._select_response(pain_level, speech_acts)

        return {
            "verbal": verbal,
            "urgency": urgency,
            "pain_level": pain_level,
            "detected_keywords": keywords,
            "speech_acts": speech_acts,
        }


# ----------------------------------------------------------------------
# Thread workers
# ----------------------------------------------------------------------
class VisionThread(threading.Thread):
    """Continuously captures frames and runs distress detection."""

    def __init__(
        self,
        frame_queue: queue.Queue,
        result_queue: queue.Queue,
        processor: VisionProcessor,
        config: dict | None = None,
    ):
        super().__init__(name="VisionThread")
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.processor = processor
        self.config = config or {}
        self.running = True
        self.cap: Any = None
        self.simulate = self.config.get("system", {}).get("simulate_hardware", False)

    def _initialize_camera(self) -> bool:
        """Initialize camera capture using OpenCV (V4L2/libcamera compatibility)."""
        try:
            # Try to open default camera
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                # Set resolution
                width = self.config.get("hardware", {}).get("camera", {}).get("width", 1920)
                height = self.config.get("hardware", {}).get("camera", {}).get("height", 1080)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                logger.info("Camera initialized successfully")
                return True
            else:
                logger.warning("Could not open camera, running in simulation mode")
                self.cap = None
                return False
        except Exception as e:
            logger.warning(f"Camera initialization failed: {e}, running in simulation mode")
            self.cap = None
            return False

    def run(self):
        """Main loop: capture frames and process distress detection."""
        logger.info("Vision thread starting")
        self._initialize_camera()

        frame_interval = 1.0 / VISION_FPS_TARGET
        last_frame_time = time.time()

        while self.running:
            start_time = time.time()

            # Capture frame
            frame = None
            if self.cap is not None and not self.simulate:
                ret, frame = self.cap.read()
                if not ret:
                    frame = None
                    logger.warning("Failed to capture frame")

            # Process frame
            result = self.processor.process_frame(frame)

            # Add timing info
            result["capture_time"] = time.time() - start_time

            # Put result in queue
            try:
                self.result_queue.put_nowait(("vision", result))
            except queue.Full:
                logger.debug("Vision result queue full, dropping frame")

            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Vision thread stopped")

    def stop(self):
        """Stop the thread."""
        self.running = False
        if self.cap:
            self.cap.release()


class AudioThread(threading.Thread):
    """Continuously captures audio and runs speech recognition."""

    def __init__(
        self,
        audio_queue: queue.Queue,
        result_queue: queue.Queue,
        processor: AudioProcessor,
        config: dict | None = None,
    ):
        super().__init__(name="AudioThread")
        self.audio_queue = audio_queue
        self.result_queue = result_queue
        self.processor = processor
        self.config = config or {}
        self.running = True

    def run(self):
        """Main loop: capture audio chunks and process STT."""
        logger.info("Audio thread starting")

        chunk_size = self.config.get("hardware", {}).get("microphone", {}).get("chunk_size", 2048)
        sample_rate = self.config.get("hardware", {}).get("microphone", {}).get("sample_rate", 16000)

        # Time per chunk in seconds
        chunk_duration = chunk_size / sample_rate

        while self.running:
            start_time = time.time()

            # Read audio chunk
            audio_data = self.processor.read_chunk()

            if audio_data:
                # Process audio
                result = self.processor.process_audio(audio_data)

                # Add timing info
                result["capture_time"] = time.time() - start_time

                # Put result in queue
                try:
                    self.result_queue.put_nowait(("audio", result))
                except queue.Full:
                    logger.debug("Audio result queue full, dropping chunk")

            # Sleep for chunk duration
            elapsed = time.time() - start_time
            sleep_time = chunk_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Audio thread stopped")

    def stop(self):
        """Stop the thread."""
        self.running = False
        self.processor.cleanup()


# ----------------------------------------------------------------------
# Main coordinator
# ----------------------------------------------------------------------
class BangbangController:
    """Orchestrates vision/audio threads and decides empathetic responses."""

    def __init__(self, config: dict | None = None):
        self.config = config or CONFIG

        # Queues
        self.frame_queue = queue.Queue(maxsize=5)
        self.audio_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=20)

        # Processors
        vision_config = self.config.get("vision", {})
        audio_config = self.config.get("audio", {})
        self.vision_processor = VisionProcessor(vision_config)
        self.audio_processor = AudioProcessor(audio_config)
        self.empathy_engine = EmpathyEngine(self.config.get("empathy", {}))

        # Threads
        self.vision_thread: Optional[VisionThread] = None
        self.audio_thread: Optional[AudioThread] = None

        # State
        self.running = False
        self.last_pain_update = time.time()
        self.last_response_time = time.time()
        self.current_response: Optional[dict] = None

        # Performance tracking
        self.latency_samples: list[float] = []
        self.max_latency_samples = 100

    def start(self):
        """Start all threads and begin main loop."""
        logger.info("Starting Bangbang V2 controller")
        self.running = True

        # Initialize and start threads
        self.vision_thread = VisionThread(
            self.frame_queue,
            self.result_queue,
            self.vision_processor,
            self.config,
        )
        self.audio_thread = AudioThread(
            self.audio_queue,
            self.result_queue,
            self.audio_processor,
            self.config,
        )

        self.vision_thread.start()
        self.audio_thread.start()

        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop()

    def _main_loop(self):
        """Process results from vision/audio and generate responses."""
        last_vision_result: dict = {}
        last_audio_result: dict = {}

        while self.running:
            try:
                # Get result with timeout
                source, result = self.result_queue.get(timeout=0.1)

                if source == "vision":
                    last_vision_result = result
                    self._handle_vision_result(result)
                elif source == "audio":
                    last_audio_result = result
                    self._handle_audio_result(result)

                    # Trigger empathy fusion on audio (user speech)
                    if result.get("transcript"):
                        response = self.empathy_engine.decide_response(
                            last_vision_result, last_audio_result
                        )
                        self._handle_response(response)

            except queue.Empty:
                pass

            # Periodic empathy update
            now = time.time()
            if now - self.last_pain_update >= PAIN_LEVEL_UPDATE_INTERVAL:
                # Check if pain level changed significantly
                vision_pain = last_vision_result.get("pain_level", 0)
                if vision_pain > 0:
                    response = self.empathy_engine.decide_response(
                        last_vision_result, last_audio_result
                    )
                    if response.get("urgency", 0) > 1:
                        self._handle_response(response)
                self.last_pain_update = now

    def _handle_vision_result(self, result: dict):
        """Log vision result for debugging."""
        logger.debug(
            f"Vision: pain={result.get('pain_level')}, "
            f"EAR={result.get('ear', 0):.3f}, "
            f"landmarks={result.get('landmarks_detected')}"
        )

    def _handle_audio_result(self, result: dict):
        """Log and process audio result."""
        transcript = result.get("transcript", "")
        keywords = result.get("keywords", [])

        if transcript:
            logger.info(f"Audio: '{transcript}' (conf={result.get('confidence', 0):.2f})")

        if keywords:
            logger.info(f"Detected keywords: {keywords}")

    def _handle_response(self, response: dict):
        """Process and display empathetic response."""
        self.current_response = response
        self.last_response_time = time.time()

        verbal = response.get("verbal", "")
        urgency = response.get("urgency", 0)
        pain_level = response.get("pain_level", 0)

        logger.info(
            f"Response [urgency={urgency}, pain={pain_level}]: {verbal}"
        )

        # Log for caregiver review (no raw audio/video)
        self._log_interaction(response)

    def _log_interaction(self, response: dict):
        """Log interaction for caregiver review."""
        # In production, write to structured log file
        log_entry = {
            "timestamp": time.time(),
            "pain_level": response.get("pain_level"),
            "urgency": response.get("urgency"),
            "keywords": response.get("detected_keywords"),
            "speech_acts": response.get("speech_acts"),
            "response": response.get("verbal"),
        }
        logger.debug(f"Interaction log: {log_entry}")

    def stop(self):
        """Gracefully stop all threads."""
        logger.info("Stopping Bangbang V2 controller")
        self.running = False

        if self.vision_thread:
            self.vision_thread.stop()
            self.vision_thread.join(timeout=2.0)

        if self.audio_thread:
            self.audio_thread.stop()
            self.audio_thread.join(timeout=2.0)

        logger.info("All threads stopped")

        # Log performance summary
        if self.latency_samples:
            avg_latency = sum(self.latency_samples) / len(self.latency_samples)
            max_latency = max(self.latency_samples)
            logger.info(f"Performance: avg={avg_latency*1000:.1f}ms, max={max_latency*1000:.1f}ms")


# ----------------------------------------------------------------------
# Signal handling for graceful shutdown
# ----------------------------------------------------------------------
def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down")
    sys.exit(0)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Bangbang V2 - Sentient Empathic Assistant")
    logger.info(f"Config: {CONFIG_PATH}")

    controller = BangbangController(CONFIG)
    controller.start()


if __name__ == "__main__":
    main()
