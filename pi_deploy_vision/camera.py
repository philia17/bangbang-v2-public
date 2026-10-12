"""
Shared camera module for Pi 5 libcamera architecture.

Provides CameraReader class with automatic camera probing,
warm-up waiting, and robust frame reading with failure tracking.
"""

import cv2
import os
import time
import numpy as np


class CameraReader:
    """Unified camera reader with auto-probe, warm-up, and failure handling."""

    def __init__(self):
        self.cap = None
        self._consecutive_failures = 0
        self._max_failures = 30
        self._img_h = 0
        self._img_w = 0
        self._probe()

    def _probe(self):
        """Try multiple camera indices to find a working one.

        Supports VIDEO_DEVICE env var for custom device path/index.
        Tries [0, 1, 4, 14] by default for Pi 5 compatibility.
        """
        candidates = [0, 1, 4, 14]
        video_device = os.environ.get('VIDEO_DEVICE', None)

        if video_device is not None:
            try:
                idx = int(video_device)
                candidates = [idx] + [c for c in candidates if c != idx]
            except ValueError:
                cap = cv2.VideoCapture(video_device)
                if cap.isOpened():
                    print(f"Camera opened: {video_device}")
                    self.cap = cap
                    self._read_dims()
                    self._warm_up()
                    return
                print(f"Cannot open video device: {video_device}")
                return

        for idx in candidates:
            print(f"Attempting to open camera index={idx}...")
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print("Camera hardware connected. Waiting for ISP buffer to warm up...")
                self.cap = cap
                self._read_dims()
                if self._warm_up():
                    return
                cap.release()
                self.cap = None
            else:
                print(f"Cannot open camera index={idx}")

        print("ERROR: No working camera found (tried indices: {})".format(candidates))

    def _read_dims(self):
        """Read expected frame dimensions from camera properties."""
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w > 0 and h > 0:
            self._img_w, self._img_h = w, h
        else:
            self._img_w, self._img_h = 640, 480

    def _reshape(self, frame):
        """Fix libcamera flat-array quirk on Pi 5.

        libcamera via V4L2 may return raw bytes as (1, W*H*C) instead
        of a properly shaped (H, W, C) numpy array. Reshape if needed.
        """
        try:
            if frame.ndim == 2 and frame.shape[0] == 1:
                n_bytes = frame.shape[1]
                # 921600 = 640*480*3 → BGR/RGB
                if n_bytes == self._img_w * self._img_h * 3:
                    frame = frame.reshape(self._img_h, self._img_w, 3)
        except Exception:
            pass
        return frame

    def _warm_up(self):
        """Warm up camera by reading frames until a valid one is captured.

        Returns True if a valid frame was captured, False otherwise.
        """
        for attempt in range(50):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = self._reshape(frame)
                if self._validate(frame):
                    print(f"Success! Captured first frame on attempt {attempt}.")
                    return True
            time.sleep(0.1)

        print("Gave up waiting for frames after 50 attempts.")
        return False

    @staticmethod
    def _validate(frame):
        """Check if frame is a usable image array."""
        if frame is None:
            return False
        if frame.size == 0:
            return False
        if frame.ndim < 2:
            return False
        if frame.shape[0] < 10 or frame.shape[1] < 10:
            return False
        return True

    def read(self):
        """Read a frame from the camera with failure tracking.

        Returns:
            (ret, frame) tuple. ret is True on success, False on failure.
            After max_failures consecutive failures, failed() returns True.
        """
        try:
            ret, frame = self.cap.read()
        except Exception as e:
            print(f"ERROR reading camera: {e}")
            self._consecutive_failures += 1
            time.sleep(0.1)
            return (False, None)

        if not ret:
            self._consecutive_failures += 1
            if self._consecutive_failures == 1:
                print("WARNING: Camera read failed, retrying...")
            if self._consecutive_failures > self._max_failures:
                print(f"ERROR: Camera read failed {self._max_failures} times, exiting")
            time.sleep(0.1)
            return (False, None)

        frame = self._reshape(frame)

        if not self._validate(frame):
            self._consecutive_failures += 1
            if self._consecutive_failures == 1:
                print("WARNING: Camera read failed (bad frame), retrying...")
            if self._consecutive_failures > self._max_failures:
                print(f"ERROR: Camera read failed {self._max_failures} times, exiting")
            time.sleep(0.1)
            return (False, None)

        self._consecutive_failures = 0
        return (True, frame)

    def failed(self):
        """Check if camera has exceeded max consecutive failures.

        Callers should check this after each read() and break
        their main loop when True.
        """
        return self._consecutive_failures > self._max_failures

    def release(self):
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_opened(self):
        """Check if camera is successfully opened."""
        return self.cap is not None and self.cap.isOpened()
