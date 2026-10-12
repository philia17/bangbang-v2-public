"""
Standalone camera test — raw capture with diagnostics.
Bypasses CameraReader.read() validation to see what the camera actually returns.
"""

import sys
import time
import cv2
from camera import CameraReader


def main():
    print("=== Camera Raw Diagnostic ===")
    cam = CameraReader()

    if not cam.is_opened():
        print("FATAL: Camera not opened")
        sys.exit(1)

    print("\nCapturing 200 raw frames from cam.cap.read()...\n")

    good = 0
    bad = 0
    printed = 0

    for i in range(200):
        try:
            ret, frame = cam.cap.read()
        except Exception as e:
            print(f"[{i}] EXCEPTION: {e}")
            bad += 1
            time.sleep(0.1)
            continue

        # Diagnostic: log every bad frame detail (up to 20)
        is_bad = False
        reason = ""

        if not ret:
            is_bad = True
            reason = "ret=False"
        elif frame is None:
            is_bad = True
            reason = "frame is None"
        elif frame.size == 0:
            is_bad = True
            reason = f"frame.size=0, shape={frame.shape}"
        elif frame.shape[0] < 10 or frame.shape[1] < 10:
            is_bad = True
            reason = f"frame too small: shape={frame.shape}"

        if is_bad:
            bad += 1
            if printed < 20:
                print(f"[{i}] BAD ({reason})")
                printed += 1
            time.sleep(0.05)
            continue

        # Good frame
        good += 1
        if good == 1:
            print(f"[{i}] FIRST GOOD: shape={frame.shape}, dtype={frame.dtype}, "
                  f"min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")
        if good <= 5:
            sys.stdout.write(f"\r  good={good}, bad={bad}...")
            sys.stdout.flush()

    print(f"\n\nResult: {good} good, {bad} bad out of 200 raw reads")

    if good > 0:
        print("\nCamera works. Issue is in CameraReader.read() validation.")
        print("Next: fix the validation logic.")

    cam.release()
    print("Done.")


if __name__ == '__main__':
    main()
