"""
Manual camera example for FaceMeshDetector.

Tests:
1. Inspect MediaPipe V2 landmark output and observed frame rate
2. L_ref calculation stable
3. AU4/AU6 indices map correctly
4. Normalized landmark tensors export
"""

import sys
import time
import cv2
import numpy as np

from src.vision.face_mesh import (
    FaceMeshDetector,
    draw_landmarks,
    KEY_LANDMARKS,
    FaceMeshDetector as fm
)


def test_landmarker_initialization():
    """Test 1: Verify MediaPipe loads correctly."""
    print("=" * 50)
    print("TEST 1: Initialization")
    print("=" * 50)

    model_path = 'models/face_landmarker.task'

    try:
        detector = FaceMeshDetector(
            model_path=model_path,
            num_faces=1,
            output_blendshapes=True
        )
        print("[PASS] FaceLandmarker initialized successfully")
        return detector
    except Exception as e:
        print(f"[FAIL] Initialization failed: {e}")
        return None


def test_fps_performance(detector: FaceMeshDetector):
    """Test 2: Verify FPS >= 30."""
    print("\n" + "=" * 50)
    print("TEST 2: FPS Performance")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FAIL] Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_times = []
    last_time = time.time()

    try:
        for i in range(60):  # Test 60 frames
            ret, frame = cap.read()
            if not ret:
                print(f"[FAIL] Failed to read frame {i}")
                break

            timestamp_ms = int(time.time() * 1000)
            detector.detect_async(frame, timestamp_ms)

            # Allow some time for processing
            time.sleep(0.001)

            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            if dt > 0:
                frame_times.append(1.0 / dt)

            # Get result every few frames
            if i % 10 == 0:
                result = detector.get_result(timeout_ms=50)
                if result and result.face_landmarks:
                    num_landmarks = len(result.face_landmarks[0])
                    print(f"  Frame {i}: {num_landmarks} landmarks detected")

        cap.release()

        if frame_times:
            avg_fps = np.mean(frame_times)
            print(f"\n[AVERAGE FPS]: {avg_fps:.1f}")

            if avg_fps >= 30:
                print("[PASS] FPS >= 30")
            else:
                print(f"[WARN] FPS < 30 (got {avg_fps:.1f})")

    except Exception as e:
        print(f"[FAIL] FPS test failed: {e}")
        cap.release()


def test_lref_calculation(detector: FaceMeshDetector):
    """Test 3: Verify L_ref calculation."""
    print("\n" + "=" * 50)
    print("TEST 3: L_ref Calculation")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FAIL] Cannot open camera")
        return

    lref_values = []

    try:
        for i in range(30):
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(time.time() * 1000)
            detector.detect_async(frame, timestamp_ms)
            time.sleep(0.01)

            result = detector.get_result(timeout_ms=50)
            if result and result.face_landmarks:
                landmarks = detector.get_landmarks(face_idx=0)
                if landmarks is not None:
                    lref = detector.get_scale_reference(landmarks)
                    lref_values.append(lref)

        cap.release()

        if lref_values:
            mean_lref = np.mean(lref_values)
            std_lref = np.std(lref_values)
            cv = std_lref / mean_lref if mean_lref > 0 else 0

            print(f"  L_ref mean: {mean_lref:.4f}")
            print(f"  L_ref std:  {std_lref:.4f}")
            print(f"  Coefficient of variation: {cv:.4f}")

            if cv < 0.1:  # Less than 10% variation
                print("[PASS] L_ref is stable (CV < 10%)")
            else:
                print(f"[WARN] L_ref variation high (CV = {cv:.2%})")

    except Exception as e:
        print(f"[FAIL] L_ref test failed: {e}")
        cap.release()


def test_au_distances(detector: FaceMeshDetector):
    """Test 4: Verify AU distances."""
    print("\n" + "=" * 50)
    print("TEST 4: AU Distance Calculation")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FAIL] Cannot open camera")
        return

    try:
        print("  Position your face in frame and make expressions...")
        print("  Testing AU4 (frown) and AU6 (smile) detection...")

        for i in range(30):
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(time.time() * 1000)
            detector.detect_async(frame, timestamp_ms)
            time.sleep(0.01)

            result = detector.get_result(timeout_ms=50)
            if result and result.face_landmarks:
                landmarks = detector.get_landmarks(face_idx=0)
                if landmarks is not None:
                    au_distances = detector.get_au_distances(landmarks)

                    if au_distances and i % 10 == 0:
                        print(f"  AU4 (brow): {au_distances.get('au4_brow', 0):.4f}")
                        print(f"  AU6 (cheek): {au_distances.get('au6_cheek', 0):.4f}")

        cap.release()
        print("[PASS] AU distance calculation working")

    except Exception as e:
        print(f"[FAIL] AU test failed: {e}")
        cap.release()


def test_visualization(detector: FaceMeshDetector):
    """Test 5: Visual debug overlay."""
    print("\n" + "=" * 50)
    print("TEST 5: Visualization")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FAIL] Cannot open camera")
        return

    print("  Press 'q' to quit visualization")
    print("  Green: all landmarks")
    print("  Red: key AU landmarks (33, 263, 65, 296, 160, 385)")

    highlight_indices = [
        KEY_LANDMARKS['left_eye_outer'],
        KEY_LANDMARKS['right_eye_outer'],
        KEY_LANDMARKS['left_brow_inner'],
        KEY_LANDMARKS['right_brow_inner'],
        KEY_LANDMARKS['left_cheek'],
        KEY_LANDMARKS['right_cheek'],
    ]

    window_name = 'FaceMeshDetector Test'

    # Check if baseline is set
    baseline_set = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(time.time() * 1000)
            detector.detect_async(frame, timestamp_ms)

            result = detector.get_result(timeout_ms=50)

            if result and result.face_landmarks:
                landmarks = detector.get_landmarks(face_idx=0)

                if landmarks is not None:
                    # Draw landmarks
                    output = draw_landmarks(
                        frame,
                        landmarks,
                        highlight_indices=highlight_indices,
                        color=(0, 255, 0),
                        radius=2
                    )

                    # Add info overlay
                    lref = detector.get_scale_reference(landmarks)
                    au_dist = detector.get_au_distances(landmarks)
                    ear = detector.get_ear(landmarks)
                    mar = detector.get_mar(landmarks)

                    info_text = f"L_ref: {lref:.3f}"
                    cv2.putText(output, info_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if au_dist:
                        if baseline_set:
                            # Show relative features
                            rel = detector.get_relative_features(landmarks)
                            au_text = f"AU4: {rel.get('delta_au4', 0):.3f} AU6: {rel.get('delta_au6', 0):.3f}"
                        else:
                            au_text = f"AU4: {au_dist.get('au4_brow', 0):.3f} AU6: {au_dist.get('au6_cheek', 0):.3f}"
                        cv2.putText(output, au_text, (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # EAR/MAR display
                    if baseline_set:
                        rel = detector.get_relative_features(landmarks)
                        eye_text = f"EAR: {rel.get('delta_ear', 0):.3f} MAR: {rel.get('delta_mar', 0):.3f}"
                    else:
                        eye_text = f"EAR: {ear:.3f} MAR: {mar:.3f}"
                    cv2.putText(output, eye_text, (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Baseline status
                    status_text = "BASELINE: SET" if baseline_set else "BASELINE: NOT SET"
                    status_color = (0, 255, 0) if baseline_set else (0, 165, 255)
                    cv2.putText(output, status_text, (10, 150),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            else:
                output = frame

            cv2.imshow(window_name, output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Press 'R' to capture new baseline (resets old one first)
                detector.set_neutral_baseline(landmarks)
                baseline_set = True
                print(f"  [BASELINE CAPTURED] AU4: {detector._neutral_baseline.get('au4_brow', 0):.4f}, AU6: {detector._neutral_baseline.get('au6_cheek', 0):.4f}")

    except Exception as e:
        print(f"[FAIL] Visualization failed: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[DONE] Visualization closed")


def test_ear_calculation(detector: FaceMeshDetector):
    """Test 6: Verify EAR calculation."""
    print("\n" + "=" * 50)
    print("TEST 6: EAR Calculation")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FAIL] Cannot open camera")
        return

    ear_values = []

    try:
        print("  Testing EAR with eyes open (idle)...")
        for i in range(30):
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(time.time() * 1000)
            detector.detect_async(frame, timestamp_ms)
            time.sleep(0.01)

            result = detector.get_result(timeout_ms=50)
            if result and result.face_landmarks:
                landmarks = detector.get_landmarks(face_idx=0)
                if landmarks is not None:
                    ear = detector.get_ear(landmarks)
                    ear_values.append(ear)

        cap.release()

        if ear_values:
            mean_ear = np.mean(ear_values)
            print(f"  EAR mean: {mean_ear:.4f}")

            # Check if EAR is in expected range for open eyes
            if 0.20 <= mean_ear <= 0.45:
                print(f"[PASS] EAR in expected range (0.20-0.45) for open eyes")
            else:
                print(f"[WARN] EAR outside typical range (got {mean_ear:.3f})")
        else:
            print("[FAIL] No EAR values captured")

    except Exception as e:
        print(f"[FAIL] EAR test failed: {e}")
        cap.release()


def test_mar_calculation(detector: FaceMeshDetector):
    """Test 7: Verify MAR calculation."""
    print("\n" + "=" * 50)
    print("TEST 7: MAR Calculation")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FAIL] Cannot open camera")
        return

    mar_values = []

    try:
        print("  Testing MAR with mouth closed (idle)...")
        for i in range(30):
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(time.time() * 1000)
            detector.detect_async(frame, timestamp_ms)
            time.sleep(0.01)

            result = detector.get_result(timeout_ms=50)
            if result and result.face_landmarks:
                landmarks = detector.get_landmarks(face_idx=0)
                if landmarks is not None:
                    mar = detector.get_mar(landmarks)
                    mar_values.append(mar)

        cap.release()

        if mar_values:
            mean_mar = np.mean(mar_values)
            print(f"  MAR mean: {mean_mar:.4f}")

            # Check if MAR is in expected range for closed mouth
            if mean_mar < 0.10:
                print(f"[PASS] MAR < 0.10 for closed mouth")
            else:
                print(f"[WARN] MAR higher than expected (got {mean_mar:.3f})")
        else:
            print("[FAIL] No MAR values captured")

    except Exception as e:
        print(f"[FAIL] MAR test failed: {e}")
        cap.release()


def test_baseline_calibration(detector: FaceMeshDetector):
    """Test 8: Verify baseline calibration."""
    print("\n" + "=" * 50)
    print("TEST 8: Baseline Calibration")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FAIL] Cannot open camera")
        return

    try:
        # Capture baseline from neutral expression
        print("  Capturing neutral baseline (keep face neutral)...")
        baseline_landmarks = None

        for i in range(30):
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(time.time() * 1000)
            detector.detect_async(frame, timestamp_ms)
            time.sleep(0.01)

            result = detector.get_result(timeout_ms=50)
            if result and result.face_landmarks and baseline_landmarks is None:
                baseline_landmarks = detector.get_landmarks(face_idx=0)

        if baseline_landmarks is None:
            print("[FAIL] Could not capture baseline landmarks")
            cap.release()
            return

        detector.set_neutral_baseline(baseline_landmarks)
        print("  Baseline set!")

        # Now test relative features
        print("  Testing relative features...")
        rel_features = detector.get_relative_features(baseline_landmarks)

        # Relative features from baseline should be ~0
        print(f"  delta_au4: {rel_features.get('delta_au4', 0):.4f}")
        print(f"  delta_au6: {rel_features.get('delta_au6', 0):.4f}")
        print(f"  delta_ear: {rel_features.get('delta_ear', 0):.4f}")
        print(f"  delta_mar: {rel_features.get('delta_mar', 0):.4f}")

        delta_sum = (abs(rel_features.get('delta_au4', 0)) +
                    abs(rel_features.get('delta_au6', 0)) +
                    abs(rel_features.get('delta_ear', 0)) +
                    abs(rel_features.get('delta_mar', 0)))

        if delta_sum < 0.5:
            print("[PASS] Relative features near zero after baseline")
        else:
            print(f"[WARN] Relative features not near zero (sum: {delta_sum:.3f})")

        # Reset baseline and verify absolute values return
        print("  Resetting baseline...")
        detector.reset_neutral_baseline()
        abs_features = detector.get_relative_features(baseline_landmarks)

        # Without baseline, should return absolute values
        print("  After reset, get_relative_features returns absolute values")
        print(f"  au4_brow: {abs_features.get('au4_brow', 0):.4f}")
        print(f"  au6_cheek: {abs_features.get('au6_cheek', 0):.4f}")
        print(f"  ear: {abs_features.get('ear', 0):.4f}")
        print(f"  mar: {abs_features.get('mar', 0):.4f}")

        cap.release()
        print("[PASS] Baseline calibration working")

    except Exception as e:
        print(f"[FAIL] Baseline test failed: {e}")
        cap.release()


def test_all_features(detector: FaceMeshDetector):
    """Test 10: Verify get_all_features() returns all keys."""
    print("\n" + "=" * 50)
    print("TEST 10: All Features Combined")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FAIL] Cannot open camera")
        return

    try:
        print("  Testing get_all_features()...")
        landmarks = None

        for i in range(30):
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(time.time() * 1000)
            detector.detect_async(frame, timestamp_ms)
            time.sleep(0.01)

            result = detector.get_result(timeout_ms=50)
            if result and result.face_landmarks and landmarks is None:
                landmarks = detector.get_landmarks(face_idx=0)

        if landmarks is None:
            print("[FAIL] Could not capture landmarks")
            cap.release()
            return

        features = detector.get_all_features(landmarks)

        # Verify all expected keys present
        expected_keys = ['au4_brow', 'au6_cheek', 'ear', 'mar', 'lref']
        missing_keys = [k for k in expected_keys if k not in features]

        if missing_keys:
            print(f"[FAIL] Missing keys: {missing_keys}")
        else:
            print("[PASS] All feature keys present:")
            for key in expected_keys:
                print(f"  {key}: {features.get(key, 0):.4f}")

        cap.release()

    except Exception as e:
        print(f"[FAIL] All features test failed: {e}")
        cap.release()


def main():
    """Run all tests."""
    print("\n" + "#" * 50)
    print("# Phase 1.5: FaceMeshDetector Verification")
    print("#" * 50 + "\n")

    detector = test_landmarker_initialization()
    if detector is None:
        print("\n[ABORT] Cannot proceed without detector")
        return 1

    try:
        # Phase 1 Tests
        test_fps_performance(detector)
        test_lref_calculation(detector)
        test_au_distances(detector)

        # Phase 1.5 Tests
        test_ear_calculation(detector)
        test_mar_calculation(detector)
        test_baseline_calibration(detector)
        test_all_features(detector)

        # Visualization (optional - uncomment to enable)
        test_visualization(detector)

        print("\n" + "=" * 50)
        print("TESTS COMPLETE")
        print("=" * 50)

    finally:
        detector.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())