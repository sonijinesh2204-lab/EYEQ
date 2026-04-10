"""
EYEQ — AI Campus Watchdog System
Main runner: starts all detection modules in parallel threads
Usage: python main.py
"""

import cv2
import threading
import time
import datetime
import json
from ai_models.helmet_detection import HelmetDetector
from ai_models.sos_gesture import SOSDetector
from ai_models.crowd_detection import CrowdPredictor
from ai_models.number_plate import NumberPlateReader
from ai_models.restricted_zone import RestrictedZoneMonitor
from utils.alert_system import AlertSystem
from utils.logger import EventLogger

# ─── Configuration ────────────────────────────────────────────────
CAMERA_SOURCE = 0           # 0 = laptop webcam | "rtsp://ip" = CCTV
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
FPS_TARGET    = 30
SHOW_WINDOW   = True        # Set False for headless server deploy

# ─── Shared state (thread-safe) ──────────────────────────────────
import threading
frame_lock  = threading.Lock()
latest_frame = [None]
running      = [True]

# ─── Module Instances ─────────────────────────────────────────────
helmet_detector = HelmetDetector()
sos_detector    = SOSDetector()
crowd_predictor = CrowdPredictor()
plate_reader    = NumberPlateReader()
zone_monitor    = RestrictedZoneMonitor()
alert_system    = AlertSystem()
logger          = EventLogger()


def capture_thread():
    """Continuously captures frames from camera."""
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    print(f"[EYEQ] Camera opened: {CAMERA_SOURCE}")
    while running[0]:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame[0] = frame.copy()
        time.sleep(1 / FPS_TARGET)
    cap.release()


def detection_thread():
    frame_count = 0
    while running[0]:
        with frame_lock:
            frame = latest_frame[0]
        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        annotated = frame.copy()

        crowd_result = crowd_predictor.detect(frame)
        if crowd_result["status"] in ("CROWD_DETECTED", "CROWD_FORMING_SOON"):
            alert_system.send(
                level   = "WARNING",
                module  = "CROWD",
                message = f"{crowd_result['status']} — {crowd_result['count']} persons",
                snapshot= frame
            )
            logger.log(crowd_result)
        # ── 1. Crowd Detection (every frame) ─────────────────────
        crowd_result = crowd_predictor.detect(frame)
        if crowd_result["status"] in ("CROWD_DETECTED", "CROWD_FORMING_SOON"):
            alert_system.send(
                level   = "WARNING",
                module  = "CROWD",
                message = f"{crowd_result['status']} — {crowd_result['count']} persons",
                snapshot= frame
            )
            logger.log(crowd_result)

        # ── 2. Helmet Detection (every 5 frames for speed) ──────
        if frame_count % 5 == 0:
            helmet_result = helmet_detector.detect(frame)
            if helmet_result["count"] > 0:
                # Trigger plate reading for vehicle capture
                plate_result = plate_reader.detect(frame)
                alert_system.send(
                    level   = "WARNING",
                    module  = "HELMET",
                    message = f"No helmet — plates: {[p['plate'] for p in plate_result]}",
                    snapshot= frame
                )
                logger.log({**helmet_result, "plates": plate_result})

        # ── 3. SOS Gesture (every 3 frames) ─────────────────────
        if frame_count % 3 == 0:
            sos_result = sos_detector.detect(frame)
            if sos_result["sos"]:
                alert_system.send(
                    level   = "CRITICAL",
                    module  = "SOS",
                    message = "SOS GESTURE DETECTED — EMERGENCY",
                    snapshot= frame
                )
                logger.log(sos_result)

        # ── 4. Restricted Zone (every 10 frames) ─────────────────
        if frame_count % 10 == 0:
            zone_result = zone_monitor.detect(frame)
            if zone_result["violation"]:
                alert_system.send(
                    level   = "ALERT",
                    module  = "ZONE",
                    message = f"Restricted area violation: {zone_result['zone']}",
                    snapshot= frame
                )
                logger.log(zone_result)

        # ── Annotate and display ──────────────────────────────────
        if SHOW_WINDOW:
            _annotate_frame(annotated, crowd_result)
            cv2.imshow("EYEQ — AI Campus Watchdog", annotated)
            if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
                running[0] = False
                break


def _annotate_frame(frame, crowd):
    """Draws HUD overlay on frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Status bar background
    cv2.rectangle(overlay, (0, 0), (w, 50), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # EYEQ watermark
    cv2.putText(frame, "EYEQ WATCHDOG", (10, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 229, 160), 2)

    # Person count
    count_text = f"PERSONS: {crowd.get('count', 0)}"
    color = (0, 69, 255) if crowd.get("status") == "CROWD_DETECTED" else (0, 229, 160)
    cv2.putText(frame, count_text, (w - 240, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)

    # Timestamp
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, ts, (w // 2 - 100, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (180, 180, 180), 1)


def main():
    print("""
╔══════════════════════════════════════════╗
║     EYEQ — AI CAMPUS WATCHDOG SYSTEM    ║
║     Starting all detection modules...   ║
╚══════════════════════════════════════════╝
    """)

    threads = [
        threading.Thread(target=capture_thread,   daemon=True, name="CaptureThread"),
        threading.Thread(target=detection_thread, daemon=True, name="DetectionThread"),
    ]

    for t in threads:
        t.start()
        print(f"[EYEQ] Started {t.name}")

    # Keep main thread alive
    try:
        while running[0]:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[EYEQ] Shutting down...")
        running[0] = False

    cv2.destroyAllWindows()
    print("[EYEQ] System stopped.")


if __name__ == "__main__":
    main()
