from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import json
import datetime
import asyncio
import cv2
import mediapipe as mp
import threading

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Shared data store ──────────────────────────────────────────────
alerts = []
metrics = {
    "persons": 0,
    "sos_events": 0,
    "helmet_violations": 0,
    "zone_violations": 0,
    "status": "NORMAL"
}
connected_clients = []
latest_frame = [None]
running = [True]

# ── WebSocket manager ──────────────────────────────────────────────
async def broadcast(data: dict):
    msg = json.dumps(data)
    for client in list(connected_clients):
        try:
            await client.send_text(msg)
        except:
            connected_clients.remove(client)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    print(f"[Dashboard] Client connected. Total: {len(connected_clients)}")
    try:
        # Send current state on connect
        await ws.send_text(json.dumps({
            "type": "init",
            "alerts": alerts[-20:],
            "metrics": metrics
        }))
        while True:
            await asyncio.sleep(1)
            # Send live metrics every second
            await ws.send_text(json.dumps({
                "type": "metrics",
                "metrics": metrics,
                "time": datetime.datetime.now().strftime("%H:%M:%S")
            }))
    except WebSocketDisconnect:
        connected_clients.remove(ws)
        print(f"[Dashboard] Client disconnected.")

@app.post("/alert")
async def post_alert(data: dict):
    alert = {
        "id": len(alerts) + 1,
        "level": data.get("level", "INFO"),
        "module": data.get("module", "SYSTEM"),
        "message": data.get("message", ""),
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "location": data.get("location", "Campus")
    }
    alerts.append(alert)
    metrics[data.get("metric_key", "persons")] = data.get("metric_val", 0)

    await broadcast({"type": "alert", "alert": alert, "metrics": metrics})
    return {"status": "sent"}

@app.get("/")
async def serve_dashboard():
    return FileResponse("frontend/index.html")

# ── AI Detection Thread ────────────────────────────────────────────
def run_detection():
    import requests

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.75)
    mp_draw = mp.solutions.drawing_utils

    try:
        from ultralytics import YOLO
        yolo = YOLO("yolov8n.pt")
        yolo_ready = True
    except:
        yolo_ready = False
        print("[Detection] YOLO not available, running gesture only")

    cap = cv2.VideoCapture(0)
    frame_count = 0
    sos_frame_count = 0

    print("[Detection] Camera started")

    while running[0]:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        h, w = frame.shape[:2]

        # ── SOS Gesture Detection ──────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        sos_now = False

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                tips = [8, 12, 16, 20]
                knuckles = [6, 10, 14, 18]
                lm = hand.landmark
                folded = all(lm[t].y > lm[k].y for t, k in zip(tips, knuckles))
                thumb = abs(lm[4].x - lm[2].x) < 0.12
                if folded and thumb:
                    sos_now = True

        sos_frame_count = sos_frame_count + 1 if sos_now else max(0, sos_frame_count - 1)

        if sos_frame_count >= 3:
            sos_frame_count = 0
            metrics["sos_events"] += 1
            try:
                requests.post("http://localhost:8000/alert", json={
                    "level": "CRITICAL",
                    "module": "SOS",
                    "message": "SOS Gesture Detected — Emergency!",
                    "location": "Webcam",
                    "metric_key": "sos_events",
                    "metric_val": metrics["sos_events"]
                }, timeout=1)
            except:
                pass
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 200), -1)
            cv2.putText(frame, "!! SOS ALERT !!", (10, 42),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

        # ── Person / Crowd Detection ───────────────────────────
        if yolo_ready and frame_count % 10 == 0:
            yolo_results = yolo(frame, classes=[0], conf=0.45, verbose=False)
            count = len(yolo_results[0].boxes)
            metrics["persons"] = count

            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)

            if count >= 5:
                metrics["status"] = "CROWD_DETECTED"
                try:
                    requests.post("http://localhost:8000/alert", json={
                        "level": "WARNING",
                        "module": "CROWD",
                        "message": f"Crowd detected — {count} persons",
                        "location": "Webcam",
                        "metric_key": "persons",
                        "metric_val": count
                    }, timeout=1)
                except:
                    pass
            else:
                metrics["status"] = "NORMAL"

        # ── HUD Overlay ────────────────────────────────────────
        cv2.putText(frame, f"EYEQ LIVE | Persons: {metrics['persons']}",
                    (10, h - 15), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 229, 160), 2)
        cv2.putText(frame,
                    datetime.datetime.now().strftime("%H:%M:%S"),
                    (w - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        latest_frame[0] = frame.copy()
        cv2.imshow("EYEQ — Live Detection", frame)

        if cv2.waitKey(1) == 27:
            running[0] = False
            break

    cap.release()
    cv2.destroyAllWindows()

# Start detection in background thread
detection_thread = threading.Thread(target=run_detection, daemon=True)
detection_thread.start()

if __name__ == "__main__":
    print("\n========================================")
    print("  EYEQ SERVER STARTING")
    print("  Dashboard: http://localhost:8000")
    print("  Press ESC in camera window to stop")
    print("========================================\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)