# EYEQ — AI Campus Watchdog System
## Complete Setup & Run Guide (36-Hour Hackathon)

---

## STEP 1 — Install Dependencies (15 minutes)

```bash
# Create virtual environment
python -m venv eyeq_env

# Activate (Windows)
eyeq_env\Scripts\activate

# Activate (Mac/Linux)
source eyeq_env/bin/activate

# Install all packages
pip install -r requirements.txt
```

---

## STEP 2 — Project Structure

```
EYEQ/
├── main.py                     ← Run this to start all modules
├── requirements.txt
├── .env                        ← Add Firebase/Twilio keys here
│
├── ai_models/
│   ├── helmet_detection.py     ← YOLOv8 helmet detector
│   ├── sos_gesture.py          ← MediaPipe SOS gesture
│   ├── crowd_detection.py      ← YOLOv8 + trend predictor
│   ├── number_plate.py         ← YOLO + EasyOCR
│   └── restricted_zone.py      ← Virtual zone monitor
│
├── backend/
│   └── app.py                  ← FastAPI server
│
├── utils/
│   ├── alert_system.py         ← Multi-channel alerts
│   └── logger.py               ← Event logger
│
├── models/                     ← Put trained .pt files here
│   ├── helmet_yolov8.pt
│   └── plate_yolov8.pt
│
├── frontend/
│   └── index.html              ← Hackathon website
│
├── snapshots/                  ← Auto-created, stores alert images
└── logs/                       ← Auto-created, stores event logs
```

---

## STEP 3 — Environment Variables (optional for full alerts)

Create `.env` file:
```
FIREBASE_SERVER_KEY=your_firebase_server_key
FCM_DEVICE_TOKEN=your_device_token
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_FROM_NUMBER=+1234567890
ALERT_PHONE_NUMBER=+91XXXXXXXXXX
DATABASE_URL=postgresql://user:password@localhost/eyeq
```

For hackathon demo, leave `.env` empty — console alerts work without any keys.

---

## STEP 4 — Run the System

### Option A: Run all modules together (RECOMMENDED)
```bash
python main.py
```

### Option B: Run individual modules for testing
```bash
# Test helmet detection only
python ai_models/helmet_detection.py

# Test SOS gesture only
python ai_models/sos_gesture.py

# Test crowd detection only
python ai_models/crowd_detection.py

# Test number plate only
python ai_models/number_plate.py

# Test zone monitor only
python ai_models/restricted_zone.py
```

### Option C: Run backend API server
```bash
cd backend
uvicorn app:app --reload --port 8000

# API docs available at:
# http://localhost:8000/docs
```

---

## STEP 5 — Open Dashboard Website

Simply open `frontend/index.html` in any browser.

For live data connection, update the WebSocket URL in the React dashboard to:
```
ws://localhost:8000/ws
```

---

## STEP 6 — Camera Configuration

**Demo (Laptop Webcam):**
```python
# In main.py, line 14:
CAMERA_SOURCE = 0    # Default webcam
```

**Production (CCTV via RTSP):**
```python
CAMERA_SOURCE = "rtsp://username:password@camera_ip:554/stream"
```
This is the ONLY line that changes. All AI models remain identical.

---

## STEP 7 — Training Custom Models (for better accuracy)

### Helmet Detection Dataset:
1. Collect 500+ images of riders with/without helmets
2. Label using Roboflow (https://roboflow.com) — free tier
3. Export in YOLOv8 format
4. Train:
```bash
yolo detect train data=helmet_dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
```

### Number Plate Dataset:
- Use: https://universe.roboflow.com/search?q=indian+number+plate
- Free Indian plate datasets available
- Same training command, different data yaml

---

## TEAM ROLES (Divide & Conquer for 36 hours)

| Member | Module | Hours |
|--------|--------|-------|
| Member 1 | YOLOv8 helmet training + testing | 12h |
| Member 2 | SOS gesture + crowd prediction logic | 12h |
| Member 3 | FastAPI backend + database + alerts | 12h |
| Member 4 | React dashboard + website + presentation | 12h |

---

## VIVA QUICK REFERENCE

**"Why AI?"**
> "Manual CCTV monitoring is inefficient and reactive. EYEQ automates detection in real-time with sub-500ms latency — impossible manually."

**"How is accuracy ensured?"**
> "Transfer learning on YOLOv8, custom campus dataset, augmentation for real-world conditions, confidence thresholding at 60-75%."

**"Biggest innovation?"**
> "Crowd PREDICTION before formation, and gesture-based SOS — two features with direct patent potential."

**"Why webcam, not CCTV?"**
> "Hardware-agnostic design. Switching to CCTV requires changing one line: `VideoCapture(0)` → `VideoCapture('rtsp://ip')`."

**"Final line for judges:"**
> "EYEQ converts traditional passive surveillance into an intelligent, predictive, and life-saving AI security system — hardware-agnostic, campus-ready, and patent-worthy."

---

## QUICK INSTALL CHECK

Run this to verify everything is installed:
```bash
python -c "
import cv2, mediapipe, easyocr, fastapi, ultralytics
print('All dependencies OK!')
print(f'OpenCV: {cv2.__version__}')
print(f'MediaPipe: {mediapipe.__version__}')
print(f'Ultralytics: {ultralytics.__version__}')
"
```

---

## TROUBLESHOOTING

| Problem | Fix |
|---------|-----|
| Camera not opening | Try `CAMERA_SOURCE = 1` instead of 0 |
| YOLOv8 slow on CPU | Use `yolov8n.pt` (nano model, fastest) |
| EasyOCR takes long to start | First load downloads models — normal |
| MediaPipe import error | `pip install mediapipe==0.10.7` |
| Port 8000 in use | `uvicorn app:app --port 8001` |
