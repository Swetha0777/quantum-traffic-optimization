# Vision Module — Real-Time Vehicle Detection & Line Counter

This application is built for **Person 1 — Traffic Simulation / Traffic Data / Vehicle Detection** in the **Quantum-Enhanced Adaptive Urban Traffic Optimization** project.

It provides a complete, standalone real-time vehicle detection, tracking, CCTV line-crossing counting, and traffic analytics web application using OpenCV, Ultralytics YOLOv8, ByteTrack, FastAPI, and WebSockets.

---

## 📁 File Structure

```
vision/
├── __init__.py
├── config.py             # Model, line position, class mappings & threshold settings
├── detector.py           # YOLO detector wrapper (loads model once, handles inference)
├── tracker.py            # Ultralytics ByteTrack integration & track history state
├── counter.py            # Line-crossing CCTV counting math (1-time count guarantee)
├── traffic_analyzer.py   # Traffic density, state (LOW/MODERATE/HIGH/VERY HIGH), queue & waiting time
├── video_processor.py   # Frame pipeline, visual box/line overlay renderer, telemetry encoder
├── backend/
│   ├── __init__.py
│   └── main.py           # FastAPI web server & WebSocket streaming backend
├── frontend/
│   ├── index.html        # Modern dark-mode dashboard UI
│   ├── style.css         # Glassmorphism design system & micro-animations
│   └── app.js            # WebSocket client, DOM updates & sliders
└── README.md             # Documentation & execution instructions
```

---

## 🚀 How to Run

### Command:
```bash
.venv\Scripts\python.exe -m vision.backend.main
```
or
```bash
python -m vision.backend.main
```

### Browser URL:
Open your browser and navigate to:
👉 **`http://127.0.0.1:8000`**

---

## ⚙️ Features & Workflow

1. **Dual Input Mode**:
   - **Upload Traffic Video**: Click `Upload MP4 Video` to upload any traffic MP4 video. The server processes it frame-by-frame (without loading the entire video into RAM).
   - **Laptop Webcam**: Click `Start Laptop Camera` to connect your laptop camera live stream.
2. **Detection Categories**:
   - Detects **Car**, **Motorcycle**, **Bus**, and **Truck** separately using YOLOv8.
3. **Persistent ByteTrack Tracking**:
   - Keeps persistent vehicle IDs across frames.
4. **CCTV Line-Crossing Counter**:
   - Draws a visual counting line across the video frame.
   - Adjust the counting line height live using the slider (10% to 90%).
   - Stores counted IDs in a `counted_ids` set to ensure every vehicle is counted **EXACTLY ONCE**.
5. **Counters**:
   - Live category counts: **CAR**, **MOTORCYCLE**, **BUS**, **TRUCK**.
   - Direction counts: **IN** (TOP → BOTTOM), **OUT** (BOTTOM → TOP).
   - **TOTAL COUNT** is strictly enforced as: $\text{Total} = \text{Car} + \text{Motorcycle} + \text{Bus} + \text{Truck}$.
6. **Traffic Analytics Telemetry**:
   - Real-time estimation of **Traffic Density**, **Queue Length**, **Average Waiting Time**, and **Traffic State** (`LOW`, `MODERATE`, `HIGH`, `VERY HIGH`).
