"""
Configuration settings for the Vision module.
Updated for dense traffic detection, configurable inference resolution (imgsz),
max detections, and persistent unique track ID confirmation counting.
"""

from typing import Dict

# YOLO Model & Detection Settings
DEFAULT_MODEL_NAME: str = "yolov8n.pt"
DEFAULT_CONF_THRESHOLD: float = 0.20   # Lower default threshold tuned for dense scenes
DEFAULT_IOU_THRESHOLD: float = 0.45    # NMS IoU threshold for overlapping vehicles
DEFAULT_IMGSZ: int = 960               # High resolution inference (640, 960, 1280) for small vehicles
DEFAULT_MAX_DET: int = 300             # Allow up to 300 detections per frame for crowded traffic

# Track Confirmation Settings
DEFAULT_CONFIRMATION_FRAMES: int = 3   # Vehicle must be detected for >= N frames before confirmed as unique

# COCO Dataset Vehicle Class IDs mapping
TARGET_VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Traffic Analysis Thresholds (based on active vehicles in frame)
TRAFFIC_STATE_THRESHOLDS = {
    "LOW": 6,          # <= 6 active vehicles
    "MODERATE": 15,    # 7 - 15 active vehicles
    "HIGH": 30,        # 16 - 30 active vehicles
    "VERY HIGH": 1000  # > 30 active vehicles
}

# Maximum capacity estimate for density calculation
MAX_FRAME_CAPACITY: int = 35

# Queue & Waiting Time Thresholds
STATIONARY_SPEED_THRESHOLD_PX: float = 2.5  # pixels displacement per frame
STATIONARY_TIME_THRESHOLD_SEC: float = 1.5  # seconds stationary before counted in queue

# Class Color Palette for Visual Overlay (BGR format for OpenCV)
CLASS_COLORS = {
    "car": (255, 178, 50),        # Blue
    "motorcycle": (50, 255, 178), # Teal / Cyan
    "bus": (178, 50, 255),        # Purple
    "truck": (50, 150, 255)       # Coral Orange
}

CONFIRMED_COLOR = (50, 255, 50)   # Neon Green glow
