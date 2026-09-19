"""
YOLO Vehicle Detector wrapper using Ultralytics API.
Supports configurable inference image size (imgsz), max detections, confidence, and ByteTrack.
"""

import logging
from typing import Dict, List, Optional
import numpy as np
from ultralytics import YOLO

from vision.config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_IMGSZ,
    DEFAULT_MAX_DET,
    TARGET_VEHICLE_CLASSES
)

logger = logging.getLogger(__name__)


class VehicleDetector:
    """Wrapper around Ultralytics YOLO model tuned for dense traffic scenes."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        imgsz: int = DEFAULT_IMGSZ,
        max_det: int = DEFAULT_MAX_DET
    ):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.max_det = max_det
        self.target_classes = list(TARGET_VEHICLE_CLASSES.keys())
        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the YOLO model weights. Executed only once."""
        try:
            logger.info(f"Loading YOLO model: {self.model_name}")
            self.model = YOLO(self.model_name)
            logger.info(f"YOLO model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model '{self.model_name}': {e}")
            raise RuntimeError(f"YOLO model loading failed: {e}")

    def update_settings(
        self,
        conf_threshold: Optional[float] = None,
        imgsz: Optional[int] = None,
        max_det: Optional[int] = None
    ) -> None:
        """Update detection & inference parameters live."""
        if conf_threshold is not None:
            self.conf_threshold = max(0.05, min(0.95, conf_threshold))
        if imgsz is not None:
            self.imgsz = imgsz
        if max_det is not None:
            self.max_det = max_det

    def detect_and_track(
        self,
        frame: np.ndarray,
        persist: bool = True,
        tracker_type: str = "bytetrack.yaml"
    ) -> List[Dict]:
        """
        Perform high-resolution detection and ByteTrack tracking on a single frame.

        Args:
            frame: BGR numpy image frame.
            persist: Keep track history across consecutive frames.
            tracker_type: Tracker configuration file ('bytetrack.yaml').

        Returns:
            List of dicts containing vehicle detections:
            [{
                'id': int,
                'class_id': int,
                'class_name': str,
                'confidence': float,
                'bbox': [x1, y1, x2, y2],
                'center': (cx, cy)
            }]
        """
        if self.model is None:
            logger.error("YOLO model is not initialized.")
            return []

        try:
            # Run Ultralytics YOLO tracking with ByteTrack and high-resolution settings
            results = self.model.track(
                source=frame,
                persist=persist,
                tracker=tracker_type,
                conf=self.conf_threshold,
                iou=DEFAULT_IOU_THRESHOLD,
                imgsz=self.imgsz,
                max_det=self.max_det,
                classes=self.target_classes,
                verbose=False
            )

            tracked_objects = []
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    classes = boxes.cls.cpu().numpy().astype(int)
                    confidences = boxes.conf.cpu().numpy()

                    if boxes.id is not None:
                        track_ids = boxes.id.cpu().numpy().astype(int)
                    else:
                        track_ids = [-1] * len(boxes)

                    for box, cls_id, conf, track_id in zip(xyxy, classes, confidences, track_ids):
                        if cls_id not in TARGET_VEHICLE_CLASSES:
                            continue

                        x1, y1, x2, y2 = map(int, box)
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        class_name = TARGET_VEHICLE_CLASSES[cls_id]

                        tracked_objects.append({
                            "id": int(track_id) if track_id is not None else -1,
                            "class_id": int(cls_id),
                            "class_name": class_name,
                            "confidence": round(float(conf), 2),
                            "bbox": [x1, y1, x2, y2],
                            "center": (cx, cy)
                        })

            return tracked_objects

        except Exception as e:
            logger.error(f"Error during detection & tracking: {e}")
            return []
