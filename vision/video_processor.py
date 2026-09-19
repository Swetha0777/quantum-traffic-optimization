"""
Video processor pipeline module.
Integrates detection, ByteTrack tracking, track confirmation, traffic analytics,
and OpenCV visual frame annotations into a real-time frame-by-frame stream processor.
"""

import cv2
import time
import base64
import logging
from typing import Dict, Tuple, Optional, List
import numpy as np

from vision.config import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMGSZ,
    DEFAULT_MAX_DET,
    DEFAULT_CONFIRMATION_FRAMES,
    CLASS_COLORS,
    CONFIRMED_COLOR
)
from vision.detector import VehicleDetector
from vision.tracker import VehicleTrackerState
from vision.counter import UniqueVehicleCounter
from vision.traffic_analyzer import TrafficAnalyzer

logger = logging.getLogger(__name__)


class VideoStreamProcessor:
    """Frame-by-frame video & webcam traffic analysis processor."""

    def __init__(
        self,
        conf_thresh: float = DEFAULT_CONF_THRESHOLD,
        imgsz: int = DEFAULT_IMGSZ,
        max_det: int = DEFAULT_MAX_DET,
        confirmation_frames: int = DEFAULT_CONFIRMATION_FRAMES
    ):
        self.detector = VehicleDetector(
            conf_threshold=conf_thresh,
            imgsz=imgsz,
            max_det=max_det
        )
        self.tracker_state = VehicleTrackerState()
        self.counter = UniqueVehicleCounter(confirmation_frames=confirmation_frames)
        self.analyzer = TrafficAnalyzer()

        # Performance timing
        self.last_proc_time: float = time.time()
        self.processing_fps: float = 0.0

    def update_config(
        self,
        conf_thresh: Optional[float] = None,
        imgsz: Optional[int] = None,
        max_det: Optional[int] = None,
        confirmation_frames: Optional[int] = None
    ) -> None:
        """Update processor settings dynamically."""
        self.detector.update_settings(
            conf_threshold=conf_thresh,
            imgsz=imgsz,
            max_det=max_det
        )
        if confirmation_frames is not None:
            self.counter.update_confirmation_frames(confirmation_frames)

    def reset(self) -> None:
        """Reset counter and tracker state."""
        self.tracker_state.reset()
        self.counter.reset()

    def process_frame(self, frame: np.ndarray, current_time: float, video_fps: float = 25.0) -> Tuple[np.ndarray, Dict]:
        """
        Process a single image frame through the detection, tracking, and confirmation pipeline.

        Returns:
            Tuple of (annotated_frame, telemetry_dict)
        """
        start_t = time.time()
        frame_h, frame_w = frame.shape[:2]

        # 1. Run YOLO detection & ByteTrack tracking (with configured imgsz & max_det)
        tracked_objects = self.detector.detect_and_track(frame)

        # 2. Update tracking state history
        processed_objects = []

        for obj in tracked_objects:
            track_id = obj["id"]
            class_name = obj["class_name"]
            curr_center = obj["center"]

            # Update tracker history and frame count
            self.tracker_state.update(track_id, curr_center, current_time)

            is_confirmed = self.tracker_state.is_confirmed(track_id, self.counter.confirmation_frames)
            stationary_dur = self.tracker_state.get_stationary_duration(track_id)

            processed_objects.append({
                "id": track_id,
                "class": class_name,
                "confidence": obj["confidence"],
                "bbox": obj["bbox"],
                "center": list(curr_center),
                "is_confirmed": is_confirmed,
                "waiting_time": round(stationary_dur, 1)
            })

        # Cleanup inactive stale tracks
        self.tracker_state.cleanup_stale_tracks(current_time)

        # 3. Process Active vs Unique Vehicle Counters
        active_summary, unique_summary = self.counter.process_frame_objects(
            processed_objects,
            self.tracker_state
        )

        # 4. Compute Traffic Analytics
        traffic_info = self.analyzer.analyze(processed_objects, self.tracker_state)

        # 5. Measure Processing FPS & Latency
        end_t = time.time()
        proc_dt = end_t - start_t
        proc_latency_ms = round(proc_dt * 1000, 1)

        dt_since_last = end_t - self.last_proc_time
        if dt_since_last > 0:
            inst_fps = 1.0 / dt_since_last
            self.processing_fps = round(0.8 * self.processing_fps + 0.2 * inst_fps, 1)
        self.last_proc_time = end_t

        # 6. Render Visual Overlay on frame (NO COUNTING LINE)
        annotated_frame = self._render_overlay(
            frame.copy(),
            processed_objects,
            active_summary["total"],
            unique_summary["total"],
            frame_h,
            frame_w
        )

        # 7. Assemble Telemetry JSON payload
        telemetry = {
            "active_vehicles": active_summary,
            "unique_vehicles": unique_summary,
            "traffic": {
                "active_vehicles": traffic_info["active_vehicles"],
                "state": traffic_info["state"],
                "density": traffic_info["density"],
                "estimated_queue_length": traffic_info["estimated_queue_length"],
                "estimated_average_waiting_time": traffic_info["estimated_average_waiting_time"]
            },
            "performance": {
                "processing_fps": self.processing_fps,
                "video_fps": round(video_fps, 1),
                "detection_latency_ms": proc_latency_ms
            },
            "config": {
                "conf_threshold": self.detector.conf_threshold,
                "imgsz": self.detector.imgsz,
                "max_det": self.detector.max_det,
                "confirmation_frames": self.counter.confirmation_frames
            },
            "tracked_objects": processed_objects
        }

        return annotated_frame, telemetry

    def _render_overlay(
        self,
        frame: np.ndarray,
        objects: list,
        active_total: int,
        unique_total: int,
        frame_h: int,
        frame_w: int
    ) -> np.ndarray:
        """Draw bounding boxes, track IDs, class tags, confidence scores, and watermark stats."""
        # 1. Draw Tracked Vehicles (NO COUNTING LINE)
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            cls_name = obj["class"]
            track_id = obj["id"]
            conf = obj["confidence"]
            cx, cy = obj["center"]
            is_confirmed = obj.get("is_confirmed", False)

            # Choose box color: Neon green if confirmed unique, else class color
            if is_confirmed:
                box_color = CONFIRMED_COLOR
                thickness = 2
            else:
                box_color = CLASS_COLORS.get(cls_name, (255, 255, 255))
                thickness = 2

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

            # Center point
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

            # Label tag: e.g. "CAR | ID:17 | 0.82"
            status_tag = " [CONFIRMED]" if is_confirmed else ""
            label_text = f"{cls_name.upper()} | ID:{track_id} | {conf}{status_tag}"

            # Label text background box
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - h - 6)), (x1 + w + 6, max(h + 6, y1)), box_color, -1)
            cv2.putText(
                frame,
                label_text,
                (x1 + 3, max(h, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        # 2. Top Watermark Overlay
        banner_text = f"ACTIVE: {active_total} | UNIQUE DETECTED: {unique_total}"
        cv2.putText(
            frame,
            banner_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        return frame

    @staticmethod
    def encode_frame_to_base64(frame: np.ndarray, quality: int = 80) -> str:
        """Compress BGR OpenCV frame to JPEG and encode to base64 string."""
        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return base64.b64encode(buffer).decode("utf-8")
