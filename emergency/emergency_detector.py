"""
Module: Emergency Vehicle Detector

Detects emergency vehicles (ambulance, fire truck, police)
from simulation flags or vision-based class detection.

In simulation mode:
    Reads emergency_active and is_emergency_route flags
    from the traffic DataFrame.

In vision mode:
    Maps detected vehicle classes to emergency types
    (if specialized detection model is available).
    Falls back to simulation flag detection.
"""

import pandas as pd
from typing import Dict, List, Optional


# ============================================================
# EMERGENCY VEHICLE TYPES
# ============================================================

EMERGENCY_TYPES = {
    "ambulance": {
        "priority": 1,
        "max_green_time": 60,
        "speed_target_kmh": 60,
    },
    "fire_truck": {
        "priority": 2,
        "max_green_time": 55,
        "speed_target_kmh": 50,
    },
    "police": {
        "priority": 3,
        "max_green_time": 50,
        "speed_target_kmh": 55,
    },
}


# ============================================================
# EMERGENCY VEHICLE DETECTOR
# ============================================================

class EmergencyVehicleDetector:
    """
    Detects and tracks emergency vehicles in the traffic system.
    """

    def __init__(self):
        self._active_emergencies: List[Dict] = []
        self._detection_history: List[Dict] = []

    def detect_from_dataframe(
        self,
        traffic_data: pd.DataFrame
    ) -> List[Dict]:
        """
        Detect emergency vehicles from simulation DataFrame flags.

        Checks for:
            - emergency_active column
            - is_emergency_route column
            - emergency_priority column

        Parameters
        ----------
        traffic_data : pd.DataFrame
            Traffic simulation data.

        Returns
        -------
        list
            List of emergency detection dicts.
        """

        detections = []

        for _, row in traffic_data.iterrows():

            intersection = str(
                row.get("intersection_id", "Unknown")
            )

            # Check emergency flags
            emergency_active = str(
                row.get("emergency_active", False)
            ).lower() == "true"

            is_route = str(
                row.get("is_emergency_route", False)
            ).lower() == "true"

            priority = str(
                row.get("emergency_priority", "NORMAL")
            ).upper()

            emergency_current = str(
                row.get("emergency_current", False)
            ).lower() == "true"

            if emergency_active or is_route:

                detection = {
                    "intersection_id": intersection,
                    "emergency_active": emergency_active,
                    "is_emergency_route": is_route,
                    "priority": priority,
                    "is_current": emergency_current,
                    "vehicle_type": "ambulance",
                    "recommended_signal": "NS_GREEN",
                    "recommended_green_time": (
                        60 if priority == "ACTIVE"
                        else 50 if priority == "PREPARE"
                        else 40
                    )
                }

                detections.append(detection)

        self._active_emergencies = detections
        self._detection_history.extend(detections)

        return detections

    def detect_from_tracking(
        self,
        tracked_objects: List[Dict]
    ) -> List[Dict]:
        """
        Detect emergency vehicles from real-time tracking data.

        Uses class name matching against emergency vehicle types.
        Standard YOLO COCO classes do not include emergency vehicles,
        so this provides a simulation-based fallback.

        Parameters
        ----------
        tracked_objects : list
            List of tracked vehicle dicts from detector.

        Returns
        -------
        list
            List of detected emergency vehicles.
        """

        detections = []

        for obj in tracked_objects:
            cls_name = obj.get("class_name", "").lower()

            # Check if the class name matches emergency types
            # Note: Standard YOLO COCO does not have ambulance/fire
            # This handles custom-trained models
            if cls_name in EMERGENCY_TYPES:
                detections.append({
                    "track_id": obj.get("id", -1),
                    "vehicle_type": cls_name,
                    "priority": EMERGENCY_TYPES[cls_name]["priority"],
                    "confidence": obj.get("confidence", 0.0),
                    "center": obj.get("center", [0, 0]),
                })

        return detections

    def has_active_emergency(self) -> bool:
        """Check if any emergency vehicle is currently active."""
        return len(self._active_emergencies) > 0

    def get_active_intersections(self) -> List[str]:
        """Get list of intersections with active emergency priority."""
        return [
            d["intersection_id"]
            for d in self._active_emergencies
            if d.get("is_current", False)
        ]

    def get_emergency_route(self) -> List[str]:
        """Get ordered list of intersections on emergency route."""
        return [
            d["intersection_id"]
            for d in self._active_emergencies
            if d.get("is_emergency_route", False)
        ]

    def get_detection_history(self) -> List[Dict]:
        """Return all historical emergency detections."""
        return self._detection_history.copy()

    def reset(self) -> None:
        """Reset detector state."""
        self._active_emergencies.clear()
        self._detection_history.clear()
