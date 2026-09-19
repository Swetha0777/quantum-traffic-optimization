"""
Traffic Analytics module.
Computes traffic state (LOW/MODERATE/HIGH/VERY HIGH), vehicle density ratio,
estimated queue length, and estimated average waiting time from active frame vehicle tracking data.
"""

from typing import Dict, List
import numpy as np

from vision.config import (
    TRAFFIC_STATE_THRESHOLDS,
    MAX_FRAME_CAPACITY,
    STATIONARY_TIME_THRESHOLD_SEC
)
from vision.tracker import VehicleTrackerState


class TrafficAnalyzer:
    """Computes traffic metrics and estimates from real video tracking data."""

    def __init__(self):
        pass

    def analyze(self, tracked_objects: List[Dict], tracker_state: VehicleTrackerState) -> Dict:
        """
        Analyze current frame active tracked objects and return traffic metrics.

        Args:
            tracked_objects: List of active vehicle dicts from detector in current frame.
            tracker_state: VehicleTrackerState instance tracking durations.

        Returns:
            Dict containing estimated traffic analytics:
            {
                'active_vehicles': int,
                'state': str,
                'density': float,
                'estimated_queue_length': int,
                'estimated_average_waiting_time': float
            }
        """
        active_count = len(tracked_objects)

        # 1. Determine Traffic State
        state = "LOW"
        if active_count <= TRAFFIC_STATE_THRESHOLDS["LOW"]:
            state = "LOW"
        elif active_count <= TRAFFIC_STATE_THRESHOLDS["MODERATE"]:
            state = "MODERATE"
        elif active_count <= TRAFFIC_STATE_THRESHOLDS["HIGH"]:
            state = "HIGH"
        else:
            state = "VERY HIGH"

        # 2. Compute Density (0.0 to 1.0 ratio)
        density = round(min(1.0, active_count / float(MAX_FRAME_CAPACITY)), 2)

        # 3. Queue & Waiting Time estimation
        waiting_times: List[float] = []
        queue_count = 0

        for obj in tracked_objects:
            tid = obj.get("id", -1)
            if tid >= 0:
                duration = tracker_state.get_stationary_duration(tid)
                if duration >= STATIONARY_TIME_THRESHOLD_SEC:
                    queue_count += 1
                    waiting_times.append(duration)

        avg_waiting_time = round(float(np.mean(waiting_times)), 1) if waiting_times else 0.0

        return {
            "active_vehicles": active_count,
            "state": state,
            "density": density,
            "estimated_queue_length": queue_count,
            "estimated_average_waiting_time": avg_waiting_time
        }
