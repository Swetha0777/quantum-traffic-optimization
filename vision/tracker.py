"""
Vehicle tracker state management module.
Tracks trajectory history, stationary frame counts, and frame detection count for track confirmation.
"""

import time
from typing import Dict, Tuple, List, Optional
import numpy as np


class VehicleTrackerState:
    """Stores per-vehicle tracking state history and frame observation count across frames."""

    def __init__(self, history_length: int = 15):
        self.history_length = history_length
        # Map: track_id -> List of (cx, cy, timestamp)
        self.positions: Dict[int, List[Tuple[int, int, float]]] = {}
        # Map: track_id -> total frame count observed
        self.detection_count: Dict[int, int] = {}
        # Map: track_id -> timestamp when vehicle was first seen
        self.first_seen: Dict[int, float] = {}
        # Map: track_id -> timestamp of last update
        self.last_seen: Dict[int, float] = {}
        # Map: track_id -> total stationary duration (seconds)
        self.stationary_duration: Dict[int, float] = {}

    def update(self, track_id: int, center: Tuple[int, int], current_time: float) -> None:
        """Update position history and increment frame observation count for a given vehicle ID."""
        if track_id < 0:
            return

        if track_id not in self.positions:
            self.positions[track_id] = []
            self.first_seen[track_id] = current_time
            self.stationary_duration[track_id] = 0.0
            self.detection_count[track_id] = 0

        self.detection_count[track_id] += 1
        self.positions[track_id].append((center[0], center[1], current_time))
        self.last_seen[track_id] = current_time

        # Trim history length
        if len(self.positions[track_id]) > self.history_length:
            self.positions[track_id].pop(0)

        # Update stationary duration calculation
        self._update_stationary_state(track_id, current_time)

    def is_confirmed(self, track_id: int, min_frames: int = 3) -> bool:
        """
        Check if a track ID has been observed for enough frames to be confirmed as a valid vehicle.
        Prevents 1-frame false positives from inflating unique vehicle counts.
        """
        if track_id < 0:
            return False
        return self.detection_count.get(track_id, 0) >= min_frames

    def get_observation_count(self, track_id: int) -> int:
        """Get total frames observed for a track ID."""
        return self.detection_count.get(track_id, 0)

    def _update_stationary_state(self, track_id: int, current_time: float) -> None:
        """Calculate displacement over recent positions to update stationary time."""
        history = self.positions.get(track_id, [])
        if len(history) < 2:
            return

        cx_curr, cy_curr, t_curr = history[-1]
        cx_prev, cy_prev, t_prev = history[0]

        dt = t_curr - t_prev
        if dt > 0.1:
            dist = np.sqrt((cx_curr - cx_prev)**2 + (cy_curr - cy_prev)**2)
            speed = dist / dt  # pixels per second

            if speed < 15.0:
                dt_step = t_curr - history[-2][2] if len(history) >= 2 else 0.033
                self.stationary_duration[track_id] += max(0.0, dt_step)
            else:
                self.stationary_duration[track_id] = 0.0

    def get_stationary_duration(self, track_id: int) -> float:
        """Get stationary duration in seconds for a track ID."""
        return self.stationary_duration.get(track_id, 0.0)

    def cleanup_stale_tracks(self, current_time: float, max_age_sec: float = 5.0) -> None:
        """Remove tracks that haven't been updated in max_age_sec."""
        stale_ids = [
            tid for tid, last_t in self.last_seen.items()
            if (current_time - last_t) > max_age_sec
        ]
        for tid in stale_ids:
            self.positions.pop(tid, None)
            self.detection_count.pop(tid, None)
            self.first_seen.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.stationary_duration.pop(tid, None)

    def reset(self) -> None:
        """Clear all tracking state."""
        self.positions.clear()
        self.detection_count.clear()
        self.first_seen.clear()
        self.last_seen.clear()
        self.stationary_duration.clear()
