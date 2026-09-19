"""
Module: Speed Estimator

Estimates vehicle speed from bounding box displacement
between frames and video FPS.

Speed is computed in pixels/second, then optionally
converted to km/h using a configurable pixels-per-meter ratio.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# ============================================================
# DEFAULT CALIBRATION
# ============================================================

# Default conversion factor: pixels per meter
# This should be calibrated per camera/scene
DEFAULT_PIXELS_PER_METER: float = 8.0

# Conversion factor: m/s to km/h
MS_TO_KMH: float = 3.6


# ============================================================
# SPEED ESTIMATOR
# ============================================================

class SpeedEstimator:
    """
    Estimates vehicle speed from tracked centroid positions
    across consecutive frames.
    """

    def __init__(
        self,
        pixels_per_meter: float = DEFAULT_PIXELS_PER_METER,
        smoothing_window: int = 5
    ):
        self.pixels_per_meter = pixels_per_meter
        self.smoothing_window = smoothing_window

        # Store per-track speed history for smoothing
        self._speed_history: Dict[int, List[float]] = {}

    def estimate_speed(
        self,
        track_id: int,
        prev_center: Tuple[int, int],
        curr_center: Tuple[int, int],
        time_delta: float
    ) -> float:
        """
        Estimate speed in km/h for a tracked vehicle.

        Parameters
        ----------
        track_id : int
            Unique tracking ID.
        prev_center : tuple
            Previous (cx, cy) position.
        curr_center : tuple
            Current (cx, cy) position.
        time_delta : float
            Time between frames in seconds.

        Returns
        -------
        float
            Estimated speed in km/h.
        """

        if time_delta <= 0.001:
            return 0.0

        # Pixel displacement
        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]
        pixel_distance = np.sqrt(dx**2 + dy**2)

        # Convert to real-world speed
        meters = pixel_distance / max(
            1.0, self.pixels_per_meter
        )

        speed_ms = meters / time_delta
        speed_kmh = speed_ms * MS_TO_KMH

        # Apply smoothing
        if track_id not in self._speed_history:
            self._speed_history[track_id] = []

        self._speed_history[track_id].append(speed_kmh)

        # Keep only last N entries
        if len(self._speed_history[track_id]) > self.smoothing_window:
            self._speed_history[track_id].pop(0)

        # Return smoothed speed
        smoothed = float(np.mean(
            self._speed_history[track_id]
        ))

        return round(max(0.0, smoothed), 2)

    def get_speed(self, track_id: int) -> float:
        """
        Get the latest smoothed speed for a track ID.

        Returns
        -------
        float
            Speed in km/h, or 0.0 if not available.
        """

        history = self._speed_history.get(track_id, [])

        if not history:
            return 0.0

        return round(
            float(np.mean(history)),
            2
        )

    def is_stationary(
        self,
        track_id: int,
        threshold_kmh: float = 5.0
    ) -> bool:
        """
        Check if a vehicle is stationary (below speed threshold).
        """

        return self.get_speed(track_id) < threshold_kmh

    def cleanup(self, active_ids: set) -> None:
        """
        Remove speed history for tracks no longer active.
        """

        stale = [
            tid for tid in self._speed_history
            if tid not in active_ids
        ]

        for tid in stale:
            del self._speed_history[tid]

    def reset(self) -> None:
        """Reset all speed tracking data."""
        self._speed_history.clear()
