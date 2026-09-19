"""
Module: Accident Detector

Algorithmic accident detection based on traffic anomalies:
    - Sudden speed drops (vehicles abruptly stopping)
    - Queue growth rate exceeding normal thresholds
    - Cluster of stationary vehicles in close proximity
    - Abnormal congestion patterns

This module works with both simulation DataFrames
and real-time tracking data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# ============================================================
# DETECTION THRESHOLDS
# ============================================================

# Speed drop threshold: if average speed drops below this,
# it may indicate an incident
SPEED_DROP_THRESHOLD: float = 8.0  # km/h

# Queue growth rate threshold: if queue grows faster than
# this many vehicles per cycle, flag as anomaly
QUEUE_GROWTH_THRESHOLD: int = 15

# Minimum stationary cluster size to trigger accident alert
MIN_CLUSTER_SIZE: int = 4

# Congestion score threshold for accident suspicion
CONGESTION_ANOMALY_THRESHOLD: float = 80.0


# ============================================================
# ACCIDENT DETECTOR
# ============================================================

class AccidentDetector:
    """
    Detects potential accidents and road blockages from
    traffic data patterns.
    """

    def __init__(self):
        # Previous cycle data for comparison
        self._previous_queues: Dict[str, int] = {}
        self._alert_history: List[Dict] = []

    def detect_from_dataframe(
        self,
        traffic_data: pd.DataFrame
    ) -> List[Dict]:
        """
        Analyze traffic DataFrame for accident indicators.

        Parameters
        ----------
        traffic_data : pd.DataFrame
            Traffic data with columns: intersection_id,
            speed_kmh, queue_length, vehicle_density,
            blocked, accident_severity.

        Returns
        -------
        list
            List of accident alert dicts.
        """

        alerts = []

        for _, row in traffic_data.iterrows():

            intersection = str(
                row.get("intersection_id", "Unknown")
            )

            # Check explicit blocked flag
            blocked = str(
                row.get("blocked", False)
            ).lower() == "true"

            accident_severity = float(
                row.get("accident_severity", 0.0)
            )

            speed = float(
                row.get("speed_kmh", 50.0)
            )

            queue = int(
                row.get("queue_length", 0)
            )

            density = float(
                row.get("vehicle_density", 0.0)
            )

            # Scoring system
            accident_score = 0.0
            reasons = []

            # 1. Explicit block flag
            if blocked:
                accident_score += 40.0
                reasons.append(
                    f"Road blocked (severity: {accident_severity:.1f})"
                )

            # 2. Speed anomaly
            if speed < SPEED_DROP_THRESHOLD:
                accident_score += 25.0
                reasons.append(
                    f"Very low speed: {speed:.1f} km/h"
                )

            # 3. Queue growth
            prev_queue = self._previous_queues.get(
                intersection, 0
            )
            queue_growth = queue - prev_queue

            if queue_growth > QUEUE_GROWTH_THRESHOLD:
                accident_score += 20.0
                reasons.append(
                    f"Rapid queue growth: +{queue_growth} vehicles"
                )

            # 4. Extreme density
            if density > 0.90:
                accident_score += 15.0
                reasons.append(
                    f"Critical density: {density:.2f}"
                )

            # Update previous queue
            self._previous_queues[intersection] = queue

            # Generate alert if score exceeds threshold
            if accident_score >= 25.0:
                severity_level = self._classify_severity(
                    accident_score
                )

                alert = {
                    "intersection_id": intersection,
                    "accident_score": round(accident_score, 1),
                    "severity": severity_level,
                    "blocked": blocked,
                    "reasons": reasons,
                    "recommended_action": (
                        self._recommend_action(severity_level)
                    )
                }

                alerts.append(alert)
                self._alert_history.append(alert)

        return alerts

    def detect_from_tracking(
        self,
        stationary_vehicles: List[Dict],
        min_cluster: int = MIN_CLUSTER_SIZE
    ) -> Optional[Dict]:
        """
        Detect accidents from real-time tracking data
        by finding clusters of stationary vehicles.

        Parameters
        ----------
        stationary_vehicles : list
            List of dicts with keys: track_id, center, duration.
        min_cluster : int
            Minimum cluster size to trigger alert.

        Returns
        -------
        dict or None
            Accident alert if detected.
        """

        if len(stationary_vehicles) < min_cluster:
            return None

        # Check if stationary vehicles are clustered
        centers = [
            v["center"] for v in stationary_vehicles
            if v.get("duration", 0) > 3.0
        ]

        if len(centers) < min_cluster:
            return None

        # Compute pairwise distances
        centers_array = np.array(centers)
        mean_center = centers_array.mean(axis=0)

        distances = np.sqrt(
            ((centers_array - mean_center) ** 2).sum(axis=1)
        )

        # If most vehicles are within a tight radius
        tight_cluster = int(
            (distances < 150).sum()
        )

        if tight_cluster >= min_cluster:
            return {
                "type": "POTENTIAL_ACCIDENT",
                "cluster_size": tight_cluster,
                "center": mean_center.tolist(),
                "severity": "HIGH" if tight_cluster >= 6 else "MODERATE",
                "recommended_action": (
                    "Alert emergency services. "
                    "Reroute traffic around affected area."
                )
            }

        return None

    @staticmethod
    def _classify_severity(score: float) -> str:
        """Classify accident severity from score."""

        if score >= 60:
            return "CRITICAL"
        elif score >= 40:
            return "HIGH"
        elif score >= 25:
            return "MODERATE"
        return "LOW"

    @staticmethod
    def _recommend_action(severity: str) -> str:
        """Generate recommended action based on severity."""

        actions = {
            "CRITICAL": (
                "IMMEDIATE: Dispatch emergency services. "
                "Block affected intersection. "
                "Activate emergency corridor bypass."
            ),
            "HIGH": (
                "URGENT: Alert traffic management. "
                "Prepare alternative routes. "
                "Reduce signal green time for blocked direction."
            ),
            "MODERATE": (
                "MONITOR: Increase monitoring frequency. "
                "Prepare contingency routing."
            ),
            "LOW": (
                "WATCH: Continue monitoring. "
                "No immediate action required."
            )
        }

        return actions.get(severity, "Monitor situation.")

    def get_alert_history(self) -> List[Dict]:
        """Return all historical alerts."""
        return self._alert_history.copy()

    def reset(self) -> None:
        """Reset detector state."""
        self._previous_queues.clear()
        self._alert_history.clear()
