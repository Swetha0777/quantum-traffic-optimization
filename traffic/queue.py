"""
Module: Queue Detector

Detects queued vehicles based on speed threshold and dwell duration.
Works with both simulation DataFrames and real-time tracking data.

A vehicle is considered queued if:
    - Its speed is below a minimum threshold (nearly stopped)
    - It has been stationary for longer than the minimum dwell time
"""

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Minimum speed (km/h) below which a vehicle is considered stopped
QUEUE_SPEED_THRESHOLD: float = 5.0

# Minimum dwell time (seconds) before counting as queued
MIN_DWELL_TIME: float = 2.0


# ============================================================
# DETECT QUEUE FROM DATAFRAME
# ============================================================

def detect_queue_from_dataframe(
    traffic_data: pd.DataFrame,
    speed_column: str = "speed_kmh",
    speed_threshold: float = QUEUE_SPEED_THRESHOLD
) -> pd.DataFrame:
    """
    Identify queued intersections from a traffic DataFrame.

    An intersection is flagged as having a queue when
    its average speed is below the threshold.

    Parameters
    ----------
    traffic_data : pd.DataFrame
        Traffic data with speed information.
    speed_column : str
        Column containing speed values.
    speed_threshold : float
        Speed below which vehicles are considered stopped.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with 'is_queued' boolean and
        'queue_severity' (0.0 to 1.0) columns.
    """

    data = traffic_data.copy()

    if speed_column in data.columns:
        speed = pd.to_numeric(
            data[speed_column],
            errors="coerce"
        ).fillna(50.0)
    else:
        speed = pd.Series(
            [50.0] * len(data),
            index=data.index
        )

    # Flag queued intersections
    data["is_queued"] = speed < speed_threshold

    # Queue severity: lower speed = higher severity
    # Normalized to 0-1 range (50 km/h free flow baseline)
    severity = (1.0 - speed / 50.0).clip(0.0, 1.0)
    data["queue_severity"] = severity.round(4)

    return data


# ============================================================
# ESTIMATE QUEUE LENGTH FROM DENSITY AND WAITING TIME
# ============================================================

def estimate_queue_length(
    density: float,
    waiting_time: float,
    road_capacity: int = 400
) -> int:
    """
    Estimate queue length from density and waiting time.

    Parameters
    ----------
    density : float
        Vehicle density (0.0 to 1.0).
    waiting_time : float
        Average waiting time in seconds.
    road_capacity : int
        Road capacity.

    Returns
    -------
    int
        Estimated number of queued vehicles.
    """

    # Higher density and longer waiting time = more queued vehicles
    queue_factor = density * 0.6 + min(waiting_time / 180.0, 1.0) * 0.4

    queue_length = int(
        road_capacity * queue_factor * 0.2
    )

    return max(0, queue_length)


# ============================================================
# AGGREGATE QUEUE METRICS
# ============================================================

def aggregate_queue_metrics(
    traffic_data: pd.DataFrame
) -> dict:
    """
    Compute aggregate queue metrics across all intersections.

    Returns
    -------
    dict
        {
            'total_queue_length': int,
            'queued_intersections': int,
            'average_queue_severity': float,
            'max_queue_length': int
        }
    """

    queue_col = "queue_length"

    if queue_col in traffic_data.columns:
        queue = pd.to_numeric(
            traffic_data[queue_col],
            errors="coerce"
        ).fillna(0)
    else:
        queue = pd.Series(
            [0] * len(traffic_data),
            index=traffic_data.index
        )

    total_queue = int(queue.sum())
    max_queue = int(queue.max()) if len(queue) > 0 else 0

    # Count intersections with non-trivial queues
    queued_count = int((queue > 3).sum())

    # Average severity if available
    if "queue_severity" in traffic_data.columns:
        severity = pd.to_numeric(
            traffic_data["queue_severity"],
            errors="coerce"
        ).fillna(0)
        avg_severity = round(float(severity.mean()), 4)
    else:
        avg_severity = 0.0

    return {
        "total_queue_length": total_queue,
        "queued_intersections": queued_count,
        "average_queue_severity": avg_severity,
        "max_queue_length": max_queue
    }
