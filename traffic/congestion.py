"""
Module: Congestion Score Calculator

Computes an overall congestion score (0-100) and estimates
waiting time impact from traffic density, queue length,
and throughput data.

Congestion Score:
    0-25   -> LOW
    25-50  -> MODERATE
    50-75  -> HIGH
    75-100 -> CRITICAL
"""

import numpy as np
import pandas as pd


# ============================================================
# CONGESTION SCORE
# ============================================================

def compute_congestion_score(
    density: float,
    queue_length: int,
    waiting_time: float,
    road_capacity: int = 400
) -> float:
    """
    Compute a congestion score (0-100) for an intersection.

    Parameters
    ----------
    density : float
        Vehicle density (0.0 to 1.0).
    queue_length : int
        Number of queued vehicles.
    waiting_time : float
        Average waiting time in seconds.
    road_capacity : int
        Maximum road capacity.

    Returns
    -------
    float
        Congestion score (0.0 to 100.0).
    """

    # Density component (0-40)
    density_score = float(
        np.clip(density, 0.0, 1.0) * 40.0
    )

    # Queue component (0-35)
    queue_ratio = min(
        1.0,
        queue_length / max(1, road_capacity * 0.15)
    )
    queue_score = queue_ratio * 35.0

    # Waiting time component (0-25)
    waiting_ratio = min(
        1.0,
        waiting_time / 180.0
    )
    waiting_score = waiting_ratio * 25.0

    total = density_score + queue_score + waiting_score

    return round(
        float(np.clip(total, 0.0, 100.0)),
        2
    )


# ============================================================
# CLASSIFY CONGESTION
# ============================================================

def classify_congestion(score: float) -> str:
    """
    Classify congestion score into a level.

    Parameters
    ----------
    score : float
        Congestion score (0-100).

    Returns
    -------
    str
        One of: LOW, MODERATE, HIGH, CRITICAL
    """

    if score < 25:
        return "LOW"
    elif score < 50:
        return "MODERATE"
    elif score < 75:
        return "HIGH"
    return "CRITICAL"


# ============================================================
# COMPUTE CONGESTION FOR DATAFRAME
# ============================================================

def compute_congestion_for_dataframe(
    traffic_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Add congestion_score and congestion_classification to DataFrame.

    Parameters
    ----------
    traffic_data : pd.DataFrame
        Traffic data with density, queue, and waiting time columns.

    Returns
    -------
    pd.DataFrame
        Updated with 'congestion_score' and 'congestion_classification'.
    """

    data = traffic_data.copy()

    def _safe_float(col, default=0.0):
        if col in data.columns:
            return pd.to_numeric(
                data[col], errors="coerce"
            ).fillna(default)
        return pd.Series(
            [default] * len(data),
            index=data.index
        )

    density = _safe_float("vehicle_density")
    queue = _safe_float("queue_length")
    waiting = _safe_float("waiting_time")
    capacity = _safe_float("road_capacity", 400)

    scores = []
    for d, q, w, c in zip(density, queue, waiting, capacity):
        scores.append(
            compute_congestion_score(
                float(d),
                int(q),
                float(w),
                int(c)
            )
        )

    data["congestion_score"] = scores

    data["congestion_classification"] = [
        classify_congestion(s) for s in scores
    ]

    return data


# ============================================================
# ESTIMATE WAITING TIME IMPACT
# ============================================================

def estimate_waiting_time_impact(
    congestion_score: float,
    base_waiting_time: float = 30.0
) -> float:
    """
    Estimate the waiting time impact based on congestion score.

    Parameters
    ----------
    congestion_score : float
        Score from 0-100.
    base_waiting_time : float
        Baseline waiting time.

    Returns
    -------
    float
        Estimated waiting time in seconds.
    """

    # Higher congestion = exponentially more waiting
    multiplier = 1.0 + (congestion_score / 100.0) ** 1.5 * 3.0

    return round(
        base_waiting_time * multiplier,
        2
    )
