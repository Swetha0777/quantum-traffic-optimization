"""
Module: Traffic Density Calculator

Computes vehicle density ratio and classifies density levels
from traffic simulation or real-time detection data.

Density levels:
    LOW      -> density < 0.35
    MEDIUM   -> 0.35 <= density < 0.65
    HIGH     -> 0.65 <= density < 0.85
    CRITICAL -> density >= 0.85
"""

import numpy as np
import pandas as pd


# ============================================================
# DENSITY THRESHOLDS
# ============================================================

DENSITY_THRESHOLDS = {
    "LOW": 0.35,
    "MEDIUM": 0.65,
    "HIGH": 0.85,
}


# ============================================================
# CLASSIFY DENSITY LEVEL
# ============================================================

def classify_density(density: float) -> str:
    """
    Classify a density value (0.0 - 1.0) into a traffic level.

    Parameters
    ----------
    density : float
        Vehicle density ratio (0.0 to 1.0).

    Returns
    -------
    str
        One of: LOW, MEDIUM, HIGH, CRITICAL
    """

    density = float(np.clip(density, 0.0, 1.0))

    if density < DENSITY_THRESHOLDS["LOW"]:
        return "LOW"

    elif density < DENSITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"

    elif density < DENSITY_THRESHOLDS["HIGH"]:
        return "HIGH"

    return "CRITICAL"


# ============================================================
# COMPUTE DENSITY FROM COUNTS
# ============================================================

def compute_density(
    vehicle_count: int,
    road_capacity: int = 400
) -> float:
    """
    Compute density ratio from vehicle count and road capacity.

    Parameters
    ----------
    vehicle_count : int
        Number of vehicles detected or simulated.
    road_capacity : int
        Maximum road capacity.

    Returns
    -------
    float
        Density ratio (0.0 to 1.0).
    """

    if road_capacity <= 0:
        return 0.0

    return float(
        np.clip(
            vehicle_count / road_capacity,
            0.0,
            1.0
        )
    )


# ============================================================
# COMPUTE DENSITY FOR DATAFRAME
# ============================================================

def compute_density_for_dataframe(
    traffic_data: pd.DataFrame,
    capacity_column: str = "road_capacity",
    count_column: str = "total_traffic",
    default_capacity: int = 400
) -> pd.DataFrame:
    """
    Add density and density_level columns to traffic DataFrame.

    Parameters
    ----------
    traffic_data : pd.DataFrame
        Traffic data with vehicle counts per intersection.
    capacity_column : str
        Column name for road capacity.
    count_column : str
        Column name for vehicle count.
    default_capacity : int
        Default capacity if column is missing.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with 'computed_density' and 'density_level' columns.
    """

    data = traffic_data.copy()

    if count_column in data.columns:
        counts = pd.to_numeric(
            data[count_column],
            errors="coerce"
        ).fillna(0)
    else:
        counts = pd.Series(
            [0] * len(data),
            index=data.index
        )

    if capacity_column in data.columns:
        capacity = pd.to_numeric(
            data[capacity_column],
            errors="coerce"
        ).fillna(default_capacity)
    else:
        capacity = pd.Series(
            [default_capacity] * len(data),
            index=data.index
        )

    # Compute density ratio
    density = (counts / capacity.clip(lower=1)).clip(0.0, 1.0)

    data["computed_density"] = density.round(4)

    data["density_level"] = density.apply(
        classify_density
    )

    return data
