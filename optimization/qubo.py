"""
Module: QUBO Formulation for Traffic Signal Optimization

Purpose:
    Convert traffic conditions at intersections into a QUBO matrix
    that can be solved by qaoa.py.

Decision variable:

    x_i = 1
        Increase green time at intersection i

    x_i = 0
        Keep current green time

Objective:
    Reduce waiting time and queue length while avoiding unnecessary
    signal extensions and excessive simultaneous changes.
"""

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_INTERSECTIONS = [
    "J1",
    "J2",
    "J3",
    "J4",
    "J5",
    "J6",
]

# Maximum number of intersections that can be changed
# simultaneously.
MAX_GREEN_EXTENSIONS = 4

# Weight parameters
WAITING_WEIGHT = 0.45
QUEUE_WEIGHT = 0.35
DENSITY_WEIGHT = 0.20

GREEN_CHANGE_COST = 0.35

EMERGENCY_BONUS = 2.5

ACCIDENT_BONUS = 1.5

NEIGHBOUR_CONFLICT_PENALTY = 0.15


# ============================================================
# NETWORK
# ============================================================

NETWORK = {
    "J1": ["J2", "J4"],
    "J2": ["J1", "J3", "J5"],
    "J3": ["J2", "J4", "J6"],
    "J4": ["J1", "J3", "J5"],
    "J5": ["J2", "J4", "J6"],
    "J6": ["J3", "J5"],
}


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize(
    value,
    minimum,
    maximum,
):
    """
    Normalize value to 0-1 range.
    """

    if maximum <= minimum:
        return 0.0

    result = (
        float(value) - minimum
    ) / (
        maximum - minimum
    )

    return float(
        np.clip(
            result,
            0.0,
            1.0,
        )
    )


# ============================================================
# SAFE NUMERIC VALUE
# ============================================================

def _numeric(
    row,
    column,
    default=0.0,
):
    """
    Safely extract numeric values from a row.
    """

    try:

        value = float(
            row.get(
                column,
                default,
            )
        )

        if not np.isfinite(value):
            return float(default)

        return value

    except (
        ValueError,
        TypeError,
    ):

        return float(default)


# ============================================================
# DEMAND SCORE
# ============================================================

def _calculate_demand_score(
    row,
):
    """
    Calculate how strongly an intersection needs
    additional green time.

    Higher score = higher priority.
    """

    waiting_time = _numeric(
        row,
        "waiting_time",
        0,
    )

    queue_length = _numeric(
        row,
        "queue_length",
        0,
    )

    density = _numeric(
        row,
        "vehicle_density",
        0,
    )

    # Normalize individual traffic indicators.

    waiting_score = _normalize(
        waiting_time,
        0,
        180,
    )

    queue_score = _normalize(
        queue_length,
        0,
        80,
    )

    density_score = np.clip(
        density,
        0.0,
        1.0,
    )

    demand = (

        WAITING_WEIGHT
        *
        waiting_score

        +

        QUEUE_WEIGHT
        *
        queue_score

        +

        DENSITY_WEIGHT
        *
        density_score
    )

    return float(
        np.clip(
            demand,
            0.0,
            1.0,
        )
    )


# ============================================================
# SPECIAL PRIORITY
# ============================================================

def _calculate_priority_bonus(
    row,
):
    """
    Give additional optimization priority to:

        Emergency corridor
        Accident / blocked roads
    """

    bonus = 0.0

    # Emergency route

    emergency_route = str(
        row.get(
            "is_emergency_route",
            False,
        )
    ).lower()

    emergency_active = str(
        row.get(
            "emergency_active",
            False,
        )
    ).lower()

    emergency_priority = str(
        row.get(
            "emergency_priority",
            "",
        )
    ).upper()

    if (
        emergency_route == "true"
        or
        emergency_active == "true"
        or
        emergency_priority in [
            "ACTIVE",
            "PREPARE",
            "CORRIDOR",
        ]
    ):

        bonus += EMERGENCY_BONUS

    # Accident / road closure

    blocked = str(
        row.get(
            "blocked",
            False,
        )
    ).lower()

    accident_severity = _numeric(
        row,
        "accident_severity",
        0,
    )

    if blocked == "true":

        bonus += (
            ACCIDENT_BONUS
            *
            max(
                0.0,
                min(
                    1.0,
                    accident_severity,
                ),
            )
        )

    return float(
        bonus
    )


# ============================================================
# CREATE QUBO
# ============================================================

def create_qubo(
    traffic_data: pd.DataFrame,
):
    """
    Convert traffic data into QUBO formulation.

    Returns
    -------
    dict

        {
            "Q": QUBO matrix,
            "intersections": intersection IDs,
            "demand_scores": demand information,
            "metadata": additional information
        }

    QUBO objective:

        H(x) = x^T Q x

    where:

        x_i = 1
            Increase green time

        x_i = 0
            Keep current signal
    """

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if traffic_data is None:

        raise ValueError(
            "traffic_data cannot be None."
        )

    if not isinstance(
        traffic_data,
        pd.DataFrame,
    ):

        raise TypeError(
            "traffic_data must be a pandas DataFrame."
        )

    if traffic_data.empty:

        raise ValueError(
            "traffic_data is empty."
        )

    if "intersection_id" not in traffic_data.columns:

        raise ValueError(
            "traffic_data must contain "
            "'intersection_id'."
        )

    # --------------------------------------------------------
    # INTERSECTIONS
    # --------------------------------------------------------

    intersections = (
        traffic_data[
            "intersection_id"
        ]
        .astype(str)
        .tolist()
    )

    # Remove duplicates while maintaining order.

    intersections = list(
        dict.fromkeys(
            intersections
        )
    )

    n = len(
        intersections
    )

    if n == 0:

        raise ValueError(
            "No intersections found."
        )

    # --------------------------------------------------------
    # Q MATRIX
    # --------------------------------------------------------

    Q = np.zeros(
        (
            n,
            n,
        ),
        dtype=float,
    )

    demand_scores = {}

    priority_scores = {}

    # --------------------------------------------------------
    # BUILD LINEAR TERMS
    # --------------------------------------------------------

    for i, intersection in enumerate(
        intersections
    ):

        matching_rows = traffic_data[
            traffic_data[
                "intersection_id"
            ].astype(str)
            ==
            intersection
        ]

        if matching_rows.empty:

            row = {}

        else:

            row = matching_rows.iloc[
                0
            ]

        # Traffic demand

        demand = _calculate_demand_score(
            row
        )

        # Emergency / accident priority

        priority = _calculate_priority_bonus(
            row
        )

        demand_scores[
            intersection
        ] = round(
            demand,
            4,
        )

        priority_scores[
            intersection
        ] = round(
            priority,
            4,
        )

        # ----------------------------------------------------
        # LINEAR QUBO COST
        # ----------------------------------------------------

        # Higher demand should make x=1 attractive.
        #
        # Therefore demand is subtracted.
        #
        # Green change has a small cost so that the system
        # doesn't unnecessarily change every signal.

        benefit = (
            demand
            +
            priority
        )

        linear_cost = (

            GREEN_CHANGE_COST
            -
            benefit
        )

        Q[
            i,
            i
        ] = linear_cost

    # --------------------------------------------------------
    # NEIGHBOUR INTERACTION
    # --------------------------------------------------------

    for i, intersection_a in enumerate(
        intersections
    ):

        neighbours = NETWORK.get(
            intersection_a,
            [],
        )

        for j in range(
            i + 1,
            n,
        ):

            intersection_b = intersections[
                j
            ]

            # Adjacent intersections

            if intersection_b in neighbours:

                Q[
                    i,
                    j
                ] += (
                    NEIGHBOUR_CONFLICT_PENALTY
                )

    # --------------------------------------------------------
    # LIMIT TOTAL GREEN EXTENSIONS
    # --------------------------------------------------------

    #
    # We cannot directly add a hard constraint with only
    # the current matrix structure.
    #
    # Instead, apply a soft penalty to discourage selecting
    # too many intersections.
    #

    extension_penalty = (
        0.10
    )

    for i in range(n):

        Q[
            i,
            i
        ] += extension_penalty

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "Q": Q,

        "intersections":
            intersections,

        "demand_scores":
            demand_scores,

        "priority_scores":
            priority_scores,

        "metadata": {

            "num_intersections":
                n,

            "objective":
                "Minimize traffic delay and queue cost",

            "decision_variable":
                "x_i = 1 means increase green time",

            "emergency_priority":
                "Enabled",

            "accident_priority":
                "Enabled",
        },
    }


# ============================================================
# QUBO COST
# ============================================================

def calculate_qubo_cost(
    Q,
    x,
):
    """
    Calculate:

        H(x) = x^T Q x
    """

    Q = np.asarray(
        Q,
        dtype=float,
    )

    x = np.asarray(
        x,
        dtype=int,
    )

    if Q.shape[0] != len(x):

        raise ValueError(
            "Q and x dimensions do not match."
        )

    return float(
        x.T @ Q @ x
    )


# ============================================================
# EXPLAIN SOLUTION
# ============================================================

def explain_solution(
    traffic_data,
    solution,
):
    """
    Explain which intersections were selected
    for green-time extension.
    """

    result = []

    for i, x in enumerate(
        solution
    ):

        if i >= len(
            traffic_data
        ):

            break

        row = traffic_data.iloc[
            i
        ]

        result.append({

            "intersection_id":
                row[
                    "intersection_id"
                ],

            "decision":
                (
                    "INCREASE_GREEN"
                    if int(x) == 1
                    else
                    "KEEP"
                ),

            "demand_score":
                round(
                    _calculate_demand_score(
                        row
                    ),
                    4,
                ),

            "waiting_time":
                _numeric(
                    row,
                    "waiting_time",
                ),

            "queue_length":
                _numeric(
                    row,
                    "queue_length",
                ),

            "vehicle_density":
                _numeric(
                    row,
                    "vehicle_density",
                ),
        })

    return pd.DataFrame(
        result
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    from simulation.traffic_simulation import (
        generate_traffic,
    )

    data = generate_traffic(
        "Heavy Congestion",
        seed=42,
    )

    result = create_qubo(
        data
    )

    print(
        "\nIntersections:"
    )

    print(
        result[
            "intersections"
        ]
    )

    print(
        "\nQUBO Matrix:"
    )

    print(
        np.round(
            result["Q"],
            3,
        )
    )

    print(
        "\nDemand Scores:"
    )

    print(
        result[
            "demand_scores"
        ]
    )

    print(
        "\nPriority Scores:"
    )

    print(
        result[
            "priority_scores"
        ]
    )