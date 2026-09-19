"""
Module: Performance & Metrics Evaluation

Purpose:
    Calculate traffic performance before and after optimization.

Metrics:
    - Total traffic
    - Queue length
    - Average waiting time
    - Traffic density
    - Throughput
    - Congestion level
    - Estimated fuel consumption
    - Estimated CO2 emissions
    - Emergency corridor waiting time
    - Emergency response delay
    - Improvement percentage
"""

import numpy as np
import pandas as pd


# ============================================================
# DEFAULT PARAMETERS
# ============================================================

DEFAULT_CAPACITY = 400

# Approximate fuel consumption used for simulation estimation
FUEL_IDLE_RATE = 0.08

# kg CO2 produced per litre of fuel
CO2_PER_LITRE = 2.31


# ============================================================
# SAFE COLUMN HELPERS
# ============================================================

def _get_column(
    df: pd.DataFrame,
    column: str,
    default=0
):
    """
    Safely return a DataFrame column.

    If the column does not exist, return a default Series.
    """

    if column in df.columns:
        return pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(default)

    return pd.Series(
        [default] * len(df),
        index=df.index
    )


def _get_bool_column(
    df: pd.DataFrame,
    column: str
):
    """
    Safely read boolean columns.
    """

    if column not in df.columns:

        return pd.Series(
            [False] * len(df),
            index=df.index
        )

    return (
        df[column]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
                "active"
            ]
        )
    )


# ============================================================
# TOTAL TRAFFIC
# ============================================================

def calculate_total_traffic(
    traffic_data: pd.DataFrame
) -> float:

    north = _get_column(
        traffic_data,
        "north_traffic"
    )

    south = _get_column(
        traffic_data,
        "south_traffic"
    )

    east = _get_column(
        traffic_data,
        "east_traffic"
    )

    west = _get_column(
        traffic_data,
        "west_traffic"
    )

    return float(
        (
            north
            + south
            + east
            + west
        ).sum()
    )


# ============================================================
# QUEUE LENGTH
# ============================================================

def calculate_total_queue(
    traffic_data: pd.DataFrame
) -> float:

    queue = _get_column(
        traffic_data,
        "queue_length"
    )

    return float(
        queue.sum()
    )


# ============================================================
# AVERAGE WAITING TIME
# ============================================================

def calculate_average_waiting_time(
    traffic_data: pd.DataFrame
) -> float:

    waiting = _get_column(
        traffic_data,
        "waiting_time"
    )

    if len(waiting) == 0:
        return 0.0

    return float(
        waiting.mean()
    )


# ============================================================
# AVERAGE DENSITY
# ============================================================

def calculate_average_density(
    traffic_data: pd.DataFrame
) -> float:

    density = _get_column(
        traffic_data,
        "vehicle_density"
    )

    if len(density) == 0:
        return 0.0

    return float(
        density.mean()
    )


# ============================================================
# CONGESTION LEVEL
# ============================================================

def calculate_congestion_level(
    density: float
) -> str:

    if density < 0.35:

        return "LOW"

    elif density < 0.65:

        return "MEDIUM"

    elif density < 0.85:

        return "HIGH"

    return "CRITICAL"


# ============================================================
# THROUGHPUT
# ============================================================

def calculate_throughput(
    traffic_data: pd.DataFrame
) -> float:

    capacity = _get_column(
        traffic_data,
        "road_capacity",
        DEFAULT_CAPACITY
    )

    density = _get_column(
        traffic_data,
        "vehicle_density"
    )

    # Effective available capacity
    throughput = (
        capacity
        *
        (
            1.0
            -
            density
        )
    )

    throughput = throughput.clip(
        lower=0
    )

    return float(
        throughput.sum()
    )


# ============================================================
# ESTIMATED FUEL CONSUMPTION
# ============================================================

def calculate_fuel_consumption(
    traffic_data: pd.DataFrame
) -> float:

    queue = _get_column(
        traffic_data,
        "queue_length"
    )

    waiting = _get_column(
        traffic_data,
        "waiting_time"
    )

    # Fuel consumed while vehicles are
    # stopped / delayed.
    #
    # This is a simulation estimate, not
    # a real vehicle fuel measurement.

    idle_fuel = (
        queue
        *
        FUEL_IDLE_RATE
    )

    delay_fuel = (
        waiting
        *
        0.01
    )

    total_fuel = (
        idle_fuel
        +
        delay_fuel
    ).sum()

    return float(
        total_fuel
    )


# ============================================================
# CO2 EMISSION
# ============================================================

def calculate_co2_emissions(
    fuel_litres: float
) -> float:

    return float(
        fuel_litres
        *
        CO2_PER_LITRE
    )


# ============================================================
# EMERGENCY WAITING TIME
# ============================================================

def calculate_emergency_waiting_time(
    traffic_data: pd.DataFrame
) -> float:

    emergency_route = _get_bool_column(
        traffic_data,
        "is_emergency_route"
    )

    if not emergency_route.any():

        return 0.0

    waiting = _get_column(
        traffic_data,
        "waiting_time"
    )

    selected = waiting[
        emergency_route
    ]

    if len(selected) == 0:

        return 0.0

    return float(
        selected.mean()
    )


# ============================================================
# EMERGENCY RESPONSE DELAY
# ============================================================

def calculate_emergency_response_delay(
    traffic_data: pd.DataFrame
) -> float:

    """
    Estimates emergency response delay.

    If emergency-specific waiting data exists,
    it is used.

    Otherwise the emergency corridor waiting
    time is used as the simulation estimate.
    """

    if (
        "emergency_waiting_time"
        in traffic_data.columns
    ):

        values = _get_column(
            traffic_data,
            "emergency_waiting_time"
        )

        return float(
            values.sum()
        )

    return calculate_emergency_waiting_time(
        traffic_data
    )


# ============================================================
# FULL PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    traffic_data: pd.DataFrame
) -> dict:

    """
    Calculate all performance metrics.
    """

    total_traffic = (
        calculate_total_traffic(
            traffic_data
        )
    )

    total_queue = (
        calculate_total_queue(
            traffic_data
        )
    )

    average_waiting = (
        calculate_average_waiting_time(
            traffic_data
        )
    )

    average_density = (
        calculate_average_density(
            traffic_data
        )
    )

    congestion = (
        calculate_congestion_level(
            average_density
        )
    )

    throughput = (
        calculate_throughput(
            traffic_data
        )
    )

    fuel = (
        calculate_fuel_consumption(
            traffic_data
        )
    )

    co2 = (
        calculate_co2_emissions(
            fuel
        )
    )

    emergency_waiting = (
        calculate_emergency_waiting_time(
            traffic_data
        )
    )

    emergency_delay = (
        calculate_emergency_response_delay(
            traffic_data
        )
    )

    return {

        "total_traffic":
            round(
                total_traffic,
                2
            ),

        "total_queue_length":
            round(
                total_queue,
                2
            ),

        "average_waiting_time":
            round(
                average_waiting,
                2
            ),

        "average_density":
            round(
                average_density,
                4
            ),

        "congestion_level":
            congestion,

        "estimated_throughput":
            round(
                throughput,
                2
            ),

        "estimated_fuel_litres":
            round(
                fuel,
                2
            ),

        "estimated_co2_kg":
            round(
                co2,
                2
            ),

        "emergency_corridor_waiting_time":
            round(
                emergency_waiting,
                2
            ),

        "emergency_response_delay":
            round(
                emergency_delay,
                2
            )
    }


# ============================================================
# BEFORE VS AFTER COMPARISON
# ============================================================

def compare_metrics(
    before_data: pd.DataFrame,
    after_data: pd.DataFrame
) -> dict:

    """
    Compare traffic performance before and after
    optimization.

    Positive improvement means:
        - Queue decreased
        - Waiting time decreased
        - Density decreased
        - Fuel decreased
        - CO2 decreased

    For throughput:
        increase is considered improvement.
    """

    before = calculate_metrics(
        before_data
    )

    after = calculate_metrics(
        after_data
    )

    comparison = {}

    # --------------------------------------------------------
    # LOWER IS BETTER
    # --------------------------------------------------------

    lower_is_better = [

        "total_queue_length",

        "average_waiting_time",

        "average_density",

        "estimated_fuel_litres",

        "estimated_co2_kg",

        "emergency_corridor_waiting_time",

        "emergency_response_delay"
    ]

    for metric in lower_is_better:

        old = float(
            before[metric]
        )

        new = float(
            after[metric]
        )

        if abs(old) < 1e-9:

            improvement = 0.0

        else:

            improvement = (
                (old - new)
                /
                abs(old)
                *
                100
            )

        comparison[metric] = {

            "before":
                round(old, 2),

            "after":
                round(new, 2),

            "change":
                round(
                    new - old,
                    2
                ),

            "improvement_percent":
                round(
                    improvement,
                    2
                )
        }

    # --------------------------------------------------------
    # HIGHER IS BETTER
    # --------------------------------------------------------

    old = float(
        before[
            "estimated_throughput"
        ]
    )

    new = float(
        after[
            "estimated_throughput"
        ]
    )

    if abs(old) < 1e-9:

        improvement = 0.0

    else:

        improvement = (
            (new - old)
            /
            abs(old)
            *
            100
        )

    comparison[
        "estimated_throughput"
    ] = {

        "before":
            round(
                old,
                2
            ),

        "after":
            round(
                new,
                2
            ),

        "change":
            round(
                new - old,
                2
            ),

        "improvement_percent":
            round(
                improvement,
                2
            )
    }

    # --------------------------------------------------------
    # CONGESTION
    # --------------------------------------------------------

    comparison[
        "congestion_level"
    ] = {

        "before":
            before[
                "congestion_level"
            ],

        "after":
            after[
                "congestion_level"
            ]
    }

    return comparison


# ============================================================
# INTERSECTION-LEVEL METRICS
# ============================================================

def intersection_metrics(
    traffic_data: pd.DataFrame
) -> pd.DataFrame:

    """
    Generate performance metrics for every intersection.
    """

    if traffic_data.empty:

        return pd.DataFrame()

    data = traffic_data.copy()

    density = _get_column(
        data,
        "vehicle_density"
    )

    queue = _get_column(
        data,
        "queue_length"
    )

    waiting = _get_column(
        data,
        "waiting_time"
    )

    capacity = _get_column(
        data,
        "road_capacity",
        DEFAULT_CAPACITY
    )

    result = pd.DataFrame()

    result[
        "intersection_id"
    ] = data[
        "intersection_id"
    ].astype(str)

    result[
        "queue_length"
    ] = queue.round(2)

    result[
        "waiting_time"
    ] = waiting.round(2)

    result[
        "density"
    ] = density.round(4)

    result[
        "throughput"
    ] = (
        capacity
        *
        (
            1
            -
            density
        )
    ).clip(
        lower=0
    ).round(2)

    result[
        "congestion_level"
    ] = density.apply(
        calculate_congestion_level
    )

    result[
        "emergency_route"
    ] = _get_bool_column(
        data,
        "is_emergency_route"
    )

    return result


# ============================================================
# PRINT REPORT
# ============================================================

def print_metrics_report(
    traffic_data: pd.DataFrame,
    title="Traffic Performance"
):

    metrics = calculate_metrics(
        traffic_data
    )

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    for key, value in metrics.items():

        print(
            f"{key:40}: {value}"
        )

    print("=" * 60)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    from simulation.traffic_simulation import (
        generate_traffic
    )

    scenarios = [

        "Normal Traffic",

        "Heavy Congestion",

        "Accident / Road Closure",

        "Emergency Vehicle"
    ]

    for scenario in scenarios:

        print()

        print(
            f"SCENARIO: {scenario}"
        )

        data = generate_traffic(
            scenario
        )

        print_metrics_report(
            data,
            title=scenario
        )