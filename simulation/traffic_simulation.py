"""
Module: Traffic Simulation Engine

Purpose:
    Generate realistic traffic states for the quantum traffic
    optimization system.

Output:
    Pandas DataFrame containing traffic conditions for
    six intersections.

Supported scenarios:
    - Normal Traffic
    - Heavy Congestion
    - Accident / Road Closure
    - Emergency Vehicle

The output is directly compatible with:
    optimization/qubo.py
    optimization/qaoa.py
    emergency/emergency_corridor.py
    metrics/performance.py
"""

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INTERSECTIONS = [
    "J1",
    "J2",
    "J3",
    "J4",
    "J5",
    "J6"
]

ROAD_CAPACITY = 400

RANDOM_SEED = 42


# ============================================================
# INTERSECTION NETWORK
# ============================================================

NETWORK = {
    "J1": ["J2", "J4"],
    "J2": ["J1", "J3", "J5"],
    "J3": ["J2", "J4", "J6"],
    "J4": ["J1", "J3", "J5"],
    "J5": ["J2", "J4", "J6"],
    "J6": ["J3", "J5"]
}


# ============================================================
# SCENARIO PARAMETERS
# ============================================================

SCENARIOS = {

    "Normal Traffic": {
        "density_min": 0.20,
        "density_max": 0.50,
        "queue_min": 3,
        "queue_max": 15,
        "waiting_min": 8,
        "waiting_max": 30,
        "emergency": False
    },

    "Heavy Congestion": {
        "density_min": 0.60,
        "density_max": 0.95,
        "queue_min": 15,
        "queue_max": 60,
        "waiting_min": 35,
        "waiting_max": 120,
        "emergency": False
    },

    "Accident / Road Closure": {
        "density_min": 0.70,
        "density_max": 1.00,
        "queue_min": 25,
        "queue_max": 80,
        "waiting_min": 50,
        "waiting_max": 180,
        "emergency": False
    },

    "Emergency Vehicle": {
        "density_min": 0.35,
        "density_max": 0.75,
        "queue_min": 8,
        "queue_max": 40,
        "waiting_min": 15,
        "waiting_max": 90,
        "emergency": True
    }
}


# ============================================================
# RANDOM GENERATOR
# ============================================================

def _get_rng(seed=None):

    if seed is None:
        seed = RANDOM_SEED

    return np.random.default_rng(seed)


# ============================================================
# SIGNAL GENERATION
# ============================================================

def _generate_signal(
    rng,
    index
):

    signals = [
        "NS_GREEN",
        "EW_GREEN"
    ]

    return signals[
        index % 2
    ]


# ============================================================
# TRAFFIC VOLUME
# ============================================================

def _calculate_traffic_volume(
    density,
    capacity,
    rng
):

    base_volume = (
        density
        *
        capacity
    )

    variation = rng.normal(
        0,
        capacity * 0.05
    )

    volume = (
        base_volume
        +
        variation
    )

    return int(
        max(
            0,
            volume
        )
    )


# ============================================================
# QUEUE CALCULATION
# ============================================================

def _calculate_queue(
    density,
    scenario,
    rng
):

    params = SCENARIOS[
        scenario
    ]

    base_queue = rng.uniform(
        params["queue_min"],
        params["queue_max"]
    )

    # Increase queue with density

    density_factor = (
        0.5
        +
        density
    )

    queue = (
        base_queue
        *
        density_factor
    )

    return int(
        max(
            0,
            round(queue)
        )
    )


# ============================================================
# WAITING TIME
# ============================================================

def _calculate_waiting_time(
    density,
    queue,
    scenario,
    rng
):

    params = SCENARIOS[
        scenario
    ]

    base_waiting = rng.uniform(
        params["waiting_min"],
        params["waiting_max"]
    )

    queue_effect = (
        queue
        *
        0.5
    )

    density_effect = (
        density
        *
        20
    )

    waiting = (
        base_waiting
        +
        queue_effect
        +
        density_effect
    )

    return round(
        max(
            0,
            waiting
        ),
        2
    )


# ============================================================
# TRAFFIC GENERATOR
# ============================================================

def generate_traffic(
    scenario="Normal Traffic",
    seed=None
):

    """
    Generate traffic data for six intersections.

    Parameters
    ----------
    scenario : str
        One of:
            Normal Traffic
            Heavy Congestion
            Accident / Road Closure
            Emergency Vehicle

    seed : int, optional
        Random seed for reproducible simulation.

    Returns
    -------
    pandas.DataFrame
    """

    if scenario not in SCENARIOS:

        raise ValueError(
            f"Unknown scenario: {scenario}. "
            f"Available scenarios: "
            f"{list(SCENARIOS.keys())}"
        )

    rng = _get_rng(seed)

    params = SCENARIOS[
        scenario
    ]

    rows = []

    for index, intersection in enumerate(
        INTERSECTIONS
    ):

        # ----------------------------------------------------
        # DENSITY
        # ----------------------------------------------------

        density = rng.uniform(
            params["density_min"],
            params["density_max"]
        )

        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------

        current_signal = _generate_signal(
            rng,
            index
        )

        # ----------------------------------------------------
        # TRAFFIC VOLUME
        # ----------------------------------------------------

        total_volume = _calculate_traffic_volume(
            density,
            ROAD_CAPACITY,
            rng
        )

        # ----------------------------------------------------
        # DIRECTIONAL DISTRIBUTION
        # ----------------------------------------------------

        north_ratio = rng.uniform(
            0.15,
            0.35
        )

        south_ratio = rng.uniform(
            0.15,
            0.35
        )

        east_ratio = rng.uniform(
            0.15,
            0.35
        )

        west_ratio = (
            1
            -
            north_ratio
            -
            south_ratio
            -
            east_ratio
        )

        # Prevent invalid negative values

        if west_ratio < 0.05:

            west_ratio = 0.05

        total_ratio = (
            north_ratio
            +
            south_ratio
            +
            east_ratio
            +
            west_ratio
        )

        north_ratio /= total_ratio
        south_ratio /= total_ratio
        east_ratio /= total_ratio
        west_ratio /= total_ratio

        north_traffic = int(
            total_volume
            *
            north_ratio
        )

        south_traffic = int(
            total_volume
            *
            south_ratio
        )

        east_traffic = int(
            total_volume
            *
            east_ratio
        )

        west_traffic = int(
            total_volume
            *
            west_ratio
        )

        # ----------------------------------------------------
        # QUEUE
        # ----------------------------------------------------

        queue_length = _calculate_queue(
            density,
            scenario,
            rng
        )

        # ----------------------------------------------------
        # WAITING
        # ----------------------------------------------------

        waiting_time = _calculate_waiting_time(
            density,
            queue_length,
            scenario,
            rng
        )

        # ----------------------------------------------------
        # ACCIDENT
        # ----------------------------------------------------

        blocked = False

        accident_severity = 0.0

        if scenario == "Accident / Road Closure":

            # Simulate one major blocked intersection

            if intersection == "J3":

                blocked = True

                accident_severity = 0.90

                queue_length = int(
                    queue_length * 1.5
                )

                waiting_time = round(
                    waiting_time * 1.5,
                    2
                )

        # ----------------------------------------------------
        # EMERGENCY
        # ----------------------------------------------------

        emergency_active = (
            scenario
            ==
            "Emergency Vehicle"
        )

        # Emergency corridor:
        #
        # J1 -> J2 -> J4

        is_emergency_route = (

            emergency_active
            and
            intersection in [
                "J1",
                "J2",
                "J4"
            ]
        )

        emergency_current = (

            emergency_active
            and
            intersection == "J1"
        )

        emergency_next = (

            emergency_active
            and
            intersection == "J2"
        )

        if emergency_current:

            # Current emergency intersection

            waiting_time *= 0.8

        # ----------------------------------------------------
        # SPEED
        # ----------------------------------------------------

        free_flow_speed = 50.0

        speed = (

            free_flow_speed
            *
            (
                1
                -
                density * 0.65
            )
        )

        if blocked:

            speed *= 0.25

        speed = max(
            5.0,
            speed
        )

        # ----------------------------------------------------
        # ROAD CAPACITY
        # ----------------------------------------------------

        road_capacity = ROAD_CAPACITY

        if blocked:

            road_capacity *= (
                1
                -
                accident_severity
            )

        road_capacity = max(
            20,
            road_capacity
        )

        # ----------------------------------------------------
        # THROUGHPUT
        # ----------------------------------------------------

        throughput = (

            road_capacity
            *
            (
                1
                -
                density
            )
        )

        throughput = max(
            0,
            throughput
        )

        # ----------------------------------------------------
        # FUEL ESTIMATE
        # ----------------------------------------------------

        fuel_consumption = (

            queue_length
            *
            0.08

            +

            waiting_time
            *
            0.01
        )

        # ----------------------------------------------------
        # CO2
        # ----------------------------------------------------

        co2_emission = (

            fuel_consumption
            *
            2.31
        )

        # ----------------------------------------------------
        # CONGESTION LEVEL
        # ----------------------------------------------------

        if density < 0.35:

            congestion_level = "LOW"

        elif density < 0.65:

            congestion_level = "MEDIUM"

        elif density < 0.85:

            congestion_level = "HIGH"

        else:

            congestion_level = "CRITICAL"

        # ----------------------------------------------------
        # ADD ROW
        # ----------------------------------------------------

        rows.append({

            "intersection_id":
                intersection,

            "current_signal":
                current_signal,

            "vehicle_density":
                round(
                    density,
                    4
                ),

            "queue_length":
                queue_length,

            "waiting_time":
                round(
                    waiting_time,
                    2
                ),

            "north_traffic":
                north_traffic,

            "south_traffic":
                south_traffic,

            "east_traffic":
                east_traffic,

            "west_traffic":
                west_traffic,

            "total_traffic":
                (
                    north_traffic
                    +
                    south_traffic
                    +
                    east_traffic
                    +
                    west_traffic
                ),

            "road_capacity":
                round(
                    road_capacity,
                    2
                ),

            "throughput":
                round(
                    throughput,
                    2
                ),

            "speed_kmh":
                round(
                    speed,
                    2
                ),

            "congestion_level":
                congestion_level,

            "fuel_consumption":
                round(
                    fuel_consumption,
                    3
                ),

            "co2_emission":
                round(
                    co2_emission,
                    3
                ),

            "blocked":
                blocked,

            "accident_severity":
                accident_severity,

            "emergency_active":
                emergency_active,

            "is_emergency_route":
                is_emergency_route,

            "emergency_current":
                emergency_current,

            "emergency_next":
                emergency_next,

            "emergency_priority":
                (
                    "ACTIVE"
                    if emergency_current
                    else
                    "PREPARE"
                    if emergency_next
                    else
                    "CORRIDOR"
                    if is_emergency_route
                    else
                    "NORMAL"
                )
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# APPLY OPTIMIZED SIGNALS
# ============================================================

def apply_optimization(
    traffic_data: pd.DataFrame,
    optimization_result: pd.DataFrame
):

    """
    Apply QAOA signal decisions to the simulated traffic.

    This creates a new DataFrame instead of modifying the
    original simulation data.
    """

    data = traffic_data.copy()

    if optimization_result is None:

        return data

    if optimization_result.empty:

        return data

    optimization_lookup = (

        optimization_result
        .set_index(
            "intersection_id"
        )
    )

    optimized_waiting = []

    optimized_queue = []

    optimized_throughput = []

    optimized_density = []

    for _, row in data.iterrows():

        intersection = str(
            row[
                "intersection_id"
            ]
        )

        original_waiting = float(
            row.get(
                "waiting_time",
                0
            )
        )

        original_queue = float(
            row.get(
                "queue_length",
                0
            )
        )

        original_density = float(
            row.get(
                "vehicle_density",
                0
            )
        )

        if intersection not in optimization_lookup.index:

            optimized_waiting.append(
                original_waiting
            )

            optimized_queue.append(
                original_queue
            )

            optimized_density.append(
                original_density
            )

            optimized_throughput.append(
                float(
                    row.get(
                        "throughput",
                        0
                    )
                )
            )

            continue

        result = optimization_lookup.loc[
            intersection
        ]

        decision = str(
            result.get(
                "decision",
                "KEEP"
            )
        )

        green_time = float(
            result.get(
                "optimized_green_time",
                30
            )
        )

        # ----------------------------------------------------
        # SIGNAL IMPROVEMENT
        # ----------------------------------------------------

        if decision == "INCREASE_GREEN":

            improvement_factor = min(
                0.40,
                max(
                    0.10,
                    (
                        green_time - 30
                    )
                    /
                    100
                )
            )

            new_waiting = (

                original_waiting
                *
                (
                    1
                    -
                    improvement_factor
                )
            )

            new_queue = (

                original_queue
                *
                (
                    1
                    -
                    improvement_factor
                    *
                    0.8
                )
            )

        elif decision == "EMERGENCY_PRIORITY":

            new_waiting = (
                original_waiting
                *
                0.35
            )

            new_queue = (
                original_queue
                *
                0.50
            )

        else:

            new_waiting = original_waiting

            new_queue = original_queue

        # ----------------------------------------------------
        # SAFETY LIMITS
        # ----------------------------------------------------

        new_waiting = max(
            0,
            new_waiting
        )

        new_queue = max(
            0,
            new_queue
        )

        # ----------------------------------------------------
        # NEW DENSITY
        # ----------------------------------------------------

        new_density = (

            new_queue
            /
            max(
                1,
                ROAD_CAPACITY * 0.15
            )
        )

        new_density = min(
            1.0,
            max(
                0.0,
                new_density
            )
        )

        # ----------------------------------------------------
        # NEW THROUGHPUT
        # ----------------------------------------------------

        capacity = float(
            row.get(
                "road_capacity",
                ROAD_CAPACITY
            )
        )

        new_throughput = (

            capacity
            *
            (
                1
                -
                new_density
            )
        )

        optimized_waiting.append(
            round(
                new_waiting,
                2
            )
        )

        optimized_queue.append(
            round(
                new_queue,
                2
            )
        )

        optimized_density.append(
            round(
                new_density,
                4
            )
        )

        optimized_throughput.append(
            round(
                max(
                    0,
                    new_throughput
                ),
                2
            )
        )

    data[
        "waiting_time"
    ] = optimized_waiting

    data[
        "queue_length"
    ] = optimized_queue

    data[
        "vehicle_density"
    ] = optimized_density

    data[
        "throughput"
    ] = optimized_throughput

    return data


# ============================================================
# SAVE SIMULATION DATA
# ============================================================

def save_simulation(
    traffic_data: pd.DataFrame,
    filepath="data/traffic_data.csv"
):

    traffic_data.to_csv(
        filepath,
        index=False
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    scenarios = [

        "Normal Traffic",

        "Heavy Congestion",

        "Accident / Road Closure",

        "Emergency Vehicle"
    ]

    for scenario in scenarios:

        print()
        print("=" * 70)

        print(
            f"SCENARIO: {scenario}"
        )

        print("=" * 70)

        data = generate_traffic(
            scenario
        )

        print(
            data[
                [
                    "intersection_id",
                    "vehicle_density",
                    "queue_length",
                    "waiting_time",
                    "throughput",
                    "congestion_level",
                    "emergency_priority"
                ]
            ].to_string(
                index=False
            )
        )