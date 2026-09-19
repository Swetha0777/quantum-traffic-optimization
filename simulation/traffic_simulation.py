"""
Module: Traffic Simulation
Assigned to: Team Member 1 (Simulation Lead)
Description: Handles generating traffic patterns and simulating network flows across scenarios for 6 interconnected intersections (J1-J6).
"""

import numpy as np
import pandas as pd


def generate_traffic(scenario: str = "Normal Traffic") -> pd.DataFrame:
    """
    Generate traffic data for 6 interconnected intersections (J1 to J6)
    based on the selected scenario.

    Parameters:
        scenario (str): Selected scenario. Options:
            - "Normal Traffic"
            - "Heavy Congestion"
            - "Accident / Road Closure"
            - "Emergency Vehicle"

    Returns:
        pd.DataFrame: DataFrame containing intersection traffic properties:
            - intersection_id
            - north_traffic
            - south_traffic
            - east_traffic
            - west_traffic
            - vehicle_density
            - queue_length
            - road_capacity
            - current_signal
            - waiting_time
            - emergency_active
            - is_emergency_route
    """
    intersections = ["J1", "J2", "J3", "J4", "J5", "J6"]

    # Deterministic seeds per scenario for reproducible results
    seed_map = {
        "Normal Traffic": 42,
        "Heavy Congestion": 101,
        "Accident / Road Closure": 202,
        "Emergency Vehicle": 303,
    }

    seed = seed_map.get(scenario, 42)
    rng = np.random.default_rng(seed)

    rows = []
    base_capacity = 400  # Total standard capacity across all directions
    emergency_route_nodes = {"J1", "J2", "J4"}

    for intersection in intersections:
        road_capacity = base_capacity
        emergency_active = False
        is_emergency_route = False

        if scenario == "Normal Traffic":
            north = rng.integers(20, 40)
            south = rng.integers(20, 40)
            east = rng.integers(20, 40)
            west = rng.integers(20, 40)
            queue_length = rng.integers(3, 10)
            waiting_time = round(float(rng.uniform(10.0, 25.0)), 1)
            current_signal = rng.choice(["NS_GREEN", "EW_GREEN"])

        elif scenario == "Heavy Congestion":
            north = rng.integers(70, 95)
            south = rng.integers(70, 95)
            east = rng.integers(70, 95)
            west = rng.integers(70, 95)
            queue_length = rng.integers(25, 50)
            waiting_time = round(float(rng.uniform(55.0, 110.0)), 1)
            current_signal = rng.choice(["NS_GREEN", "EW_GREEN"])

        elif scenario == "Accident / Road Closure":
            if intersection == "J3":
                # Reduced capacity at affected intersection (80% capacity drop)
                road_capacity = 80
                north = rng.integers(85, 100)
                south = rng.integers(85, 100)
                east = rng.integers(10, 20)  # Blocked direction
                west = rng.integers(85, 100)
                queue_length = rng.integers(45, 65)
                waiting_time = round(float(rng.uniform(90.0, 150.0)), 1)
                current_signal = "NS_GREEN"
            elif intersection in ["J1", "J5"]:
                # Spillover congestion on connected intersections
                north = rng.integers(60, 85)
                south = rng.integers(60, 85)
                east = rng.integers(60, 85)
                west = rng.integers(60, 85)
                queue_length = rng.integers(20, 38)
                waiting_time = round(float(rng.uniform(45.0, 80.0)), 1)
                current_signal = rng.choice(["NS_GREEN", "EW_GREEN"])
            else:
                north = rng.integers(30, 50)
                south = rng.integers(30, 50)
                east = rng.integers(30, 50)
                west = rng.integers(30, 50)
                queue_length = rng.integers(8, 18)
                waiting_time = round(float(rng.uniform(20.0, 40.0)), 1)
                current_signal = rng.choice(["NS_GREEN", "EW_GREEN"])

        elif scenario == "Emergency Vehicle":
            north = rng.integers(25, 45)
            south = rng.integers(25, 45)
            east = rng.integers(25, 45)
            west = rng.integers(25, 45)
            queue_length = rng.integers(5, 15)
            waiting_time = round(float(rng.uniform(12.0, 30.0)), 1)
            current_signal = rng.choice(["NS_GREEN", "EW_GREEN"])

            emergency_active = True
            if intersection in emergency_route_nodes:
                is_emergency_route = True
        else:
            north = rng.integers(20, 40)
            south = rng.integers(20, 40)
            east = rng.integers(20, 40)
            west = rng.integers(20, 40)
            queue_length = rng.integers(3, 10)
            waiting_time = round(float(rng.uniform(10.0, 25.0)), 1)
            current_signal = "NS_GREEN"

        total_traffic = north + south + east + west
        vehicle_density = round(min(1.0, float(total_traffic / road_capacity)), 3)

        rows.append({
            "intersection_id": intersection,
            "north_traffic": int(north),
            "south_traffic": int(south),
            "east_traffic": int(east),
            "west_traffic": int(west),
            "vehicle_density": float(vehicle_density),
            "queue_length": int(queue_length),
            "road_capacity": int(road_capacity),
            "current_signal": str(current_signal),
            "waiting_time": float(waiting_time),
            "emergency_active": bool(emergency_active),
            "is_emergency_route": bool(is_emergency_route),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Test all 4 scenarios
    scenarios = [
        "Normal Traffic",
        "Heavy Congestion",
        "Accident / Road Closure",
        "Emergency Vehicle",
    ]
    for s in scenarios:
        df = generate_traffic(s)
        print(f"=== Scenario: {s} ===")
        print(df[["intersection_id", "vehicle_density", "queue_length", "road_capacity", "waiting_time", "emergency_active", "is_emergency_route"]])
        print()
