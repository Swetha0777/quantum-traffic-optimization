"""
Module: Emergency Corridor Optimization
Assigned to: Team Member 3 (Emergency Response Lead)
Description: Dynamically creates green light corridors for emergency vehicles navigating through traffic networks.
"""

import pandas as pd


def create_emergency_corridor(route: list, signal_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a temporary green corridor for an emergency vehicle along a specified route.

    Parameters:
        route (list): List of intersection IDs along the emergency corridor (e.g., ["J1", "J2", "J3", "J5"]).
        signal_data (pd.DataFrame): Optimized signal DataFrame produced by optimize_signals().

    Returns:
        pd.DataFrame: Modified DataFrame with green corridor priorities applied:
            - intersection_id
            - signal_status ("GREEN" for route nodes, "NORMAL" for non-route)
            - optimized_green_time (60s priority for route nodes, preserved for non-route)
            - emergency_priority (True for route nodes, False for non-route)
    """
    df = signal_data.copy()
    route_set = set(route)

    signal_status_list = []
    opt_green_list = []
    emergency_priority_list = []

    for _, row in df.iterrows():
        inter_id = row["intersection_id"]
        orig_green = int(row.get("optimized_green_time", 45))

        if inter_id in route_set:
            signal_status_list.append("GREEN")
            # Grant priority green duration (at least 60 seconds)
            opt_green_list.append(max(60, orig_green))
            emergency_priority_list.append(True)
        else:
            signal_status_list.append("NORMAL")
            opt_green_list.append(orig_green)
            emergency_priority_list.append(False)

    df["signal_status"] = signal_status_list
    df["optimized_green_time"] = opt_green_list
    df["emergency_priority"] = emergency_priority_list

    # Ensure required schema columns appear first
    req_cols = ["intersection_id", "signal_status", "optimized_green_time", "emergency_priority"]
    other_cols = [c for c in df.columns if c not in req_cols]
    return df[req_cols + other_cols]


def restore_normal_signals(signal_data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove emergency priority flags and return normal optimized signal configuration.

    Parameters:
        signal_data (pd.DataFrame): Emergency or modified signal DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame with normal signal statuses and emergency_priority=False.
    """
    df = signal_data.copy()
    df["signal_status"] = "NORMAL"
    df["emergency_priority"] = False

    req_cols = ["intersection_id", "signal_status", "optimized_green_time", "emergency_priority"]
    other_cols = [c for c in df.columns if c not in req_cols]
    return df[req_cols + other_cols]


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from simulation.traffic_simulation import generate_traffic
    from optimization.qaoa import optimize_signals

    t_data = generate_traffic("Emergency Vehicle")
    opt_signals = optimize_signals(t_data)
    route = ["J1", "J2", "J4"]

    print("=== Emergency Green Corridor ===")
    emerg_signals = create_emergency_corridor(route, opt_signals)
    print(emerg_signals.to_string(index=False))

    print("\n=== Restored Normal Signals ===")
    restored = restore_normal_signals(emerg_signals)
    print(restored.to_string(index=False))
