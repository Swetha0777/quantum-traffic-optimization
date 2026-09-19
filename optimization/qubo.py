"""
Module: Quantum Optimization - QUBO Formulation
Assigned to: Team Member 2 (Quantum Optimization Lead)
Description: Converts traffic flow & signal timing problems into Quadratic Unconstrained Binary Optimization (QUBO) matrices.
"""

import numpy as np
import pandas as pd


def create_qubo(traffic_data: pd.DataFrame) -> dict:
    """
    Convert traffic data into a Quadratic Unconstrained Binary Optimization (QUBO) problem.

    Binary Decision Variables:
        x_i = 0 -> Keep current green signal duration for intersection i
        x_i = 1 -> Increase green signal duration for intersection i

    QUBO Objective Function:
        Minimize H(x) = x^T * Q * x = sum_i (Q_ii * x_i) + sum_{i < j} ((Q_ij + Q_ji) * x_i * x_j)

        1. Diagonal terms Q_ii (Linear terms):
           - Represent the net benefit/cost of giving extra green time to intersection i.
           - Computed based on traffic pressure P_i = w1*density + w2*queue_ratio + w3*wait_ratio (+ capacity penalty).
           - High traffic pressure P_i yields negative Q_ii, encouraging x_i = 1 to minimize H(x).

        2. Off-diagonal terms Q_ij (Quadratic penalty terms):
           - Represent network interaction penalties between adjacent connected intersections.
           - If both adjacent intersections i and j get extra green time simultaneously (x_i=1 and x_j=1),
             a positive penalty is added to prevent downstream bottleneck congestion.

    Parameters:
        traffic_data (pd.DataFrame): DataFrame returned by generate_traffic().

    Returns:
        dict: QUBO formulation object containing:
            - 'Q': 6x6 NumPy array matrix
            - 'qubo_dict': dictionary mapping variable pairs (x_i, x_j) to Q_ij weights
            - 'variables': list of binary variable names ['x_J1', 'x_J2', ...]
            - 'intersections': list of intersection IDs ['J1', ..., 'J6']
            - 'pressure_scores': dict mapping intersection_id to computed pressure P_i
    """
    intersections = list(traffic_data["intersection_id"])
    n = len(intersections)
    idx_map = {inter_id: idx for idx, inter_id in enumerate(intersections)}

    # Topo-graph adjacency of the 6 interconnected intersections:
    # J1 -- J2 -- J4
    #  |     |     |
    # J3 -- J5 -- J6
    adjacency = [
        ("J1", "J2"), ("J1", "J3"),
        ("J2", "J4"), ("J2", "J5"),
        ("J3", "J5"),
        ("J4", "J6"),
        ("J5", "J6")
    ]

    # Initialize N x N QUBO matrix
    Q = np.zeros((n, n), dtype=float)

    # Weights for traffic pressure formula
    w_density = 0.40
    w_queue = 0.30
    w_wait = 0.30

    pressure_scores = {}

    # 1. Diagonal Terms (Linear node incentives)
    for _, row in traffic_data.iterrows():
        inter_id = row["intersection_id"]
        i = idx_map[inter_id]

        density = float(row.get("vehicle_density", 0.0))
        queue = float(row.get("queue_length", 0))
        waiting_time = float(row.get("waiting_time", 0.0))
        capacity = float(row.get("road_capacity", 400.0))

        # Normalize components to [0, 1] range
        norm_density = min(1.0, max(0.0, density))
        norm_queue = min(1.0, max(0.0, queue / 50.0))
        norm_wait = min(1.0, max(0.0, waiting_time / 100.0))

        # Composite traffic pressure score P_i in [0.0, 1.0]
        pressure = w_density * norm_density + w_queue * norm_queue + w_wait * norm_wait

        # Severe capacity restriction penalty (e.g. road closure / accident at J3)
        if capacity < 200.0:
            pressure = min(1.0, pressure + 0.35)

        pressure_scores[inter_id] = round(float(pressure), 3)

        # In minimization min(x^T Q x), a negative diagonal entry Q_ii reduces total cost when x_i = 1.
        # Neutral baseline threshold is 0.35:
        # If P_i > 0.35 -> Q_ii < 0 -> setting x_i = 1 lowers energy H(x).
        # If P_i <= 0.35 -> Q_ii >= 0 -> setting x_i = 1 increases energy H(x).
        Q[i, i] = 0.35 - pressure

    # 2. Off-Diagonal Terms (Quadratic edge penalties for adjacent intersections)
    penalty = 0.25
    for u, v in adjacency:
        if u in idx_map and v in idx_map:
            i, j = idx_map[u], idx_map[v]
            # Symmetrically distribute pair penalty across Q[i, j] and Q[j, i]
            Q[i, j] += penalty / 2.0
            Q[j, i] += penalty / 2.0

    # Build dictionary representation
    qubo_dict = {}
    var_names = [f"x_{inter}" for inter in intersections]
    for i in range(n):
        for j in range(n):
            if Q[i, j] != 0:
                qubo_dict[(var_names[i], var_names[j])] = round(float(Q[i, j]), 4)

    return {
        "Q": Q,
        "qubo_dict": qubo_dict,
        "variables": var_names,
        "intersections": intersections,
        "pressure_scores": pressure_scores,
    }
