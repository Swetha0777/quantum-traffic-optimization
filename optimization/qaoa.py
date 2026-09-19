"""
Module: Quantum Optimization - QAOA & Hybrid Solver
Assigned to: Team Member 2 (Quantum Optimization Lead)
Description: Solves the QUBO problem formulated in qubo.py using QAOA (or exact classical QUBO solver fallback)
             to optimize traffic signal timings across 6 intersections.
"""

import numpy as np
import pandas as pd
try:
    from optimization.qubo import create_qubo
except ModuleNotFoundError:
    from qubo import create_qubo


def _solve_qubo_classical(Q: np.ndarray) -> np.ndarray:
    """
    Solve small N=6 QUBO problem by evaluating all 2^6 = 64 binary state vectors.
    Finds the exact binary vector x* in {0, 1}^N that minimizes H(x) = x^T * Q * x.
    """
    n = Q.shape[0]
    best_cost = float("inf")
    best_x = np.zeros(n, dtype=int)

    # Exhaustive evaluation of all 2^N state vectors
    for num in range(1 << n):
        x = np.array([(num >> k) & 1 for k in range(n)], dtype=int)
        cost = float(x.T @ Q @ x)
        if cost < best_cost:
            best_cost = cost
            best_x = x

    return best_x


def _solve_qubo_qiskit(Q: np.ndarray) -> np.ndarray:
    """
    Attempt to solve QUBO using Qiskit / QAOA if installed in environment.
    Falls back to classical exact matrix solver if Qiskit is not available.
    """
    try:
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        from qiskit_algorithms import QAOA
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit_primitives import Sampler

        n = Q.shape[0]
        qp = QuadraticProgram()
        for i in range(n):
            qp.binary_var(name=f"x_{i}")

        linear = {}
        quadratic = {}
        for i in range(n):
            linear[f"x_{i}"] = Q[i, i]
            for j in range(i + 1, n):
                weight = Q[i, j] + Q[j, i]
                if weight != 0:
                    quadratic[(f"x_{i}", f"x_{j}")] = weight

        qp.minimize(linear=linear, quadratic=quadratic)

        qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(maxiter=100))
        optimizer = MinimumEigenOptimizer(qaoa)
        result = optimizer.solve(qp)
        return np.array([int(result.x[i]) for i in range(n)], dtype=int)

    except Exception:
        # Documented hybrid fallback solver
        return _solve_qubo_classical(Q)


def optimize_signals(traffic_data: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize signal timings for intersections using QUBO formulation and QAOA/Hybrid optimization.

    Parameters:
        traffic_data (pd.DataFrame): Input traffic DataFrame from simulation.traffic_simulation.generate_traffic().

    Returns:
        pd.DataFrame: Optimized signal timings containing:
            - intersection_id
            - current_signal
            - original_green_time
            - optimized_green_time
            - decision ("INCREASE_GREEN" or "KEEP")
            - estimated_waiting_time
    """
    # 1. Generate QUBO representation
    qubo_info = create_qubo(traffic_data)
    Q = qubo_info["Q"]
    intersections = qubo_info["intersections"]

    # 2. Solve QUBO to obtain binary decision vector x*
    best_x = _solve_qubo_qiskit(Q)

    # 3. Construct signal timing recommendations
    output_rows = []

    for idx, row in traffic_data.iterrows():
        inter_id = row["intersection_id"]
        i = intersections.index(inter_id)
        current_signal = str(row.get("current_signal", "NS_GREEN"))
        waiting_time = float(row.get("waiting_time", 20.0))
        density = float(row.get("vehicle_density", 0.3))
        queue = int(row.get("queue_length", 5))

        # Base green time duration
        original_green_time = 45 if "GREEN" in current_signal.upper() else 30

        decision_val = best_x[i]

        if decision_val == 1:
            decision = "INCREASE_GREEN"
            # Allocate additional green duration proportional to demand (+15s to +30s)
            extra_time = int(min(30, max(15, 15 + queue * 0.4 + density * 10)))
            optimized_green_time = original_green_time + extra_time
            # Estimated waiting time decreases due to green extension
            estimated_waiting_time = round(max(5.0, waiting_time * 0.55), 1)
        else:
            decision = "KEEP"
            optimized_green_time = original_green_time
            estimated_waiting_time = round(waiting_time, 1)

        output_rows.append({
            "intersection_id": inter_id,
            "current_signal": current_signal,
            "original_green_time": int(original_green_time),
            "optimized_green_time": int(optimized_green_time),
            "decision": decision,
            "estimated_waiting_time": float(estimated_waiting_time),
        })

    return pd.DataFrame(output_rows)


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from simulation.traffic_simulation import generate_traffic

    scenarios = [
        "Normal Traffic",
        "Heavy Congestion",
        "Accident / Road Closure",
    ]

    for scenario in scenarios:
        print(f"\n================ Scenario: '{scenario}' ================")
        t_data = generate_traffic(scenario)
        opt_res = optimize_signals(t_data)
        print(opt_res.to_string(index=False))
