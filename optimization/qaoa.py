"""
Module: Quantum Optimization - QAOA & Hybrid Solver
Assigned to: Team Member 2 (Quantum Optimization Lead)
Description: Solves the QUBO problem formulated in qubo.py using QAOA (or exact classical QUBO solver fallback)
             to optimize traffic signal timings across 6 intersections.

Solver Metadata:
    solver_used  : str  - "Qiskit QAOA" or "Classical Exact Fallback"
    quantum_used : bool - True if Qiskit QAOA was used, False for classical
    objective_value : float - Final objective value H(x*) = x*^T Q x*
"""

import numpy as np
import pandas as pd
try:
    from optimization.qubo import create_qubo, calculate_qubo_cost
except ModuleNotFoundError:
    from qubo import create_qubo, calculate_qubo_cost


# Module-level solver metadata (updated after each solve)
_solver_metadata = {
    "solver_used": "Not yet solved",
    "quantum_used": False,
    "objective_value": 0.0,
}


def _solve_qubo_classical(Q: np.ndarray) -> np.ndarray:
    """
    Solve small N=6 QUBO problem by evaluating all 2^6 = 64 binary state vectors.
    Finds the exact binary vector x* in {0, 1}^N that minimizes H(x) = x^T * Q * x.
    """
    global _solver_metadata
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

    _solver_metadata["solver_used"] = "Classical Exact Fallback"
    _solver_metadata["quantum_used"] = False
    _solver_metadata["objective_value"] = round(best_cost, 6)

    return best_x


def _solve_qubo_qiskit(Q: np.ndarray) -> np.ndarray:
    """
    Attempt to solve QUBO using Qiskit / QAOA if installed in environment.
    Supports Qiskit 1.x and 2.x APIs with graceful fallback.
    Falls back to classical exact matrix solver if Qiskit is not available.
    """
    global _solver_metadata
    try:
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer

        n = Q.shape[0]
        qp = QuadraticProgram()
        for i in range(n):
            qp.binary_var(name=f"x_{i}")

        linear = {}
        quadratic = {}
        for i in range(n):
            linear[f"x_{i}"] = float(Q[i, i])
            for j in range(i + 1, n):
                weight = float(Q[i, j] + Q[j, i])
                if abs(weight) > 1e-10:
                    quadratic[(f"x_{i}", f"x_{j}")] = weight

        qp.minimize(linear=linear, quadratic=quadratic)

        # Try Qiskit 2.x primitives first, then 1.x
        qaoa = None
        try:
            from qiskit.primitives import StatevectorSampler
            from qiskit_algorithms import QAOA
            from qiskit_algorithms.optimizers import COBYLA
            qaoa = QAOA(sampler=StatevectorSampler(), optimizer=COBYLA(maxiter=100))
        except (ImportError, Exception):
            try:
                from qiskit_algorithms import QAOA
                from qiskit_algorithms.optimizers import COBYLA
                from qiskit.primitives import Sampler
                qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(maxiter=100))
            except (ImportError, Exception):
                pass

        if qaoa is not None:
            optimizer = MinimumEigenOptimizer(qaoa)
            result = optimizer.solve(qp)
            solution = np.array([int(result.x[i]) for i in range(n)], dtype=int)
            obj_value = float(solution.T @ Q @ solution)

            _solver_metadata["solver_used"] = "Qiskit QAOA"
            _solver_metadata["quantum_used"] = True
            _solver_metadata["objective_value"] = round(obj_value, 6)

            return solution

        # If QAOA failed to initialize, use classical
        return _solve_qubo_classical(Q)

    except Exception:
        # Documented hybrid fallback solver
        return _solve_qubo_classical(Q)


def get_solver_metadata() -> dict:
    """
    Get metadata about the last QUBO solve.

    Returns
    -------
    dict
        {
            'solver_used': str,
            'quantum_used': bool,
            'objective_value': float
        }
    """
    return _solver_metadata.copy()


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
            - solver_used
            - quantum_used
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

    result_df = pd.DataFrame(output_rows)

    # Attach solver metadata as DataFrame attributes
    result_df.attrs["solver_used"] = _solver_metadata["solver_used"]
    result_df.attrs["quantum_used"] = _solver_metadata["quantum_used"]
    result_df.attrs["objective_value"] = _solver_metadata["objective_value"]

    return result_df


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
