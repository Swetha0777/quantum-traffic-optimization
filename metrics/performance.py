"""
Module: Performance & Metrics Evaluation
Assigned to: Team Member 4 (Analytics & Metrics Lead)
Description: Calculates traffic performance metrics (delay, queue length, throughput, fuel consumption, CO2 emissions)
             for classical baseline vs optimized traffic signal configurations.
"""

import pandas as pd


def calculate_metrics(traffic_data: pd.DataFrame, signal_data: pd.DataFrame) -> dict:
    """
    Calculate estimated simulation metrics for optimized/emergency traffic signal timing.

    Formulas & Estimates:
        1. Average Waiting Time (s): Mean of estimated_waiting_time across intersections.
        2. Total Queue Length (vehicles): Sum of queue lengths reduced by additional green light duration.
        3. Throughput (veh/min): Estimated cleared vehicles per minute based on active green times and road capacity.
        4. Fuel Consumption (L): Idling fuel consumption = Total Queue * Wait Time * 0.00033 L/sec.
        5. CO2 Emissions (kg): Carbon emissions = Fuel Consumption * 2.31 kg CO2/L.

    Parameters:
        traffic_data (pd.DataFrame): Input traffic simulation DataFrame.
        signal_data (pd.DataFrame): Optimized or emergency signal DataFrame.

    Returns:
        dict: Performance KPI metrics dictionary.
    """
    merged = pd.merge(traffic_data, signal_data, on="intersection_id", how="inner")

    est_wait_times = []
    eff_queues = []
    throughputs = []

    for _, row in merged.iterrows():
        capacity = float(row.get("road_capacity", 400.0))
        density = float(row.get("vehicle_density", 0.3))
        queue = float(row.get("queue_length", 10.0))
        wait = float(row.get("waiting_time", 25.0))

        opt_green = float(row.get("optimized_green_time", 45.0))
        orig_green = float(row.get("original_green_time", 30.0))
        signal_status = str(row.get("signal_status", "NORMAL"))
        decision = str(row.get("decision", "KEEP"))

        # Calculate estimated waiting time
        if "estimated_waiting_time" in row:
            est_wait = float(row["estimated_waiting_time"])
        elif decision == "INCREASE_GREEN" or signal_status == "GREEN":
            est_wait = max(5.0, wait * 0.55)
        else:
            est_wait = wait

        # Calculate effective queue reduction from extra green duration
        extra_green = max(0.0, opt_green - orig_green)
        queue_cleared_extra = extra_green * 0.4
        eff_queue = max(0.0, queue - queue_cleared_extra)

        # Throughput (vehicles cleared per minute)
        # Base flow rate scaled by green ratio (opt_green / 60)
        throughput_val = (capacity * density * (opt_green / 60.0)) / 4.0  # Normalized per minute per junction

        est_wait_times.append(est_wait)
        eff_queues.append(eff_queue)
        throughputs.append(throughput_val)

    avg_wait = float(sum(est_wait_times) / len(est_wait_times)) if est_wait_times else 0.0
    total_queue = float(sum(eff_queues))
    total_throughput = float(sum(throughputs))

    # Fuel consumption formula: idling burn rate ~ 0.00033 L/sec per waiting vehicle
    # Total idling vehicle-seconds = sum(eff_queue * est_wait)
    idle_vehicle_seconds = sum(q * w for q, w in zip(eff_queues, est_wait_times))
    fuel_consumption = idle_vehicle_seconds * 0.00033  # Liters

    # CO2 emissions: 2.31 kg CO2 per Liter of gasoline burned
    co2_emissions = fuel_consumption * 2.31  # kg CO2

    return {
        "average_waiting_time": round(avg_wait, 2),
        "total_queue_length": round(total_queue, 1),
        "throughput": round(total_throughput, 1),
        "fuel_consumption": round(fuel_consumption, 2),
        "co2_emissions": round(co2_emissions, 2),
    }


def calculate_classical_baseline(traffic_data: pd.DataFrame) -> dict:
    """
    Calculate performance metrics for a classical fixed-time signal system.
    Baseline green duration is derived from current signal phase timing (45s for GREEN, 30s for RED).

    Parameters:
        traffic_data (pd.DataFrame): Input traffic simulation DataFrame.

    Returns:
        dict: Baseline performance KPI metrics dictionary.
    """
    wait_times = []
    queues = []
    throughputs = []

    for _, row in traffic_data.iterrows():
        capacity = float(row.get("road_capacity", 400.0))
        density = float(row.get("vehicle_density", 0.3))
        queue = float(row.get("queue_length", 10.0))
        wait = float(row.get("waiting_time", 25.0))
        current_signal = str(row.get("current_signal", "NS_GREEN"))

        # Baseline green time matching standard signal phase
        orig_green = 45.0 if "GREEN" in current_signal.upper() else 30.0

        # Fixed time throughput
        tp = (capacity * density * (orig_green / 60.0)) / 4.0

        wait_times.append(wait)
        queues.append(queue)
        throughputs.append(tp)

    avg_wait = float(sum(wait_times) / len(wait_times)) if wait_times else 0.0
    total_queue = float(sum(queues))
    total_throughput = float(sum(throughputs))

    idle_vehicle_seconds = sum(q * w for q, w in zip(queues, wait_times))
    fuel_consumption = idle_vehicle_seconds * 0.00033
    co2_emissions = fuel_consumption * 2.31

    return {
        "average_waiting_time": round(avg_wait, 2),
        "total_queue_length": round(total_queue, 1),
        "throughput": round(total_throughput, 1),
        "fuel_consumption": round(fuel_consumption, 2),
        "co2_emissions": round(co2_emissions, 2),
    }


def compare_performance(classical_metrics: dict, optimized_metrics: dict) -> dict:
    """
    Compare classical baseline metrics against optimized signal metrics.

    Parameters:
        classical_metrics (dict): KPI dict from calculate_classical_baseline().
        optimized_metrics (dict): KPI dict from calculate_metrics().

    Returns:
        dict: Performance comparison containing:
            - 'classical': baseline values
            - 'optimized': optimized values
            - 'percentage_change': percentage change for each metric
    """
    pct_change = {}

    # Helper function for percentage change calculation
    def calc_pct(c_val, o_val, lower_is_better=True):
        if c_val == 0:
            return 0.0
        diff = ((c_val - o_val) / c_val) * 100.0 if lower_is_better else ((o_val - c_val) / c_val) * 100.0
        return round(diff, 2)

    pct_change["waiting_time_reduction_pct"] = calc_pct(
        classical_metrics["average_waiting_time"], optimized_metrics["average_waiting_time"], lower_is_better=True
    )
    pct_change["queue_reduction_pct"] = calc_pct(
        classical_metrics["total_queue_length"], optimized_metrics["total_queue_length"], lower_is_better=True
    )
    pct_change["throughput_improvement_pct"] = calc_pct(
        classical_metrics["throughput"], optimized_metrics["throughput"], lower_is_better=False
    )
    pct_change["fuel_savings_pct"] = calc_pct(
        classical_metrics["fuel_consumption"], optimized_metrics["fuel_consumption"], lower_is_better=True
    )
    pct_change["co2_savings_pct"] = calc_pct(
        classical_metrics["co2_emissions"], optimized_metrics["co2_emissions"], lower_is_better=True
    )

    return {
        "classical": classical_metrics,
        "optimized": optimized_metrics,
        "percentage_change": pct_change,
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from simulation.traffic_simulation import generate_traffic
    from optimization.qaoa import optimize_signals

    t_data = generate_traffic("Heavy Congestion")
    opt_signals = optimize_signals(t_data)

    class_m = calculate_classical_baseline(t_data)
    opt_m = calculate_metrics(t_data, opt_signals)
    comp = compare_performance(class_m, opt_m)

    print("=== Classical Baseline Metrics ===")
    print(class_m)
    print("\n=== Optimized Signal Metrics ===")
    print(opt_m)
    print("\n=== Performance Comparison ===")
    print(comp["percentage_change"])
