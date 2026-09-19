"""
Comprehensive Test Suite for Quantum Traffic Optimization System

Tests all modules:
    1. Traffic simulation (4 scenarios)
    2. Traffic density, queue, congestion analysis
    3. QUBO matrix formulation
    4. QAOA / Classical solver with metadata
    5. Emergency corridor management
    6. Accident detection
    7. Emergency vehicle detection
    8. Route optimization (shortest, traffic-aware, emergency bypass)
    9. Performance metrics (before/after comparison)
    10. Visualization (map and charts creation)
"""

import sys
import os
import unittest

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))

import numpy as np
import pandas as pd


# ============================================================
# TEST 1: TRAFFIC SIMULATION
# ============================================================

class TestTrafficSimulation(unittest.TestCase):
    """Test traffic generation for all 4 scenarios."""

    def setUp(self):
        from simulation.traffic_simulation import (
            generate_traffic, INTERSECTIONS
        )
        self.generate = generate_traffic
        self.intersections = INTERSECTIONS

    def test_normal_traffic(self):
        data = self.generate("Normal Traffic", seed=42)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 6)
        self.assertIn("intersection_id", data.columns)
        self.assertIn("vehicle_density", data.columns)
        self.assertIn("queue_length", data.columns)
        self.assertIn("waiting_time", data.columns)

    def test_heavy_congestion(self):
        data = self.generate("Heavy Congestion", seed=42)
        self.assertEqual(len(data), 6)
        # Congestion should produce higher densities
        avg_density = data["vehicle_density"].mean()
        self.assertGreater(avg_density, 0.5)

    def test_accident_scenario(self):
        data = self.generate("Accident / Road Closure", seed=42)
        self.assertEqual(len(data), 6)
        # J3 should be blocked
        j3 = data[data["intersection_id"] == "J3"]
        self.assertTrue(j3.iloc[0]["blocked"])

    def test_emergency_scenario(self):
        data = self.generate("Emergency Vehicle", seed=42)
        self.assertEqual(len(data), 6)
        # Emergency should be active
        self.assertTrue(
            data["emergency_active"].any()
        )

    def test_invalid_scenario_raises(self):
        with self.assertRaises(ValueError):
            self.generate("Invalid Scenario")

    def test_all_required_columns(self):
        data = self.generate("Normal Traffic", seed=42)
        required_cols = [
            "intersection_id", "current_signal",
            "vehicle_density", "queue_length",
            "waiting_time", "throughput",
            "speed_kmh", "congestion_level"
        ]
        for col in required_cols:
            self.assertIn(
                col, data.columns,
                f"Missing column: {col}"
            )


# ============================================================
# TEST 2: TRAFFIC ANALYSIS
# ============================================================

class TestTrafficAnalysis(unittest.TestCase):
    """Test density, queue, and congestion modules."""

    def test_density_classification(self):
        from traffic.density import classify_density

        self.assertEqual(classify_density(0.1), "LOW")
        self.assertEqual(classify_density(0.5), "MEDIUM")
        self.assertEqual(classify_density(0.75), "HIGH")
        self.assertEqual(classify_density(0.95), "CRITICAL")

    def test_density_computation(self):
        from traffic.density import compute_density

        d = compute_density(200, 400)
        self.assertAlmostEqual(d, 0.5)

        d = compute_density(400, 400)
        self.assertAlmostEqual(d, 1.0)

    def test_queue_detection(self):
        from traffic.queue import detect_queue_from_dataframe
        from simulation.traffic_simulation import generate_traffic

        data = generate_traffic("Heavy Congestion", seed=42)
        result = detect_queue_from_dataframe(data)

        self.assertIn("is_queued", result.columns)
        self.assertIn("queue_severity", result.columns)

    def test_congestion_score(self):
        from traffic.congestion import compute_congestion_score

        score = compute_congestion_score(
            density=0.9,
            queue_length=50,
            waiting_time=120,
            road_capacity=400
        )
        self.assertGreater(score, 50)

    def test_congestion_classification(self):
        from traffic.congestion import classify_congestion

        self.assertEqual(classify_congestion(10), "LOW")
        self.assertEqual(classify_congestion(30), "MODERATE")
        self.assertEqual(classify_congestion(60), "HIGH")
        self.assertEqual(classify_congestion(90), "CRITICAL")


# ============================================================
# TEST 3: QUBO FORMULATION
# ============================================================

class TestQUBO(unittest.TestCase):
    """Test QUBO matrix creation."""

    def setUp(self):
        from simulation.traffic_simulation import generate_traffic
        self.traffic_data = generate_traffic(
            "Heavy Congestion", seed=42
        )

    def test_qubo_matrix_shape(self):
        from optimization.qubo import create_qubo

        result = create_qubo(self.traffic_data)
        Q = result["Q"]
        self.assertEqual(Q.shape, (6, 6))

    def test_qubo_returns_all_keys(self):
        from optimization.qubo import create_qubo

        result = create_qubo(self.traffic_data)
        self.assertIn("Q", result)
        self.assertIn("intersections", result)
        self.assertIn("demand_scores", result)
        self.assertIn("metadata", result)

    def test_qubo_cost_calculation(self):
        from optimization.qubo import (
            create_qubo, calculate_qubo_cost
        )

        result = create_qubo(self.traffic_data)
        Q = result["Q"]

        # Test all zeros
        x = np.zeros(6, dtype=int)
        cost = calculate_qubo_cost(Q, x)
        self.assertEqual(cost, 0.0)

        # Test with some solution
        x = np.array([1, 0, 1, 0, 1, 0], dtype=int)
        cost = calculate_qubo_cost(Q, x)
        self.assertIsInstance(cost, float)

    def test_qubo_empty_raises(self):
        from optimization.qubo import create_qubo

        with self.assertRaises(ValueError):
            create_qubo(pd.DataFrame())


# ============================================================
# TEST 4: QAOA SOLVER
# ============================================================

class TestQAOA(unittest.TestCase):
    """Test QAOA/Hybrid solver."""

    def setUp(self):
        from simulation.traffic_simulation import generate_traffic
        self.traffic_data = generate_traffic(
            "Heavy Congestion", seed=42
        )

    def test_optimize_signals(self):
        from optimization.qaoa import optimize_signals

        result = optimize_signals(self.traffic_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 6)

    def test_optimization_columns(self):
        from optimization.qaoa import optimize_signals

        result = optimize_signals(self.traffic_data)
        required = [
            "intersection_id", "current_signal",
            "original_green_time", "optimized_green_time",
            "decision", "estimated_waiting_time"
        ]
        for col in required:
            self.assertIn(col, result.columns)

    def test_solver_metadata(self):
        from optimization.qaoa import (
            optimize_signals, get_solver_metadata
        )

        optimize_signals(self.traffic_data)
        meta = get_solver_metadata()

        self.assertIn("solver_used", meta)
        self.assertIn("quantum_used", meta)
        self.assertIn("objective_value", meta)
        self.assertIsInstance(meta["quantum_used"], bool)
        self.assertIsInstance(meta["objective_value"], float)

    def test_decisions_valid(self):
        from optimization.qaoa import optimize_signals

        result = optimize_signals(self.traffic_data)
        valid_decisions = {
            "INCREASE_GREEN", "KEEP"
        }
        for decision in result["decision"]:
            self.assertIn(decision, valid_decisions)


# ============================================================
# TEST 5: EMERGENCY CORRIDOR
# ============================================================

class TestEmergencyCorridor(unittest.TestCase):
    """Test emergency corridor management."""

    def test_vehicle_creation(self):
        from emergency.emergency_corridor import EmergencyVehicle

        ev = EmergencyVehicle(
            vehicle_id="EV-01",
            route=["J1", "J2", "J4"]
        )
        self.assertEqual(ev.current_intersection, "J1")
        self.assertEqual(ev.next_intersection, "J2")
        self.assertTrue(ev.active)

    def test_vehicle_advance(self):
        from emergency.emergency_corridor import EmergencyVehicle

        ev = EmergencyVehicle(route=["J1", "J2", "J4"])
        ev.advance()
        self.assertEqual(ev.current_intersection, "J2")
        ev.advance()
        self.assertEqual(ev.current_intersection, "J4")
        ev.advance()
        self.assertFalse(ev.active)

    def test_corridor_priority(self):
        from emergency.emergency_corridor import (
            EmergencyVehicle, EmergencyCorridor
        )

        ev = EmergencyVehicle(route=["J1", "J2", "J4"])
        corridor = EmergencyCorridor(route=["J1", "J2", "J4"])

        self.assertEqual(
            corridor.get_priority("J1", ev),
            "ACTIVE"
        )
        self.assertEqual(
            corridor.get_priority("J2", ev),
            "PREPARE"
        )
        self.assertEqual(
            corridor.get_priority("J4", ev),
            "CORRIDOR"
        )
        self.assertEqual(
            corridor.get_priority("J5", ev),
            "NORMAL"
        )

    def test_signal_override(self):
        from emergency.emergency_corridor import (
            EmergencyVehicle, EmergencyCorridor
        )
        from simulation.traffic_simulation import generate_traffic

        ev = EmergencyVehicle(route=["J1", "J2", "J4"])
        corridor = EmergencyCorridor(route=["J1", "J2", "J4"])
        data = generate_traffic("Emergency Vehicle", seed=42)

        overrides = corridor.create_signal_override(data, ev)
        self.assertIn("J1", overrides)
        self.assertEqual(overrides["J1"]["priority"], "ACTIVE")
        self.assertEqual(overrides["J1"]["green_time"], 60)


# ============================================================
# TEST 6: ACCIDENT DETECTION
# ============================================================

class TestAccidentDetection(unittest.TestCase):
    """Test accident detection."""

    def test_detect_from_accident_scenario(self):
        from emergency.accident_detector import AccidentDetector
        from simulation.traffic_simulation import generate_traffic

        data = generate_traffic(
            "Accident / Road Closure", seed=42
        )
        detector = AccidentDetector()
        alerts = detector.detect_from_dataframe(data)

        # J3 should be flagged
        j3_alerts = [
            a for a in alerts
            if a["intersection_id"] == "J3"
        ]
        self.assertTrue(len(j3_alerts) > 0)
        self.assertTrue(j3_alerts[0]["blocked"])

    def test_no_accident_in_normal(self):
        from emergency.accident_detector import AccidentDetector
        from simulation.traffic_simulation import generate_traffic

        data = generate_traffic(
            "Normal Traffic", seed=42
        )
        detector = AccidentDetector()
        alerts = detector.detect_from_dataframe(data)

        # Should have no critical alerts for normal traffic
        critical = [
            a for a in alerts
            if a["severity"] == "CRITICAL"
        ]
        self.assertEqual(len(critical), 0)


# ============================================================
# TEST 7: EMERGENCY VEHICLE DETECTION
# ============================================================

class TestEmergencyDetection(unittest.TestCase):
    """Test emergency vehicle detection."""

    def test_detect_emergency_flags(self):
        from emergency.emergency_detector import (
            EmergencyVehicleDetector
        )
        from simulation.traffic_simulation import generate_traffic

        data = generate_traffic(
            "Emergency Vehicle", seed=42
        )
        detector = EmergencyVehicleDetector()
        detections = detector.detect_from_dataframe(data)

        self.assertTrue(len(detections) > 0)
        self.assertTrue(detector.has_active_emergency())


# ============================================================
# TEST 8: ROUTE OPTIMIZATION
# ============================================================

class TestRouteOptimizer(unittest.TestCase):
    """Test NetworkX route optimizer."""

    def setUp(self):
        from routing.route_optimizer import RouteOptimizer
        self.optimizer = RouteOptimizer()

    def test_shortest_path(self):
        result = self.optimizer.shortest_path("J1", "J6")
        self.assertIsNotNone(result)
        self.assertEqual(result["path"][0], "J1")
        self.assertEqual(result["path"][-1], "J6")

    def test_traffic_aware_path(self):
        from simulation.traffic_simulation import generate_traffic

        data = generate_traffic("Normal Traffic", seed=42)
        self.optimizer.update_weights_from_traffic(data)

        result = self.optimizer.traffic_aware_path("J1", "J6")
        self.assertIsNotNone(result)
        self.assertIn("total_time", result)
        self.assertIn("total_distance", result)

    def test_emergency_bypass(self):
        result = self.optimizer.emergency_route(
            "J1", "J6",
            blocked_nodes=["J3"]
        )
        self.assertIsNotNone(result)
        self.assertNotIn("J3", result["path"])

    def test_all_routes(self):
        result = self.optimizer.get_all_routes(
            "J1", "J6",
            blocked_nodes=["J3"]
        )
        self.assertIn("shortest", result)
        self.assertIn("traffic_aware", result)
        self.assertIn("emergency", result)


# ============================================================
# TEST 9: PERFORMANCE METRICS
# ============================================================

class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics calculation."""

    def setUp(self):
        from simulation.traffic_simulation import generate_traffic
        from optimization.qaoa import optimize_signals
        from simulation.traffic_simulation import apply_optimization

        self.before_data = generate_traffic(
            "Heavy Congestion", seed=42
        )
        self.opt_result = optimize_signals(self.before_data)
        self.after_data = apply_optimization(
            self.before_data, self.opt_result
        )

    def test_calculate_metrics(self):
        from metrics.performance import calculate_metrics

        metrics = calculate_metrics(self.before_data)
        self.assertIn("total_traffic", metrics)
        self.assertIn("total_queue_length", metrics)
        self.assertIn("average_waiting_time", metrics)
        self.assertIn("estimated_throughput", metrics)
        self.assertIn("estimated_fuel_litres", metrics)
        self.assertIn("estimated_co2_kg", metrics)

    def test_compare_metrics(self):
        from metrics.performance import compare_metrics

        comparison = compare_metrics(
            self.before_data, self.after_data
        )

        self.assertIn("total_queue_length", comparison)
        self.assertIn("average_waiting_time", comparison)
        self.assertIn("estimated_throughput", comparison)

        # Queue should decrease after optimization
        queue_change = comparison["total_queue_length"]
        self.assertIn("before", queue_change)
        self.assertIn("after", queue_change)
        self.assertIn("improvement_percent", queue_change)


# ============================================================
# TEST 10: VISUALIZATION
# ============================================================

class TestVisualization(unittest.TestCase):
    """Test map and chart creation."""

    def setUp(self):
        from simulation.traffic_simulation import generate_traffic
        self.traffic_data = generate_traffic(
            "Normal Traffic", seed=42
        )

    def test_create_traffic_map(self):
        from visualization.map_view import create_traffic_map
        import folium

        m = create_traffic_map(self.traffic_data)
        self.assertIsInstance(m, folium.Map)

    def test_create_comparison_chart(self):
        from metrics.performance import compare_metrics
        from optimization.qaoa import optimize_signals
        from simulation.traffic_simulation import apply_optimization
        from visualization.charts import create_comparison_chart

        opt = optimize_signals(self.traffic_data)
        after = apply_optimization(self.traffic_data, opt)
        comparison = compare_metrics(self.traffic_data, after)

        fig = create_comparison_chart(comparison)
        self.assertIsNotNone(fig)

    def test_create_queue_chart(self):
        from visualization.charts import create_queue_chart

        fig = create_queue_chart(self.traffic_data)
        self.assertIsNotNone(fig)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("QUANTUM TRAFFIC OPTIMIZATION - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print()

    unittest.main(verbosity=2)
