"""
Module: Route Optimizer

NetworkX weighted graph route optimizer for the traffic network.

Supports:
    - Shortest path (unweighted)
    - Traffic-aware path (weighted by congestion)
    - Emergency route (bypassing blocked roads)
    - Alternative routes when primary is blocked

Network topology (6 intersections):
    J1 -- J2 -- J3
    |         |
    J4 -- J5 -- J6
"""

import networkx as nx
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ============================================================
# NETWORK TOPOLOGY
# ============================================================

# Edge definitions with base travel times (seconds)
NETWORK_EDGES = [
    ("J1", "J2", {"base_time": 30, "distance_km": 0.5}),
    ("J1", "J4", {"base_time": 35, "distance_km": 0.6}),
    ("J2", "J3", {"base_time": 25, "distance_km": 0.4}),
    ("J2", "J5", {"base_time": 30, "distance_km": 0.5}),
    ("J3", "J4", {"base_time": 35, "distance_km": 0.6}),
    ("J3", "J6", {"base_time": 30, "distance_km": 0.5}),
    ("J4", "J5", {"base_time": 25, "distance_km": 0.4}),
    ("J5", "J6", {"base_time": 30, "distance_km": 0.5}),
]

# Node positions for visualization (lat, lon approximation)
NODE_POSITIONS = {
    "J1": (12.9716, 77.5946),
    "J2": (12.9716, 77.6046),
    "J3": (12.9716, 77.6146),
    "J4": (12.9616, 77.5946),
    "J5": (12.9616, 77.6046),
    "J6": (12.9616, 77.6146),
}


# ============================================================
# ROUTE OPTIMIZER
# ============================================================

class RouteOptimizer:
    """
    NetworkX-based route optimizer for the traffic network.
    Computes shortest, traffic-aware, and emergency routes.
    """

    def __init__(self):
        self.graph = nx.Graph()
        self._build_graph()

    def _build_graph(self) -> None:
        """Build the base network graph."""

        for src, dst, attrs in NETWORK_EDGES:
            self.graph.add_edge(
                src, dst,
                weight=attrs["base_time"],
                base_time=attrs["base_time"],
                distance_km=attrs["distance_km"],
                congestion_factor=1.0,
                blocked=False
            )

        # Add node positions
        for node, pos in NODE_POSITIONS.items():
            self.graph.nodes[node]["pos"] = pos
            self.graph.nodes[node]["lat"] = pos[0]
            self.graph.nodes[node]["lon"] = pos[1]

    def update_weights_from_traffic(
        self,
        traffic_data: pd.DataFrame
    ) -> None:
        """
        Update edge weights based on live traffic conditions.

        Higher congestion → higher travel time → higher weight.

        Parameters
        ----------
        traffic_data : pd.DataFrame
            Traffic data with intersection_id, vehicle_density,
            queue_length, waiting_time, blocked columns.
        """

        # Build lookup for intersection conditions
        conditions = {}

        for _, row in traffic_data.iterrows():

            intersection = str(
                row.get("intersection_id", "")
            )

            conditions[intersection] = {
                "density": float(
                    row.get("vehicle_density", 0.3)
                ),
                "queue": int(
                    row.get("queue_length", 0)
                ),
                "waiting": float(
                    row.get("waiting_time", 20)
                ),
                "blocked": str(
                    row.get("blocked", False)
                ).lower() == "true",
                "speed": float(
                    row.get("speed_kmh", 50)
                )
            }

        # Update each edge
        for u, v, data in self.graph.edges(data=True):

            cond_u = conditions.get(u, {})
            cond_v = conditions.get(v, {})

            # Average density of both endpoints
            avg_density = (
                cond_u.get("density", 0.3) +
                cond_v.get("density", 0.3)
            ) / 2.0

            # Congestion factor: 1.0 (free flow) to 5.0 (jammed)
            congestion = 1.0 + avg_density * 4.0

            # Check if either endpoint is blocked
            is_blocked = (
                cond_u.get("blocked", False) or
                cond_v.get("blocked", False)
            )

            if is_blocked:
                congestion = 100.0  # Very high penalty

            data["congestion_factor"] = round(
                congestion, 2
            )

            data["blocked"] = is_blocked

            # Update weight = base_time * congestion
            data["weight"] = round(
                data["base_time"] * congestion,
                2
            )

    def shortest_path(
        self,
        source: str,
        target: str
    ) -> Optional[Dict]:
        """
        Find shortest path (unweighted / hop count).

        Returns
        -------
        dict or None
            {
                'path': list,
                'hops': int,
                'method': str
            }
        """

        try:
            path = nx.shortest_path(
                self.graph,
                source=source,
                target=target
            )

            return {
                "path": path,
                "hops": len(path) - 1,
                "method": "Shortest Path (Unweighted)"
            }

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def traffic_aware_path(
        self,
        source: str,
        target: str
    ) -> Optional[Dict]:
        """
        Find traffic-aware path using congestion-weighted edges.

        Returns
        -------
        dict or None
            {
                'path': list,
                'total_time': float,
                'total_distance': float,
                'method': str
            }
        """

        try:
            path = nx.shortest_path(
                self.graph,
                source=source,
                target=target,
                weight="weight"
            )

            total_time = nx.shortest_path_length(
                self.graph,
                source=source,
                target=target,
                weight="weight"
            )

            # Calculate total distance
            total_distance = 0.0
            for i in range(len(path) - 1):
                edge = self.graph[path[i]][path[i + 1]]
                total_distance += edge.get(
                    "distance_km", 0.5
                )

            return {
                "path": path,
                "total_time": round(total_time, 1),
                "total_distance": round(total_distance, 2),
                "method": "Traffic-Aware Path (Weighted)"
            }

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def emergency_route(
        self,
        source: str,
        target: str,
        blocked_nodes: Optional[List[str]] = None
    ) -> Optional[Dict]:
        """
        Find emergency route bypassing blocked intersections.

        Parameters
        ----------
        source : str
            Start intersection.
        target : str
            Destination intersection.
        blocked_nodes : list, optional
            Intersections to avoid.

        Returns
        -------
        dict or None
            Route information with bypass path.
        """

        if blocked_nodes is None:
            blocked_nodes = []

        # Create temporary graph excluding blocked nodes
        temp_graph = self.graph.copy()

        for node in blocked_nodes:
            if node in temp_graph:
                temp_graph.remove_node(node)

        try:
            path = nx.shortest_path(
                temp_graph,
                source=source,
                target=target,
                weight="weight"
            )

            total_time = nx.shortest_path_length(
                temp_graph,
                source=source,
                target=target,
                weight="weight"
            )

            total_distance = 0.0
            for i in range(len(path) - 1):
                edge = temp_graph[path[i]][path[i + 1]]
                total_distance += edge.get(
                    "distance_km", 0.5
                )

            return {
                "path": path,
                "total_time": round(total_time, 1),
                "total_distance": round(total_distance, 2),
                "blocked_avoided": blocked_nodes,
                "method": "Emergency Bypass Route"
            }

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_all_routes(
        self,
        source: str,
        target: str,
        blocked_nodes: Optional[List[str]] = None
    ) -> Dict:
        """
        Get all three route types for comparison.

        Returns
        -------
        dict
            {
                'shortest': ...,
                'traffic_aware': ...,
                'emergency': ...
            }
        """

        result = {
            "shortest": self.shortest_path(
                source, target
            ),
            "traffic_aware": self.traffic_aware_path(
                source, target
            ),
            "emergency": self.emergency_route(
                source, target, blocked_nodes
            )
        }

        return result

    def get_edge_data(self) -> List[Dict]:
        """
        Get all edge data for visualization.

        Returns
        -------
        list
            List of edge dicts with source, target, weight, etc.
        """

        edges = []

        for u, v, data in self.graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "weight": data.get("weight", 30),
                "base_time": data.get("base_time", 30),
                "distance_km": data.get("distance_km", 0.5),
                "congestion_factor": data.get(
                    "congestion_factor", 1.0
                ),
                "blocked": data.get("blocked", False),
            })

        return edges

    def get_node_positions(self) -> Dict[str, Tuple[float, float]]:
        """Get node lat/lon positions."""
        return NODE_POSITIONS.copy()
