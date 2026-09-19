"""
Module: Interactive Map Visualization

Creates Folium interactive maps showing:
    - Traffic intersection nodes with color-coded congestion
    - Emergency route polylines
    - Accident markers
    - Signal state indicators
    - Route optimization paths
"""

import folium
import pandas as pd
from typing import Dict, List, Optional


# ============================================================
# NODE POSITIONS (lat, lon)
# ============================================================

NODE_POSITIONS = {
    "J1": (12.9716, 77.5946),
    "J2": (12.9716, 77.6046),
    "J3": (12.9716, 77.6146),
    "J4": (12.9616, 77.5946),
    "J5": (12.9616, 77.6046),
    "J6": (12.9616, 77.6146),
}

# Road connections
ROAD_CONNECTIONS = [
    ("J1", "J2"),
    ("J1", "J4"),
    ("J2", "J3"),
    ("J2", "J5"),
    ("J3", "J4"),
    ("J3", "J6"),
    ("J4", "J5"),
    ("J5", "J6"),
]


# ============================================================
# CONGESTION COLORS
# ============================================================

CONGESTION_COLORS = {
    "LOW": "#22c55e",
    "MEDIUM": "#eab308",
    "HIGH": "#f97316",
    "CRITICAL": "#ef4444",
}


# ============================================================
# CREATE TRAFFIC MAP
# ============================================================

def create_traffic_map(
    traffic_data: pd.DataFrame,
    emergency_route: Optional[List[str]] = None,
    blocked_intersections: Optional[List[str]] = None,
    optimized_route: Optional[List[str]] = None,
    center: tuple = (12.9666, 77.6046),
    zoom: int = 15
) -> folium.Map:
    """
    Create an interactive Folium map showing traffic state.

    Parameters
    ----------
    traffic_data : pd.DataFrame
        Traffic data with intersection_id, congestion_level, etc.
    emergency_route : list, optional
        Intersections on the emergency corridor.
    blocked_intersections : list, optional
        Blocked/accident intersections.
    optimized_route : list, optional
        Optimized route path.
    center : tuple
        Map center (lat, lon).
    zoom : int
        Initial zoom level.

    Returns
    -------
    folium.Map
    """

    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="CartoDB dark_matter"
    )

    if blocked_intersections is None:
        blocked_intersections = []

    if emergency_route is None:
        emergency_route = []

    # Build lookup from traffic data
    data_lookup = {}

    for _, row in traffic_data.iterrows():

        intersection = str(
            row.get("intersection_id", "")
        )

        data_lookup[intersection] = {
            "density": float(
                row.get("vehicle_density", 0.3)
            ),
            "queue": int(
                row.get("queue_length", 0)
            ),
            "waiting": float(
                row.get("waiting_time", 20)
            ),
            "congestion": str(
                row.get("congestion_level", "LOW")
            ),
            "signal": str(
                row.get("current_signal", "NS_GREEN")
            ),
            "speed": float(
                row.get("speed_kmh", 50)
            ),
            "blocked": str(
                row.get("blocked", False)
            ).lower() == "true",
        }

    # --------------------------------------------------------
    # DRAW ROAD CONNECTIONS
    # --------------------------------------------------------

    for src, dst in ROAD_CONNECTIONS:

        if src in NODE_POSITIONS and dst in NODE_POSITIONS:

            src_pos = NODE_POSITIONS[src]
            dst_pos = NODE_POSITIONS[dst]

            # Color based on average congestion
            src_data = data_lookup.get(src, {})
            dst_data = data_lookup.get(dst, {})

            avg_density = (
                src_data.get("density", 0.3) +
                dst_data.get("density", 0.3)
            ) / 2.0

            if avg_density < 0.35:
                color = "#22c55e"
            elif avg_density < 0.65:
                color = "#eab308"
            elif avg_density < 0.85:
                color = "#f97316"
            else:
                color = "#ef4444"

            # Check if road is blocked
            if (src in blocked_intersections or
                    dst in blocked_intersections):
                color = "#dc2626"
                dash_array = "10 5"
            else:
                dash_array = None

            folium.PolyLine(
                locations=[src_pos, dst_pos],
                color=color,
                weight=4,
                opacity=0.8,
                dash_array=dash_array,
                tooltip=f"{src} → {dst}"
            ).add_to(m)

    # --------------------------------------------------------
    # DRAW INTERSECTION NODES
    # --------------------------------------------------------

    for node_id, (lat, lon) in NODE_POSITIONS.items():

        node_data = data_lookup.get(node_id, {})
        congestion = node_data.get("congestion", "LOW")
        color = CONGESTION_COLORS.get(congestion, "#22c55e")

        # Icon and styling
        is_blocked = node_id in blocked_intersections
        is_emergency = node_id in emergency_route

        if is_blocked:
            icon_symbol = "ban"
            icon_color = "red"
        elif is_emergency:
            icon_symbol = "ambulance"
            icon_color = "blue"
        else:
            icon_symbol = "traffic-light"
            icon_color = "green" if congestion == "LOW" else (
                "orange" if congestion in ["MEDIUM", "HIGH"]
                else "red"
            )

        # Popup content
        popup_html = f"""
        <div style="font-family: Arial; width: 220px;">
            <h4 style="margin: 0 0 8px 0; color: {color};">
                🚦 {node_id}
            </h4>
            <table style="font-size: 12px; width: 100%;">
                <tr>
                    <td><b>Congestion:</b></td>
                    <td style="color: {color};">
                        {congestion}
                    </td>
                </tr>
                <tr>
                    <td><b>Density:</b></td>
                    <td>{node_data.get('density', 0):.2f}</td>
                </tr>
                <tr>
                    <td><b>Queue:</b></td>
                    <td>{node_data.get('queue', 0)}</td>
                </tr>
                <tr>
                    <td><b>Waiting:</b></td>
                    <td>{node_data.get('waiting', 0):.1f}s</td>
                </tr>
                <tr>
                    <td><b>Speed:</b></td>
                    <td>{node_data.get('speed', 0):.1f} km/h</td>
                </tr>
                <tr>
                    <td><b>Signal:</b></td>
                    <td>{node_data.get('signal', 'NS_GREEN')}</td>
                </tr>
                <tr>
                    <td><b>Blocked:</b></td>
                    <td>{'⛔ YES' if is_blocked else '✅ No'}</td>
                </tr>
            </table>
        </div>
        """

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{node_id}: {congestion}",
            icon=folium.Icon(
                color=icon_color,
                icon=icon_symbol,
                prefix="fa"
            )
        ).add_to(m)

        # Density circle
        folium.CircleMarker(
            location=[lat, lon],
            radius=15 + node_data.get("density", 0) * 25,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.3,
            weight=2
        ).add_to(m)

    # --------------------------------------------------------
    # DRAW EMERGENCY ROUTE
    # --------------------------------------------------------

    if emergency_route and len(emergency_route) >= 2:
        route_coords = [
            NODE_POSITIONS[n]
            for n in emergency_route
            if n in NODE_POSITIONS
        ]

        if len(route_coords) >= 2:
            folium.PolyLine(
                locations=route_coords,
                color="#3b82f6",
                weight=6,
                opacity=0.9,
                dash_array="15 10",
                tooltip="🚑 Emergency Route"
            ).add_to(m)

    # --------------------------------------------------------
    # DRAW OPTIMIZED ROUTE
    # --------------------------------------------------------

    if optimized_route and len(optimized_route) >= 2:
        opt_coords = [
            NODE_POSITIONS[n]
            for n in optimized_route
            if n in NODE_POSITIONS
        ]

        if len(opt_coords) >= 2:
            folium.PolyLine(
                locations=opt_coords,
                color="#8b5cf6",
                weight=5,
                opacity=0.8,
                tooltip="🛣️ Optimized Route"
            ).add_to(m)

    # --------------------------------------------------------
    # ACCIDENT MARKERS
    # --------------------------------------------------------

    for node_id in blocked_intersections:
        if node_id in NODE_POSITIONS:
            lat, lon = NODE_POSITIONS[node_id]

            folium.Marker(
                location=[lat, lon],
                tooltip=f"⚠️ ACCIDENT at {node_id}",
                icon=folium.Icon(
                    color="red",
                    icon="exclamation-triangle",
                    prefix="fa"
                )
            ).add_to(m)

    return m
