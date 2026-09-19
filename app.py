"""
Quantum Traffic Optimization
Integrated Streamlit Application

Architecture:

LIVE MODE
Camera / Video
      ↓
FastAPI Vision Backend :8000
      ↓
YOLO + Tracking
      ↓
WebSocket /ws/traffic
      ↓
Telemetry Adapter
      ↓
Traffic Analytics
      ↓
QUBO
      ↓
QAOA / Classical Fallback
      ↓
Emergency / Accident / Routing
      ↓
Performance Dashboard

SIMULATION MODE
Traffic Simulation
      ↓
Traffic Analytics
      ↓
QUBO
      ↓
QAOA
      ↓
Emergency / Routing / Metrics
"""

import sys
import json
import base64
import threading
from pathlib import Path
from typing import Optional

import requests
import numpy as np
import pandas as pd
import streamlit as st

# Optional WebSocket dependency
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# BACKEND CONFIGURATION
# ============================================================

BACKEND_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws/traffic"


# ============================================================
# BACKEND REST CLIENT
# ============================================================

def backend_get(endpoint: str, timeout: int = 10):
    """GET request to FastAPI backend."""
    try:
        response = requests.get(
            f"{BACKEND_URL}{endpoint}",
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        return None

    except requests.exceptions.Timeout:
        return None

    except requests.exceptions.RequestException:
        return None


def backend_post(
    endpoint: str,
    payload=None,
    timeout: int = 60
):
    """POST request to FastAPI backend."""
    try:
        response = requests.post(
            f"{BACKEND_URL}{endpoint}",
            json=payload or {},
            timeout=timeout
        )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        st.error(f"Backend error: {e}")
        return None


# ============================================================
# BACKEND STATUS
# ============================================================

def backend_online():
    """Check whether FastAPI backend is available."""
    return backend_get("/api/status") is not None


# ============================================================
# WEBSOCKET LIVE CLIENT
# ============================================================

def receive_live_frame(timeout=5):
    """
    Receive one frame + telemetry packet from FastAPI WebSocket.
    """

    if not WEBSOCKET_AVAILABLE:
        return None

    try:
        ws = websocket.create_connection(
            WS_URL,
            timeout=timeout
        )

        raw = ws.recv()

        ws.close()

        if not raw:
            return None

        return json.loads(raw)

    except Exception:
        return None


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from simulation.traffic_simulation import (
    generate_traffic,
    apply_optimization,
    save_simulation,
    INTERSECTIONS,
)

from optimization.qaoa import (
    optimize_signals,
    get_solver_metadata,
)

from optimization.qubo import (
    create_qubo,
)

from emergency.emergency_corridor import (
    EmergencyVehicle,
    EmergencyCorridor,
)

from emergency.accident_detector import (
    AccidentDetector,
)

from emergency.emergency_detector import (
    EmergencyVehicleDetector,
)

from metrics.performance import (
    calculate_metrics,
    compare_metrics,
    intersection_metrics,
)

from traffic.density import (
    compute_density_for_dataframe,
)

from traffic.congestion import (
    compute_congestion_for_dataframe,
)

from traffic.queue import (
    detect_queue_from_dataframe,
    aggregate_queue_metrics,
)

from routing.route_optimizer import (
    RouteOptimizer,
)

from visualization.map_view import (
    create_traffic_map,
)

from visualization.charts import (
    create_comparison_chart,
    create_queue_chart,
    create_signal_timing_chart,
    create_improvement_chart,
    create_density_gauge,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Quantum Traffic Optimization",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .backend-online {
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #21c55d;
        background: rgba(33,197,93,0.08);
    }

    .backend-offline {
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #ef4444;
        background: rgba(239,68,68,0.08);
    }

    .emergency-box {
        padding: 18px;
        border-radius: 12px;
        border: 2px solid #ff4b4b;
        background-color: rgba(255, 75, 75, 0.08);
        margin-bottom: 20px;
    }

    .success-box {
        padding: 18px;
        border-radius: 12px;
        border: 2px solid #21c55d;
        background-color: rgba(33, 197, 93, 0.08);
    }

    .solver-box {
        padding: 15px;
        border-radius: 12px;
        border: 2px solid #6366f1;
        background-color: rgba(99, 102, 241, 0.08);
        margin-bottom: 15px;
    }

    .accident-box {
        padding: 15px;
        border-radius: 12px;
        border: 2px solid #f59e0b;
        background-color: rgba(245, 158, 11, 0.08);
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "traffic_data": None,
    "optimized_data": None,
    "optimization_result": None,
    "scenario": "Normal Traffic",
    "emergency_vehicle": None,
    "solver_metadata": None,
    "live_packet": None,
    "live_frame": None,
    "live_telemetry": None,
    "live_dataframe": None,
    "backend_status": None,
    "live_mode": False,
    "camera_started": False,
    "uploaded_video": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚦 Quantum Traffic Optimization</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Real-Time Vision + Traffic Intelligence + QUBO + QAOA +
    Emergency Priority + Route Optimization
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BACKEND STATUS
# ============================================================

status = backend_get("/api/status")

if status:

    st.markdown(
        """
        <div class="backend-online">
        🟢 <b>Vision Backend Connected</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
        <div class="backend-offline">
        🔴 <b>Vision Backend Offline</b>
        <br>
        Start:
        <code>python -m vision.backend.main</code>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Control Center")

mode = st.sidebar.radio(
    "Data Source",
    [
        "Live Vision",
        "Traffic Simulation",
    ],
)

scenario = st.sidebar.selectbox(
    "Traffic Scenario",
    [
        "Normal Traffic",
        "Heavy Congestion",
        "Accident / Road Closure",
        "Emergency Vehicle",
    ],
)

seed = st.sidebar.number_input(
    "Simulation Seed",
    min_value=0,
    max_value=999999,
    value=42,
    step=1,
)


# ============================================================
# LIVE VISION CONTROLS
# ============================================================

if mode == "Live Vision":

    st.sidebar.divider()
    st.sidebar.subheader("📹 Vision Control")

    start_camera_button = st.sidebar.button(
        "▶️ Start Camera",
        use_container_width=True,
    )

    stop_camera_button = st.sidebar.button(
        "⏹ Stop Camera",
        use_container_width=True,
    )

    reset_camera_button = st.sidebar.button(
        "🔄 Reset Counters",
        use_container_width=True,
    )

    if start_camera_button:

        result = backend_post(
            "/api/camera/start"
        )

        if result:

            st.session_state.camera_started = True
            st.session_state.live_mode = True

            st.sidebar.success(
                "Camera connected."
            )

    if stop_camera_button:

        result = backend_post(
            "/api/camera/stop"
        )

        if result:

            st.session_state.camera_started = False
            st.session_state.live_mode = False

            st.sidebar.success(
                "Camera stopped."
            )

    if reset_camera_button:

        result = backend_post(
            "/api/reset"
        )

        if result:

            st.sidebar.success(
                "Counters reset."
            )


# ============================================================
# SIMULATION CONTROLS
# ============================================================

run_button = st.sidebar.button(
    "🚀 Run Optimization",
    use_container_width=True,
)

generate_button = st.sidebar.button(
    "🔄 Generate Traffic",
    use_container_width=True,
)


# ============================================================
# ROUTE CONTROL
# ============================================================

st.sidebar.divider()

st.sidebar.subheader("🛣️ Route Optimizer")

route_source = st.sidebar.selectbox(
    "Route Source",
    INTERSECTIONS,
    index=0,
)

route_target = st.sidebar.selectbox(
    "Route Target",
    INTERSECTIONS,
    index=len(INTERSECTIONS) - 1,
)


# ============================================================
# LIVE TELEMETRY → DATAFRAME
# ============================================================

def telemetry_to_dataframe(telemetry):
    """
    Convert real-time vision telemetry into the traffic
    DataFrame expected by the optimization modules.
    """

    if not telemetry:
        return None

    # --------------------------------------------------------
    # Extract possible vehicle counts
    # --------------------------------------------------------

    total_vehicles = (
        telemetry.get("active_count")
        or telemetry.get("vehicle_count")
        or telemetry.get("total_vehicles")
        or telemetry.get("count")
        or 0
    )

    unique_vehicles = (
        telemetry.get("unique_count")
        or telemetry.get("unique_vehicles")
        or telemetry.get("unique_vehicle_count")
        or total_vehicles
    )

    # --------------------------------------------------------
    # Handle intersection-level telemetry
    # --------------------------------------------------------

    intersections = telemetry.get(
        "intersections"
    )

    rows = []

    if isinstance(intersections, dict):

        for intersection_id, data in intersections.items():

            if not isinstance(data, dict):
                data = {}

            vehicles = data.get(
                "vehicle_count",
                data.get("count", total_vehicles)
            )

            queue = data.get(
                "queue_length",
                max(0, int(vehicles * 0.25))
            )

            density = data.get(
                "vehicle_density",
                min(1.0, vehicles / 50.0)
            )

            waiting = data.get(
                "waiting_time",
                max(0.0, queue * 2.0)
            )

            rows.append({
                "intersection_id": intersection_id,
                "current_signal": data.get(
                    "current_signal",
                    "NS_GREEN"
                ),
                "vehicle_density": float(density),
                "queue_length": int(queue),
                "waiting_time": float(waiting),
                "throughput": float(
                    data.get(
                        "throughput",
                        vehicles
                    )
                ),
                "vehicle_count": int(vehicles),
                "unique_vehicle_count": int(
                    data.get(
                        "unique_vehicle_count",
                        unique_vehicles
                    )
                ),
            })

    # --------------------------------------------------------
    # No intersection telemetry
    # --------------------------------------------------------

    if not rows:

        intersection_list = (
            INTERSECTIONS
            if INTERSECTIONS
            else ["J1", "J2", "J3", "J4", "J5", "J6"]
        )

        # Distribute detected traffic across intersections
        count_per_intersection = max(
            0,
            int(total_vehicles / max(1, len(intersection_list)))
        )

        for intersection_id in intersection_list:

            density = min(
                1.0,
                count_per_intersection / 50.0
            )

            queue = max(
                0,
                int(count_per_intersection * 0.25)
            )

            waiting = queue * 2.0

            rows.append({
                "intersection_id": intersection_id,
                "current_signal": "NS_GREEN",
                "vehicle_density": float(density),
                "queue_length": int(queue),
                "waiting_time": float(waiting),
                "throughput": float(
                    count_per_intersection
                ),
                "vehicle_count": int(
                    count_per_intersection
                ),
                "unique_vehicle_count": int(
                    unique_vehicles
                ),
            })

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Calculate additional traffic intelligence
    # --------------------------------------------------------

    try:
        df = compute_density_for_dataframe(df)
    except Exception:
        pass

    try:
        df = compute_congestion_for_dataframe(df)
    except Exception:
        pass

    try:
        df = detect_queue_from_dataframe(df)
    except Exception:
        pass

    return df


# ============================================================
# LIVE VISION PROCESSING
# ============================================================

if mode == "Live Vision":

    st.header("🎥 Real-Time Vision")

    if not WEBSOCKET_AVAILABLE:

        st.error(
            "Install websocket-client:"
        )

        st.code(
            "pip install websocket-client"
        )

    elif not st.session_state.camera_started:

        st.info(
            "Start the camera from the sidebar."
        )

    else:

        with st.spinner(
            "Receiving YOLO detection..."
        ):

            packet = receive_live_frame(
                timeout=10
            )

        if packet:

            st.session_state.live_packet = packet

            telemetry = packet.get(
                "telemetry",
                {}
            )

            st.session_state.live_telemetry = (
                telemetry
            )

            st.session_state.live_frame = (
                packet.get("frame")
            )

            live_df = telemetry_to_dataframe(
                telemetry
            )

            st.session_state.live_dataframe = (
                live_df
            )

        if st.session_state.live_frame:

            try:

                image_data = (
                    st.session_state.live_frame
                )

                if "," in image_data:
                    image_data = (
                        image_data.split(",", 1)[1]
                    )

                image_bytes = base64.b64decode(
                    image_data
                )

                st.image(
                    image_bytes,
                    caption="Live YOLO Detection",
                    use_container_width=True,
                )

            except Exception as e:

                st.warning(
                    f"Could not display detection frame: {e}"
                )

        telemetry = (
            st.session_state.live_telemetry
        )

        if telemetry:

            st.subheader(
                "📡 Live Detection Telemetry"
            )

            # ------------------------------------------------
            # Top live metrics
            # ------------------------------------------------

            active_count = (
                telemetry.get(
                    "active_count",
                    telemetry.get(
                        "vehicle_count",
                        telemetry.get(
                            "count",
                            0
                        )
                    )
                )
            )

            unique_count = (
                telemetry.get(
                    "unique_count",
                    telemetry.get(
                        "unique_vehicles",
                        0
                    )
                )
            )

            fps = telemetry.get(
                "fps",
                telemetry.get(
                    "processing_fps",
                    0
                )
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "🚗 Active Vehicles",
                    int(active_count or 0)
                )

            with col2:
                st.metric(
                    "🆔 Unique Vehicles",
                    int(unique_count or 0)
                )

            with col3:
                st.metric(
                    "⚡ Processing FPS",
                    f"{float(fps or 0):.1f}"
                )

            # ------------------------------------------------
            # Raw telemetry
            # ------------------------------------------------

            with st.expander(
                "Raw Vision Telemetry"
            ):

                st.json(telemetry)

        # ----------------------------------------------------
        # Use vision data as optimization input
        # ----------------------------------------------------

        if st.session_state.live_dataframe is not None:

            traffic_data = (
                st.session_state.live_dataframe
            )

            st.session_state.traffic_data = (
                traffic_data
            )

        else:

            traffic_data = (
                st.session_state.traffic_data
            )

else:

    # ========================================================
    # SIMULATION MODE
    # ========================================================

    if generate_button or (
        st.session_state.traffic_data is None
        or st.session_state.scenario != scenario
    ):

        try:

            traffic_data = generate_traffic(
                scenario=scenario,
                seed=seed,
            )

            st.session_state.traffic_data = (
                traffic_data
            )

            st.session_state.scenario = (
                scenario
            )

            st.session_state.optimized_data = None
            st.session_state.optimization_result = None
            st.session_state.solver_metadata = None

        except Exception as e:

            st.error(
                f"Traffic generation failed: {e}"
            )

            st.stop()

    traffic_data = (
        st.session_state.traffic_data
    )


# ============================================================
# ENSURE DATA EXISTS
# ============================================================

traffic_data = (
    st.session_state.traffic_data
    if traffic_data is None
    else traffic_data
)

traffic_data = st.session_state.get("traffic_data")

if traffic_data is None:

    st.info(
        "No traffic data available."
    )

    st.stop()


# ============================================================
# EMERGENCY VEHICLE
# ============================================================

if scenario == "Emergency Vehicle":

    if (
        st.session_state.emergency_vehicle
        is None
    ):

        st.session_state.emergency_vehicle = (
            EmergencyVehicle(
                vehicle_id="EV-01",
                route=[
                    "J1",
                    "J2",
                    "J4",
                ],
            )
        )

    emergency_vehicle = (
        st.session_state.emergency_vehicle
    )

    corridor = EmergencyCorridor(
        route=[
            "J1",
            "J2",
            "J4",
        ]
    )

    try:

        traffic_data = (
            corridor.mark_traffic_data(
                traffic_data,
                emergency_vehicle,
            )
        )

        st.session_state.traffic_data = (
            traffic_data
        )

    except Exception as e:

        st.warning(
            f"Emergency corridor update failed: {e}"
        )


# ============================================================
# SCENARIO INFORMATION
# ============================================================

if scenario == "Emergency Vehicle":

    ev = (
        st.session_state.emergency_vehicle
    )

    st.markdown(
        f"""
        <div class="emergency-box">
        <h3>🚑 Emergency Vehicle Active</h3>

        <b>Vehicle:</b> {ev.vehicle_id}<br>
        <b>Current Intersection:</b>
        {ev.current_intersection}<br>
        <b>Next Intersection:</b>
        {ev.next_intersection}<br>
        <b>Emergency Route:</b>
        J1 → J2 → J4

        </div>
        """,
        unsafe_allow_html=True,
    )

elif scenario == "Accident / Road Closure":

    st.warning(
        "⚠️ Accident scenario active."
    )

    try:

        accident_detector = AccidentDetector()

        alerts = (
            accident_detector
            .detect_from_dataframe(
                traffic_data
            )
        )

        if alerts:

            for alert in alerts:

                st.markdown(
                    f"""
                    <div class="accident-box">
                    <b>⚠️ Incident Alert —
                    {alert['intersection_id']}</b><br>

                    Severity:
                    <b>{alert['severity']}</b>
                    |
                    Score:
                    {alert['accident_score']}<br>

                    Reasons:
                    {', '.join(alert['reasons'])}<br>

                    Action:
                    {alert['recommended_action']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    except Exception as e:

        st.warning(
            f"Accident detection unavailable: {e}"
        )

elif scenario == "Heavy Congestion":

    st.warning(
        "🚗 Heavy congestion scenario active."
    )

else:

    st.success(
        "✅ Normal traffic scenario active."
    )


# ============================================================
# RUN QAOA / OPTIMIZATION
# ============================================================

if run_button:

    with st.spinner(
        "Running QUBO + QAOA optimization..."
    ):

        try:

            optimization_result = (
                optimize_signals(
                    traffic_data
                )
            )

            try:

                st.session_state.solver_metadata = (
                    get_solver_metadata()
                )

            except Exception:

                st.session_state.solver_metadata = {}

            # ------------------------------------------------
            # EMERGENCY OVERRIDE
            # ------------------------------------------------

            if scenario == "Emergency Vehicle":

                emergency_vehicle = (
                    st.session_state
                    .emergency_vehicle
                )

                corridor = EmergencyCorridor(
                    route=[
                        "J1",
                        "J2",
                        "J4",
                    ]
                )

                overrides = (
                    corridor.create_signal_override(
                        traffic_data,
                        emergency_vehicle,
                    )
                )

                optimization_result = (
                    optimization_result.copy()
                )

                if "emergency_priority" not in (
                    optimization_result.columns
                ):

                    optimization_result[
                        "emergency_priority"
                    ] = "NONE"

                for index, row in (
                    optimization_result.iterrows()
                ):

                    intersection = str(
                        row[
                            "intersection_id"
                        ]
                    )

                    if intersection not in overrides:
                        continue

                    override = (
                        overrides[
                            intersection
                        ]
                    )

                    priority = (
                        override[
                            "priority"
                        ]
                    )

                    if priority in [
                        "ACTIVE",
                        "PREPARE",
                        "CORRIDOR",
                    ]:

                        optimization_result.loc[
                            index,
                            "optimized_green_time"
                        ] = override[
                            "green_time"
                        ]

                        optimization_result.loc[
                            index,
                            "decision"
                        ] = (
                            "EMERGENCY_PRIORITY"
                        )

                        optimization_result.loc[
                            index,
                            "emergency_priority"
                        ] = priority

            # ------------------------------------------------
            # APPLY OPTIMIZATION
            # ------------------------------------------------

            optimized_data = (
                apply_optimization(
                    traffic_data,
                    optimization_result,
                )
            )

            # ------------------------------------------------
            # EMERGENCY FINAL ADJUSTMENT
            # ------------------------------------------------

            if (
                scenario == "Emergency Vehicle"
                and st.session_state.emergency_vehicle
            ):

                active_intersection = (
                    st.session_state
                    .emergency_vehicle
                    .current_intersection
                )

                for index, row in (
                    optimized_data.iterrows()
                ):

                    if str(
                        row["intersection_id"]
                    ) == str(
                        active_intersection
                    ):

                        if "waiting_time" in (
                            optimized_data.columns
                        ):

                            optimized_data.loc[
                                index,
                                "waiting_time"
                            ] *= 0.35

                        if "queue_length" in (
                            optimized_data.columns
                        ):

                            optimized_data.loc[
                                index,
                                "queue_length"
                            ] *= 0.50

            # ------------------------------------------------
            # STORE
            # ------------------------------------------------

            st.session_state.optimization_result = (
                optimization_result
            )

            st.session_state.optimized_data = (
                optimized_data
            )

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            try:

                save_simulation(
                    optimized_data,
                    "data/traffic_data.csv",
                )

            except Exception:
                pass

            st.success(
                "✅ Optimization completed successfully."
            )

        except Exception as e:

            st.error(
                f"Optimization failed: {e}"
            )

            st.exception(e)


# ============================================================
# RESULTS
# ============================================================

optimization_result = (
    st.session_state.optimization_result
)

optimized_data = (
    st.session_state.optimized_data
)


# ============================================================
# SOLVER METADATA
# ============================================================

solver_meta = (
    st.session_state.solver_metadata
)

if solver_meta:

    solver_name = solver_meta.get(
        "solver_used",
        "N/A"
    )

    quantum_used = solver_meta.get(
        "quantum_used",
        False
    )

    objective = solver_meta.get(
        "objective_value",
        0
    )

    try:
        objective_text = (
            f"{float(objective):.6f}"
        )
    except Exception:
        objective_text = str(objective)

    st.markdown(
        f"""
        <div class="solver-box">

        <h4>⚛️ Quantum Solver Details</h4>

        <b>Solver:</b> {solver_name}<br>

        <b>Quantum Used:</b>
        {"✅ Yes" if quantum_used else "❌ Classical Fallback"}<br>

        <b>Objective Value:</b>
        {objective_text}

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TRAFFIC METRICS
# ============================================================

st.header("📊 Traffic Overview")

try:

    before_metrics = calculate_metrics(
        traffic_data
    )

except Exception:

    before_metrics = {
        "total_traffic": len(traffic_data),
        "total_queue_length": 0,
        "average_waiting_time": 0,
        "estimated_throughput": 0,
        "average_density": 0,
        "estimated_fuel_litres": 0,
        "estimated_co2_kg": 0,
        "congestion_level": "Unknown",
    }


if optimized_data is not None:

    try:

        after_metrics = calculate_metrics(
            optimized_data
        )

    except Exception:

        after_metrics = before_metrics

else:

    after_metrics = before_metrics


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "🚗 Total Traffic",
        f"{before_metrics.get('total_traffic', 0):.0f}",
    )

with col2:

    st.metric(
        "🚦 Queue",
        f"{before_metrics.get('total_queue_length', 0):.0f}",
    )

with col3:

    st.metric(
        "⏱ Avg Waiting",
        f"{before_metrics.get('average_waiting_time', 0):.1f} s",
    )

with col4:

    st.metric(
        "📈 Throughput",
        f"{before_metrics.get('estimated_throughput', 0):.0f}",
    )


# ============================================================
# LIVE / CURRENT TRAFFIC TABLE
# ============================================================

st.header("🛣️ Current Traffic State")

display_columns = [
    "intersection_id",
    "current_signal",
    "vehicle_density",
    "queue_length",
    "waiting_time",
    "throughput",
    "congestion_level",
    "vehicle_count",
    "unique_vehicle_count",
]

available_columns = [
    c for c in display_columns
    if c in traffic_data.columns
]

st.dataframe(
    traffic_data[available_columns],
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# QUBO MATRIX
# ============================================================

if optimization_result is not None:

    with st.expander(
        "🔢 QUBO Matrix",
        expanded=False
    ):

        try:

            qubo_result = create_qubo(
                traffic_data
            )

            Q = qubo_result["Q"]

            st.write(
                "**QUBO Matrix Q:**"
            )

            q_df = pd.DataFrame(
                np.round(Q, 3),
                columns=qubo_result[
                    "intersections"
                ],
                index=qubo_result[
                    "intersections"
                ],
            )

            st.dataframe(
                q_df,
                use_container_width=True,
            )

            st.write(
                "**Demand Scores:**"
            )

            st.json(
                qubo_result.get(
                    "demand_scores",
                    {}
                )
            )

            if qubo_result.get(
                "priority_scores"
            ):

                st.write(
                    "**Priority Scores:**"
                )

                st.json(
                    qubo_result[
                        "priority_scores"
                    ]
                )

        except Exception as e:

            st.error(
                f"QUBO display error: {e}"
            )


# ============================================================
# OPTIMIZATION RESULTS
# ============================================================

if optimization_result is not None:

    st.header(
        "⚛️ Quantum Optimization Results"
    )

    result_columns = [
        "intersection_id",
        "current_signal",
        "original_green_time",
        "optimized_green_time",
        "decision",
        "estimated_waiting_time",
        "emergency_priority",
    ]

    available_result_columns = [
        c for c in result_columns
        if c in optimization_result.columns
    ]

    st.dataframe(
        optimization_result[
            available_result_columns
        ],
        use_container_width=True,
        hide_index=True,
    )

    try:

        st.plotly_chart(
            create_signal_timing_chart(
                optimization_result
            ),
            use_container_width=True,
        )

    except Exception as e:

        st.warning(
            f"Signal chart unavailable: {e}"
        )


# ============================================================
# BEFORE VS AFTER
# ============================================================

if optimized_data is not None:

    st.header(
        "📈 Before vs After Optimization"
    )

    try:

        comparison = compare_metrics(
            traffic_data,
            optimized_data,
        )

        queue_change = comparison[
            "total_queue_length"
        ]

        waiting_change = comparison[
            "average_waiting_time"
        ]

        throughput_change = comparison[
            "estimated_throughput"
        ]

        co2_change = comparison[
            "estimated_co2_kg"
        ]

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Queue",
                f"{queue_change['after']:.1f}",
                f"{queue_change['change']:.1f}",
            )

        with col2:

            st.metric(
                "Waiting Time",
                f"{waiting_change['after']:.1f} s",
                f"{waiting_change['change']:.1f} s",
            )

        with col3:

            st.metric(
                "Throughput",
                f"{throughput_change['after']:.1f}",
                f"{throughput_change['change']:.1f}",
            )

        with col4:

            st.metric(
                "CO₂",
                f"{co2_change['after']:.2f} kg",
                f"{co2_change['change']:.2f}",
            )

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            st.plotly_chart(
                create_comparison_chart(
                    comparison
                ),
                use_container_width=True,
            )

        with chart_col2:

            st.plotly_chart(
                create_improvement_chart(
                    comparison
                ),
                use_container_width=True,
            )

    except Exception as e:

        st.warning(
            f"Comparison unavailable: {e}"
        )


# ============================================================
# INTERSECTION ANALYSIS
# ============================================================

st.header(
    "📍 Intersection Performance"
)

try:

    intersection_data = intersection_metrics(
        optimized_data
        if optimized_data is not None
        else traffic_data
    )

    if not intersection_data.empty:

        st.dataframe(
            intersection_data,
            use_container_width=True,
            hide_index=True,
        )

except Exception as e:

    st.warning(
        f"Intersection analysis unavailable: {e}"
    )


# ============================================================
# QUEUE CHART
# ============================================================

try:

    st.plotly_chart(
        create_queue_chart(
            optimized_data
            if optimized_data is not None
            else traffic_data
        ),
        use_container_width=True,
    )

except Exception:
    pass


# ============================================================
# DENSITY GAUGES
# ============================================================

gauge_col1, gauge_col2 = st.columns(2)

with gauge_col1:

    try:

        avg_density = before_metrics.get(
            "average_density",
            0
        )

        st.plotly_chart(
            create_density_gauge(
                avg_density,
                "Before Optimization"
            ),
            use_container_width=True,
        )

    except Exception:
        pass


with gauge_col2:

    try:

        after_density = (
            after_metrics.get(
                "average_density",
                0
            )
        )

        st.plotly_chart(
            create_density_gauge(
                after_density,
                "After Optimization"
            ),
            use_container_width=True,
        )

    except Exception:
        pass


# ============================================================
# ROUTE OPTIMIZATION
# ============================================================

st.header(
    "🛣️ Route Optimization"
)

routes = {}

try:

    router = RouteOptimizer()

    current_traffic = (
        optimized_data
        if optimized_data is not None
        else traffic_data
    )

    router.update_weights_from_traffic(
        current_traffic
    )

    blocked_nodes = []

    if scenario == "Accident / Road Closure":
        blocked_nodes = ["J3"]

    routes = router.get_all_routes(
        route_source,
        route_target,
        blocked_nodes=blocked_nodes,
    )

    route_col1, route_col2, route_col3 = (
        st.columns(3)
    )

    with route_col1:

        st.subheader(
            "📏 Shortest Path"
        )

        shortest = routes.get(
            "shortest"
        )

        if shortest:

            st.write(
                f"**Path:** "
                f"{' → '.join(shortest['path'])}"
            )

            st.write(
                f"**Hops:** "
                f"{shortest['hops']}"
            )

        else:

            st.warning(
                "No path found."
            )

    with route_col2:

        st.subheader(
            "🚦 Traffic-Aware"
        )

        traffic_route = routes.get(
            "traffic_aware"
        )

        if traffic_route:

            st.write(
                f"**Path:** "
                f"{' → '.join(traffic_route['path'])}"
            )

            st.write(
                f"**Est. Time:** "
                f"{traffic_route['total_time']:.1f}s"
            )

            st.write(
                f"**Distance:** "
                f"{traffic_route['total_distance']:.2f} km"
            )

        else:

            st.warning(
                "No path found."
            )

    with route_col3:

        st.subheader(
            "🚑 Emergency Bypass"
        )

        emergency_route = routes.get(
            "emergency"
        )

        if emergency_route:

            st.write(
                f"**Path:** "
                f"{' → '.join(emergency_route['path'])}"
            )

            st.write(
                f"**Est. Time:** "
                f"{emergency_route['total_time']:.1f}s"
            )

            if emergency_route.get(
                "blocked_avoided"
            ):

                st.write(
                    f"**Avoided:** "
                    f"{', '.join(emergency_route['blocked_avoided'])}"
                )

        else:

            st.warning(
                "No bypass route available."
            )

except Exception as e:

    st.error(
        f"Route optimization error: {e}"
    )


# ============================================================
# INTERACTIVE MAP
# ============================================================

st.header(
    "🗺️ Traffic Network Map"
)

try:

    import streamlit_folium

    emergency_route_nodes = []

    if scenario == "Emergency Vehicle":

        emergency_route_nodes = [
            "J1",
            "J2",
            "J4"
        ]

    blocked_nodes_map = []

    if scenario == "Accident / Road Closure":

        blocked_nodes_map = ["J3"]

    opt_route = None

    if routes.get(
        "traffic_aware"
    ):

        opt_route = routes[
            "traffic_aware"
        ]["path"]

    traffic_map = create_traffic_map(
        traffic_data=current_traffic,
        emergency_route=(
            emergency_route_nodes
        ),
        blocked_intersections=(
            blocked_nodes_map
        ),
        optimized_route=opt_route,
    )

    streamlit_folium.folium_static(
        traffic_map,
        width=None,
        height=500,
    )

except ImportError:

    st.info(
        "Install streamlit-folium:"
    )

    st.code(
        "pip install streamlit-folium"
    )

except Exception as e:

    st.error(
        f"Map display error: {e}"
    )


# ============================================================
# EMERGENCY METRICS
# ============================================================

if scenario == "Emergency Vehicle":

    st.header(
        "🚑 Emergency Performance"
    )

    data_for_emergency = (
        optimized_data
        if optimized_data is not None
        else traffic_data
    )

    try:

        emergency_metrics = (
            calculate_metrics(
                data_for_emergency
            )
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Emergency Corridor Wait",
                f"{emergency_metrics.get('emergency_corridor_waiting_time', 0):.2f} s",
            )

        with col2:

            st.metric(
                "Emergency Response Delay",
                f"{emergency_metrics.get('emergency_response_delay', 0):.2f} s",
            )

        with col3:

            st.metric(
                "Active Intersection",
                str(
                    st.session_state
                    .emergency_vehicle
                    .current_intersection
                ),
            )

        st.markdown(
            """
            <div class="success-box">

            🚑 <b>Emergency corridor:</b>

            J1 → J2 → J4

            <br><br>

            The active emergency intersection receives
            the highest signal priority.

            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as e:

        st.warning(
            f"Emergency metrics unavailable: {e}"
        )


# ============================================================
# ENVIRONMENTAL METRICS
# ============================================================

st.header(
    "🌱 Environmental Impact"
)

environment_data = (
    optimized_data
    if optimized_data is not None
    else traffic_data
)

try:

    environment_metrics = (
        calculate_metrics(
            environment_data
        )
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Fuel Consumption",
            f"{environment_metrics.get('estimated_fuel_litres', 0):.2f} L",
        )

    with col2:

        st.metric(
            "CO₂ Emissions",
            f"{environment_metrics.get('estimated_co2_kg', 0):.2f} kg",
        )

    with col3:

        st.metric(
            "Congestion",
            environment_metrics.get(
                "congestion_level",
                "Unknown"
            ),
        )

except Exception as e:

    st.warning(
        f"Environmental metrics unavailable: {e}"
    )


# ============================================================
# EXPORT
# ============================================================

st.header(
    "📥 Export"
)

csv_data = (
    optimized_data
    if optimized_data is not None
    else traffic_data
)

csv_bytes = csv_data.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Traffic Results",
    data=csv_bytes,
    file_name="optimized_traffic.csv",
    mime="text/csv",
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Quantum Traffic Optimization | "
    "Real-Time YOLO Detection + "
    "QUBO + QAOA + Emergency Priority + "
    "Traffic Analytics + Route Optimization + "
    "Performance Analytics"
)