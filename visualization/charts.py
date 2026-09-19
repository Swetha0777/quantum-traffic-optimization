"""
Module: Charts Visualization

Creates Plotly charts for:
    - Before vs After comparison bar charts
    - Throughput comparison
    - Fuel & CO2 emissions comparison
    - Queue length by intersection
    - Signal timing recommendations
    - Congestion score radar
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Optional


# ============================================================
# COLOR PALETTE
# ============================================================

COLORS = {
    "before": "#ef4444",
    "after": "#22c55e",
    "primary": "#3b82f6",
    "secondary": "#8b5cf6",
    "warning": "#f97316",
    "success": "#10b981",
    "danger": "#dc2626",
    "background": "#0f172a",
    "text": "#e2e8f0",
    "grid": "#1e293b",
}


# ============================================================
# BEFORE VS AFTER COMPARISON
# ============================================================

def create_comparison_chart(
    comparison: Dict,
    title: str = "Before vs After Optimization"
) -> go.Figure:
    """
    Create a grouped bar chart comparing before/after metrics.

    Parameters
    ----------
    comparison : dict
        Output from metrics.performance.compare_metrics().
    title : str
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
    """

    metrics = []
    before_values = []
    after_values = []
    improvements = []

    display_names = {
        "total_queue_length": "Queue Length",
        "average_waiting_time": "Avg Waiting (s)",
        "average_density": "Avg Density",
        "estimated_throughput": "Throughput",
        "estimated_fuel_litres": "Fuel (L)",
        "estimated_co2_kg": "CO₂ (kg)",
    }

    for key, name in display_names.items():
        if key in comparison:
            data = comparison[key]
            metrics.append(name)
            before_values.append(
                float(data.get("before", 0))
            )
            after_values.append(
                float(data.get("after", 0))
            )
            improvements.append(
                float(data.get("improvement_percent", 0))
            )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Before",
        x=metrics,
        y=before_values,
        marker_color=COLORS["before"],
        text=[f"{v:.1f}" for v in before_values],
        textposition="auto",
    ))

    fig.add_trace(go.Bar(
        name="After",
        x=metrics,
        y=after_values,
        marker_color=COLORS["after"],
        text=[f"{v:.1f}" for v in after_values],
        textposition="auto",
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=COLORS["text"])
        ),
        barmode="group",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(
            gridcolor=COLORS["grid"],
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            gridcolor=COLORS["grid"],
            title="Value"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400,
        margin=dict(l=40, r=20, t=60, b=40)
    )

    return fig


# ============================================================
# INTERSECTION QUEUE CHART
# ============================================================

def create_queue_chart(
    traffic_data: pd.DataFrame,
    title: str = "Queue Length by Intersection"
) -> go.Figure:
    """
    Create a bar chart showing queue length per intersection.
    """

    intersections = traffic_data["intersection_id"].astype(str).tolist()

    if "queue_length" in traffic_data.columns:
        queues = pd.to_numeric(
            traffic_data["queue_length"],
            errors="coerce"
        ).fillna(0).tolist()
    else:
        queues = [0] * len(intersections)

    # Color by congestion level
    if "congestion_level" in traffic_data.columns:
        colors = [
            COLORS["success"] if c == "LOW" else
            COLORS["warning"] if c in ["MEDIUM", "HIGH"] else
            COLORS["danger"]
            for c in traffic_data["congestion_level"]
        ]
    else:
        colors = [COLORS["primary"]] * len(intersections)

    fig = go.Figure(
        go.Bar(
            x=intersections,
            y=queues,
            marker_color=colors,
            text=[f"{q:.0f}" for q in queues],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color=COLORS["text"])
        ),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(
            title="Intersection",
            gridcolor=COLORS["grid"]
        ),
        yaxis=dict(
            title="Queue Length",
            gridcolor=COLORS["grid"]
        ),
        height=350,
        margin=dict(l=40, r=20, t=50, b=40)
    )

    return fig


# ============================================================
# SIGNAL TIMING CHART
# ============================================================

def create_signal_timing_chart(
    optimization_result: pd.DataFrame,
    title: str = "Signal Timing Optimization"
) -> go.Figure:
    """
    Create a chart showing original vs optimized green times.
    """

    if optimization_result is None or optimization_result.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No optimization data available",
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["background"],
            font=dict(color=COLORS["text"])
        )
        return fig

    intersections = optimization_result[
        "intersection_id"
    ].astype(str).tolist()

    original = pd.to_numeric(
        optimization_result.get(
            "original_green_time", pd.Series()
        ),
        errors="coerce"
    ).fillna(30).tolist()

    optimized = pd.to_numeric(
        optimization_result.get(
            "optimized_green_time", pd.Series()
        ),
        errors="coerce"
    ).fillna(30).tolist()

    decisions = optimization_result.get(
        "decision", pd.Series()
    ).fillna("KEEP").tolist()

    # Color optimized bars by decision
    opt_colors = [
        COLORS["success"] if d == "INCREASE_GREEN" else
        COLORS["danger"] if d == "EMERGENCY_PRIORITY" else
        COLORS["primary"]
        for d in decisions
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Original",
        x=intersections,
        y=original,
        marker_color=COLORS["before"],
        opacity=0.7,
    ))

    fig.add_trace(go.Bar(
        name="Optimized",
        x=intersections,
        y=optimized,
        marker_color=opt_colors,
        text=[f"{v:.0f}s" for v in optimized],
        textposition="auto",
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color=COLORS["text"])
        ),
        barmode="group",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(
            title="Intersection",
            gridcolor=COLORS["grid"]
        ),
        yaxis=dict(
            title="Green Time (seconds)",
            gridcolor=COLORS["grid"]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=350,
        margin=dict(l=40, r=20, t=50, b=40)
    )

    return fig


# ============================================================
# IMPROVEMENT SUMMARY CHART
# ============================================================

def create_improvement_chart(
    comparison: Dict,
    title: str = "Optimization Improvement (%)"
) -> go.Figure:
    """
    Create a horizontal bar chart showing improvement percentages.
    """

    metrics = []
    values = []
    colors = []

    display_names = {
        "total_queue_length": "Queue Reduction",
        "average_waiting_time": "Waiting Time Reduction",
        "estimated_throughput": "Throughput Increase",
        "estimated_fuel_litres": "Fuel Savings",
        "estimated_co2_kg": "CO₂ Reduction",
    }

    for key, name in display_names.items():
        if key in comparison:
            imp = float(
                comparison[key].get(
                    "improvement_percent", 0
                )
            )
            metrics.append(name)
            values.append(round(imp, 1))
            colors.append(
                COLORS["success"] if imp > 0 else
                COLORS["danger"]
            )

    fig = go.Figure(
        go.Bar(
            x=values,
            y=metrics,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in values],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color=COLORS["text"])
        ),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(
            title="Improvement (%)",
            gridcolor=COLORS["grid"],
            zeroline=True,
            zerolinecolor=COLORS["text"],
            zerolinewidth=1
        ),
        yaxis=dict(
            gridcolor=COLORS["grid"]
        ),
        height=350,
        margin=dict(l=160, r=60, t=50, b=40)
    )

    return fig


# ============================================================
# DENSITY GAUGE
# ============================================================

def create_density_gauge(
    density: float,
    title: str = "Average Traffic Density"
) -> go.Figure:
    """
    Create a gauge chart showing traffic density.
    """

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=density * 100,
            title=dict(
                text=title,
                font=dict(
                    size=14,
                    color=COLORS["text"]
                )
            ),
            number=dict(
                suffix="%",
                font=dict(
                    color=COLORS["text"]
                )
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickwidth=1,
                    tickcolor=COLORS["text"]
                ),
                bar=dict(
                    color=COLORS["primary"]
                ),
                bgcolor=COLORS["grid"],
                borderwidth=2,
                bordercolor=COLORS["text"],
                steps=[
                    dict(
                        range=[0, 35],
                        color="#166534"
                    ),
                    dict(
                        range=[35, 65],
                        color="#854d0e"
                    ),
                    dict(
                        range=[65, 85],
                        color="#9a3412"
                    ),
                    dict(
                        range=[85, 100],
                        color="#991b1b"
                    ),
                ],
                threshold=dict(
                    line=dict(
                        color=COLORS["danger"],
                        width=4
                    ),
                    thickness=0.75,
                    value=density * 100
                )
            )
        )
    )

    fig.update_layout(
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        height=250,
        margin=dict(l=30, r=30, t=40, b=20)
    )

    return fig
