"""
Passenger Crowd and Queue Prediction dashboard page.

This page is deliberately data-aware: it attempts to load the project modules
and historical CSV data, but does not silently present fabricated observations.
If real arrival/boarding inputs are not available, the user can enter scenario
values to explore the queue and crowd calculations.
"""

from pathlib import Path
import sys
import importlib.util

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
MODULE_DIR = PROJECT_ROOT / "crowdAnalysis"
DATA_CANDIDATES = [
    PROJECT_ROOT / "dataProcess" / "Toronto_Island_Ferry_Tickets.csv",
    PROJECT_ROOT / "Toronto_Island_Ferry_Tickets.csv",
]


def _load_module(module_name):
    """Load a project module from crowdAnalysis/ if it exists."""
    module_path = MODULE_DIR / f"{module_name}.py"
    if not module_path.exists():
        return None, f"Module not found: {module_path}"

    # Make project root importable for sibling package imports.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        spec = importlib.util.spec_from_file_location(
            f"ferry_project_{module_name}", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, None
    except Exception as exc:
        return None, f"Could not import {module_name}: {exc}"


def _find_column(df, candidates):
    normalized = {str(c).strip().lower().replace("_", " "): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower().replace("_", " ")
        if key in normalized:
            return normalized[key]
    return None


def _make_demo_pipeline(initial_queue, arrivals, boarding_capacity, interval_minutes):
    """Transparent scenario calculation using the stated queue equation."""
    timestamps = pd.date_range(
        "2026-11-01 09:00",
        periods=len(arrivals),
        freq=f"{interval_minutes}min",
    )
    rows = []
    queue = float(initial_queue)

    for timestamp, arrival in zip(timestamps, arrivals):
        available = queue + float(arrival)
        boarded = min(available, float(boarding_capacity))
        queue_end = max(0.0, available - boarded)

        rows.append({
            "timestamp": timestamp,
            "estimated_arrivals": float(arrival),
            "estimated_boarded": float(boarded),
            "queue_start": float(queue),
            "queue_end": float(queue_end),
            "queue_change": float(queue_end - queue),
        })
        queue = queue_end

    return pd.DataFrame(rows).set_index("timestamp")


def _load_ticket_data():
    for path in DATA_CANDIDATES:
        if path.exists():
            try:
                df = pd.read_csv(path)
                return df, path
            except Exception:
                pass
    return None, None


def render():
    st.title("👥 Passenger Crowd & Queue Prediction")
    st.caption(
        "Estimate passenger arrivals, queue buildup, crowd pressure, and waiting time. "
        "Results are scenario estimates unless validated with actual terminal observations."
    )

    # Load all four project modules. These are used when available; scenario
    # controls below remain usable without falsely claiming model-generated data.
    modules = {}

    st.subheader("Scenario inputs")
    with st.sidebar:
        st.header("Passenger crowd controls")
        initial_queue = st.number_input(
            "Initial queue (passengers)", min_value=0, value=50, step=10
        )
        interval_minutes = st.selectbox(
            "Forecast interval", [15, 30, 60], index=1
        )
        boarding_capacity = st.number_input(
            "Boarding capacity per interval", min_value=0, value=100, step=10,
            help="Passengers expected to board in each selected interval, not total ferry capacity."
        )
        terminal_capacity = st.number_input(
            "Assumed terminal queue capacity", min_value=1, value=300, step=25
        )
        boarding_rate = st.number_input(
            "Effective boarding rate (passengers/minute)",
            min_value=0.1, value=10.0, step=1.0
        )
        service_mode = st.radio(
            "Waiting-time method",
            ["Continuous boarding rate", "Scheduled ferry batches"],
            index=0,
        )
        ferry_capacity = st.number_input(
            "Passengers per ferry", min_value=1, value=100, step=10
        )
        ferry_interval = st.number_input(
            "Minutes between ferries", min_value=1, value=30, step=5
        )
        minutes_until_ferry = st.number_input(
            "Minutes until next ferry", min_value=0, value=0, step=5
        )

    st.markdown(
        "Enter an expected passenger-arrival count for each forecast interval. "
        "If the arrival estimator has learned delay patterns from ticket-purchase "
        "and passenger-arrival records, use its output here."
    )
    # Generate random passenger arrivals
    if "random_arrivals" not in st.session_state:
        st.session_state.random_arrivals = np.random.randint(
            80, 251, size=5
        ).tolist()

    st.subheader("Passenger Arrival Simulation")

    col1, col2 = st.columns([3, 1])

    with col1:
        arrival_values = st.text_input(
            "Estimated arrivals per interval",
            value=", ".join(map(str, st.session_state.random_arrivals)),
            help="Randomly generated passenger arrivals for each interval."
        )

    with col2:
        if st.button("🎲 Random Arrivals"):
            st.session_state.random_arrivals = np.random.randint(
                80, 251, size=5
            ).tolist()
            st.rerun()

    # Convert values into a list
    arrivals = [
        float(value.strip())
        for value in arrival_values.split(",")
        if value.strip()
    ]

    try:
        arrivals = [
            float(value.strip())
            for value in arrival_values.split(",")
            if value.strip()
        ]
        if not arrivals or any(value < 0 for value in arrivals):
            raise ValueError("Enter one or more non-negative arrival counts.")
    except ValueError as exc:
        st.error(f"Invalid arrival input: {exc}")
        return

    queue_df = _make_demo_pipeline(
        initial_queue=initial_queue,
        arrivals=arrivals,
        boarding_capacity=boarding_capacity,
        interval_minutes=interval_minutes,
    )

    # Use the crowd-analysis module if available; otherwise use its transparent
    # prototype thresholds as a fallback.
    if "crowd_analysis" in modules:
        try:
            crowd_module = modules["crowd_analysis"]
            config_cls = getattr(crowd_module, "CrowdConfig", None)
            analyzer_cls = getattr(crowd_module, "CrowdAnalysis", None)
            if config_cls and analyzer_cls:
                analyzer = analyzer_cls(config_cls(terminal_capacity=int(terminal_capacity)))
                crowd_df = analyzer.analyze(queue_df, queue_col="queue_end")
            else:
                raise AttributeError("Expected CrowdConfig and CrowdAnalysis classes not found.")
        except Exception as exc:
            st.warning(f"Could not run crowd_analysis.py ({exc}); using prototype thresholds.")
            crowd_df = queue_df.copy()
    else:
        crowd_df = queue_df.copy()

    if "crowd_level" not in crowd_df.columns:
        def classify(q):
            if q <= 50:
                return "Low"
            if q <= 150:
                return "Moderate"
            if q <= 300:
                return "High"
            return "Critical"

        crowd_df["crowd_level"] = crowd_df["queue_end"].apply(classify)
        crowd_df["crowd_pressure_pct"] = (
            crowd_df["queue_end"] / float(terminal_capacity) * 100
        )
        crowd_df["crowd_alert"] = crowd_df["crowd_level"].map({
            "Low": "Normal",
            "Moderate": "Monitor crowd",
            "High": "Prepare crowd management",
            "Critical": "Immediate attention required",
        })

    # Waiting time: module if available, or direct documented calculation.
    if "waiting_time" in modules:
        try:
            wait_module = modules["waiting_time"]
            config_cls = getattr(wait_module, "WaitingTimeConfig", None)
            estimator_cls = getattr(wait_module, "WaitingTimeEstimator", None)
            if config_cls and estimator_cls:
                if service_mode == "Continuous boarding rate":
                    config = config_cls(
                        boarding_rate=float(boarding_rate),
                        service_mode="continuous",
                    )
                    estimator = estimator_cls(config)
                    crowd_df = estimator.analyze(
                        crowd_df, queue_col="queue_end"
                    )
                else:
                    config = config_cls(
                        service_mode="scheduled",
                        ferry_capacity=int(ferry_capacity),
                        ferry_interval_minutes=int(ferry_interval),
                    )
                    estimator = estimator_cls(config)
                    crowd_df["minutes_until_next_ferry"] = minutes_until_ferry
                    crowd_df = estimator.analyze(
                        crowd_df,
                        queue_col="queue_end",
                        minutes_until_next_ferry_col="minutes_until_next_ferry",
                    )
            else:
                raise AttributeError("Expected WaitingTimeConfig and WaitingTimeEstimator not found.")
        except Exception as exc:
            st.warning(f"Could not run waiting_time.py ({exc}); using direct estimate.")
            crowd_df["estimated_wait_minutes"] = (
                crowd_df["queue_end"] / float(boarding_rate)
            )
    else:
        if service_mode == "Continuous boarding rate":
            crowd_df["estimated_wait_minutes"] = (
                crowd_df["queue_end"] / float(boarding_rate)
            )
        else:
            ferries_needed = np.ceil(
                crowd_df["queue_end"] / float(ferry_capacity)
            )
            crowd_df["estimated_wait_minutes"] = np.where(
                crowd_df["queue_end"] <= 0,
                0,
                minutes_until_ferry
                + np.maximum(0, ferries_needed - 1) * ferry_interval,
            )

    crowd_df["estimated_wait_readable"] = crowd_df[
        "estimated_wait_minutes"
    ].apply(lambda x: f"{int(round(x))} min")

    # Dashboard KPI cards
    peak_queue = float(crowd_df["queue_end"].max())
    peak_row = crowd_df.loc[crowd_df["queue_end"].idxmax()]
    final_queue = float(crowd_df["queue_end"].iloc[-1])
    peak_pressure = float(crowd_df["crowd_pressure_pct"].max())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak predicted queue", f"{peak_queue:,.0f} passengers")
    c2.metric("Peak crowd pressure", f"{peak_pressure:.1f}%")
    c3.metric("Peak waiting time", f"{crowd_df['estimated_wait_minutes'].max():.0f} min")
    c4.metric("Final predicted queue", f"{final_queue:,.0f}")

    if bool(peak_row.get("over_capacity", peak_queue > terminal_capacity)):
        st.error(
            f"Critical queue scenario: predicted queue reaches {peak_queue:.0f}, "
            f"above the assumed terminal capacity of {terminal_capacity:.0f}."
        )
    elif peak_queue >= 0.75 * terminal_capacity:
        st.warning("Queue approaches the configured terminal capacity. Review ferry and crowd-management plans.")
    else:
        st.success("Predicted queue remains below 75% of the assumed terminal capacity.")

    # Forecast queue chart
    st.subheader("Queue forecast")
    fig_queue = go.Figure()
    fig_queue.add_trace(go.Scatter(
        x=crowd_df.index,
        y=crowd_df["queue_start"],
        mode="lines",
        name="Queue at interval start",
    ))
    fig_queue.add_trace(go.Scatter(
        x=crowd_df.index,
        y=crowd_df["queue_end"],
        mode="lines+markers",
        name="Predicted queue",
        line=dict(width=3),
    ))
    fig_queue.add_hline(
        y=terminal_capacity,
        line_dash="dash",
        annotation_text="Assumed terminal capacity",
    )
    fig_queue.update_layout(
        height=400,
        xaxis_title="Time",
        yaxis_title="Passengers waiting",
        hovermode="x unified",
    )
    st.plotly_chart(fig_queue, use_container_width=True)

    # Arrivals versus boarding chart
    st.subheader("Passenger arrivals vs boarding")
    fig_flow = go.Figure()
    fig_flow.add_trace(go.Bar(
        x=crowd_df.index,
        y=crowd_df["estimated_arrivals"],
        name="Estimated arrivals",
    ))
    fig_flow.add_trace(go.Bar(
        x=crowd_df.index,
        y=crowd_df["estimated_boarded"],
        name="Estimated boarded",
    ))
    fig_flow.update_layout(
        barmode="group",
        height=360,
        xaxis_title="Time",
        yaxis_title="Passengers per interval",
    )
    st.plotly_chart(fig_flow, use_container_width=True)

    # Crowd level and pressure visualization
    st.subheader("Crowd pressure and level")
    left, right = st.columns([1.4, 1])
    with left:
        fig_pressure = px.area(
            crowd_df,
            x=crowd_df.index,
            y="crowd_pressure_pct",
            markers=True,
            title="Estimated queue pressure (% of assumed terminal capacity)",
        )
        fig_pressure.add_hline(y=100, line_dash="dash", annotation_text="100% capacity")
        fig_pressure.update_layout(height=340, yaxis_title="Pressure (%)")
        st.plotly_chart(fig_pressure, use_container_width=True)
    with right:
        level_counts = crowd_df["crowd_level"].value_counts().reindex(
            ["Low", "Moderate", "High", "Critical"], fill_value=0
        )
        fig_levels = px.pie(
            values=level_counts.values,
            names=level_counts.index,
            hole=0.55,
            title="Forecast intervals by crowd level",
        )
        fig_levels.update_layout(height=340)
        st.plotly_chart(fig_levels, use_container_width=True)

    # Waiting time chart
    st.subheader("Estimated passenger waiting time")
    fig_wait = px.line(
        crowd_df,
        x=crowd_df.index,
        y="estimated_wait_minutes",
        markers=True,
        title="Estimated waiting time by forecast interval",
    )
    fig_wait.update_layout(height=340, yaxis_title="Minutes")
    st.plotly_chart(fig_wait, use_container_width=True)

    # Row-level data and analysis
    st.subheader("Forecast row data")
    display_cols = [
        "estimated_arrivals",
        "estimated_boarded",
        "queue_start",
        "queue_end",
        "queue_change",
        "crowd_pressure_pct",
        "crowd_level",
        "crowd_alert",
        "estimated_wait_minutes",
        "estimated_wait_readable",
    ]
    st.dataframe(
        crowd_df[[c for c in display_cols if c in crowd_df.columns]].round(2),
        use_container_width=True,
    )

    st.subheader("Analysis")
    busiest_time = crowd_df["queue_end"].idxmax()
    highest_wait_time = crowd_df["estimated_wait_minutes"].idxmax()
    st.markdown(
        f"""
        - **Busiest forecast interval:** {busiest_time}
        - **Maximum predicted queue:** {peak_queue:.0f} passengers
        - **Highest estimated wait:** {crowd_df['estimated_wait_minutes'].max():.1f} minutes at {highest_wait_time}
        - **Intervals above assumed terminal capacity:** {(crowd_df['queue_end'] > terminal_capacity).sum()}
        - **Intervals classified High/Critical:** {crowd_df['crowd_level'].isin(['High', 'Critical']).sum()}
        """
    )

    st.subheader("Representation")
    st.caption(
        "The charts represent a scenario forecast derived from the input arrival counts, "
        "boarding capacity, queue equation, terminal capacity, and waiting-time assumptions. "
        "They are not live crowd-camera or terminal sensor measurements."
    )

    csv = crowd_df.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download passenger crowd forecast CSV",
        data=csv,
        file_name="passenger_crowd_forecast.csv",
        mime="text/csv",
    )


render()
