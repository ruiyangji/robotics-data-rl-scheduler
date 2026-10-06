"""
Streamlit Interactive Dashboard for Robotics Data Platform & RL Scheduler.

Features:
1. Live Fleet View: 2D robot spatial positions, battery gauges, real-time bandwidth consumption.
2. Replay & Packet Inspector: Scrub through timeline, inspect stale packet drops and quarantine reasons.
3. Policy Benchmark: Comparative performance charts (PPO vs FIFO vs RR vs SJF vs HVF).
4. Chronos Failure Analysis: Automated scenario inspection explaining where and why PPO loses to baselines.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from robotics_platform.fleet.fleet_simulator import FleetSimulator
from robotics_platform.fleet.robot import RobotStatus
from robotics_platform.ingest.pipeline import IngestionPipeline
from robotics_platform.ingest.storage import TelemetryStorage
from robotics_platform.scheduler.env import RoboticsDataEnv
from robotics_platform.scheduler.policy import RoboticsAttentionPolicy
from robotics_platform.evaluation.runner import RoboticsEvaluationHarness
from robotics_platform.evaluation.failure_analysis import ChronosFailureAnalyzer


st.set_page_config(
    page_title="Robotics Telemetry RL Cluster Scheduler",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Robotics Data Platform RL Cluster Scheduler")
st.caption("Discrete-Event Telemetry Ingestion, Timestamp Sync, and Action-Masked PPO Dispatch")

# Sidebar Configuration
st.sidebar.header("Fleet & Pipeline Settings")
num_robots = st.sidebar.slider("Fleet Size (Robots)", min_value=8, max_value=32, value=16, step=4)
jitter_ms = st.sidebar.slider("Network Jitter Std (ms)", min_value=5.0, max_value=80.0, value=25.0)
stale_thresh_ms = st.sidebar.slider("Joint Staleness Threshold (ms)", min_value=100, max_value=1000, value=500)
bandwidth_cap = st.sidebar.slider("Fleet Bandwidth Cap (MB/s)", min_value=20, max_value=120, value=60)

tabs = st.tabs([
    "🛰️ Live Fleet & Telemetry",
    "🔍 Replay & Packet Inspector",
    "📊 Policy Evaluation Benchmark",
    "🔬 Chronos Failure Analysis",
])

# ----------------- TAB 1: LIVE FLEET & TELEMETRY -----------------
with tabs[0]:
    st.subheader("Simulated Fleet Telemetry Monitor")
    col_sim_btn, col_stats = st.columns([1, 3])

    if "fleet_sim" not in st.session_state:
        st.session_state.fleet_sim = FleetSimulator(num_robots=num_robots, seed=42)
        st.session_state.pipeline = IngestionPipeline()
        st.session_state.sim_time = 0.0

    if col_sim_btn.button("Run Simulation Step (+1.0s)", use_container_width=True):
        events = st.session_state.fleet_sim.simulate_step(
            current_time=st.session_state.sim_time, dt=1.0
        )
        st.session_state.pipeline.process_batch(events)
        st.session_state.sim_time += 1.0

    metrics = st.session_state.pipeline.get_metrics_summary()

    # Top KPI Metrics
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Packets Ingested", metrics.get("total_received", 0))
    kpi2.metric("Valid Packets", metrics.get("valid_count", 0))
    kpi3.metric("Stale Drop Rate", f"{metrics.get('drop_rate_pct', 0.0):.1f}%")
    kpi4.metric("p50 Ingest Latency", f"{metrics.get('latency_p50_ms', 0.0):.1f} ms")
    kpi5.metric("p95 Ingest Latency", f"{metrics.get('latency_p95_ms', 0.0):.1f} ms")

    # 2D Map & Robot State Table
    col_map, col_table = st.columns([1, 1])
    robots = st.session_state.fleet_sim.robots

    with col_map:
        st.markdown("**Robot Fleet Spatial Coordinates (100x100m Arena)**")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_facecolor("#1e1e1e")
        fig.patch.set_facecolor("#1e1e1e")
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.grid(True, color="#333333", linestyle="--", alpha=0.7)

        color_map = {
            RobotStatus.IDLE: "#00c853",
            RobotStatus.COLLECTING: "#2979ff",
            RobotStatus.TRANSIT: "#ffab00",
            RobotStatus.ERROR: "#d50000",
        }

        for r in robots:
            ax.scatter(r.x, r.y, color=color_map.get(r.status, "#ffffff"), s=120, edgecolors="white")
            ax.annotate(r.robot_id, (r.x + 1.2, r.y + 1.2), color="white", fontsize=8)

        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#444444")
        st.pyplot(fig)

    with col_table:
        st.markdown("**Fleet Health & Battery Telemetry**")
        robot_df = pd.DataFrame(
            [
                {
                    "Robot ID": r.robot_id,
                    "Status": r.status.value,
                    "Battery %": f"{r.battery_pct:.1f}%",
                    "Encoder Health": f"{r.sensor_health['joint_encoder']*100:.0f}%",
                    "IMU Health": f"{r.sensor_health['imu']*100:.0f}%",
                    "Camera": f"{r.sensor_health['camera']*100:.0f}%",
                    "Clock Skew (ms)": f"{r.clock_offset*1000:.1f}ms",
                }
                for r in robots
            ]
        )
        st.dataframe(robot_df, use_container_width=True, height=350)

# ----------------- TAB 2: REPLAY & PACKET INSPECTOR -----------------
with tabs[1]:
    st.subheader("Telemetry Replay & Stale Packet Quarantine Inspector")
    st.markdown(
        "> **Systems Muscle**: Inspect exactly *why* each packet was accepted, dropped, or quarantined "
        "(e.g. clock offset skew, late arrival past 500ms watermarked threshold, or out-of-order sequence)."
    )

    storage: TelemetryStorage = st.session_state.pipeline.storage
    recent_events = storage.query(limit=500)

    if recent_events:
        event_df = pd.DataFrame(recent_events)
        status_filter = st.selectbox(
            "Filter by Ingestion Decision Status:",
            ["ALL", "STALE_DROPPED", "OUT_OF_ORDER_QUARANTINED", "VALID"],
        )

        if status_filter != "ALL":
            filtered_df = event_df[event_df["status"] == status_filter]
        else:
            filtered_df = event_df

        st.caption(f"Displaying {len(filtered_df)} events:")
        cols_to_show = [
            "event_id",
            "robot_id",
            "stream_type",
            "sequence_number",
            "event_timestamp",
            "ingest_timestamp",
            "corrected_timestamp",
            "status",
            "drop_reason",
        ]
        st.dataframe(filtered_df[cols_to_show], use_container_width=True, height=400)
    else:
        st.info("No packets in storage yet. Run simulation steps in Tab 1 to populate events.")

# ----------------- TAB 3: POLICY BENCHMARK -----------------
with tabs[2]:
    st.subheader("Scheduling Policy Benchmark: Baselines vs Action-Masked PPO")
    st.markdown(
        "Replays identical multi-rate collection job traces across **FIFO, Round Robin, SJF, HVF, and PPO**."
    )

    if st.button("Run Multi-Scenario Evaluation Suite", type="primary"):
        with st.spinner("Executing replay scenarios across all scheduler policies..."):
            harness = RoboticsEvaluationHarness(
                num_robots=num_robots,
                max_bandwidth_mbps=bandwidth_cap,
                num_jobs_per_episode=35,
            )
            policy = RoboticsAttentionPolicy(robot_dim=5, job_dim=5, sys_dim=3, d_model=32, nhead=2)
            detailed_df, summary_df = harness.run_benchmark(ppo_policy=policy, num_scenarios=5)
            st.session_state.detailed_bench = detailed_df
            st.session_state.summary_bench = summary_df

    if "summary_bench" in st.session_state:
        sum_df = st.session_state.summary_bench
        st.dataframe(sum_df, use_container_width=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**Total Value Collected (Higher is Better)**")
            fig1, ax1 = plt.subplots(figsize=(6, 3.5))
            ax1.bar(sum_df["policy"], sum_df["total_value_collected"], color="#4caf50")
            ax1.set_ylabel("Value Score")
            plt.xticks(rotation=25)
            st.pyplot(fig1)

        with col_c2:
            st.markdown("**Deadline Miss Rate % (Lower is Better)**")
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            ax2.bar(sum_df["policy"], sum_df["deadline_miss_rate_pct"], color="#f44336")
            ax2.set_ylabel("Miss Rate %")
            plt.xticks(rotation=25)
            st.pyplot(fig2)

# ----------------- TAB 4: CHRONOS FAILURE ANALYSIS -----------------
with tabs[3]:
    st.subheader("The Chronos Habit: System Failure Analysis")
    st.markdown(
        """
        Jerry's Meta Chronos evaluation pipeline backtested 2,000+ capacity proposals.
        Here, we apply that same discipline: **systematically reporting scenarios where PPO lost to baselines and explaining the mechanical root cause.**
        """
    )

    if "detailed_bench" in st.session_state:
        analyzer = ChronosFailureAnalyzer(st.session_state.detailed_bench)
        analysis = analyzer.analyze_failures()
        st.markdown(analysis["report_markdown"])
    else:
        st.info("Run the evaluation benchmark in Tab 3 to generate Chronos failure reports.")
