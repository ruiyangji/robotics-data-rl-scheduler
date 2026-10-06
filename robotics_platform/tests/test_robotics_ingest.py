"""
Unit and Integration Tests for Robotics Data Platform Ingestion & Sync (Milestone 2).
"""

import os
import tempfile
import pytest
import numpy as np

from robotics_platform.fleet.robot import VirtualRobot, RobotStatus
from robotics_platform.fleet.streams import StreamGenerator
from robotics_platform.fleet.faults import FaultInjector, NetworkFaultConfig
from robotics_platform.fleet.fleet_simulator import FleetSimulator
from robotics_platform.sync.clock_sync import ClockSyncTracker
from robotics_platform.sync.watermarking import WatermarkTracker
from robotics_platform.sync.staleness import StalePacketPolicy
from robotics_platform.sync.quality_metrics import DataQualityTracker
from robotics_platform.ingest.schema import TelemetryEvent
from robotics_platform.ingest.storage import TelemetryStorage
from robotics_platform.ingest.pipeline import IngestionPipeline


def test_robot_and_streams():
    robot = VirtualRobot(robot_id="robot_01", clock_offset=0.15)
    rng = np.random.default_rng(42)

    # Hardware time includes offset
    t_hw = robot.get_hardware_time(1.0)
    assert abs(t_hw - 1.15) < 1e-4

    # Joint IMU stream
    joint = StreamGenerator.generate_joint_imu(robot, t_hw, rng)
    assert len(joint["joint_positions"]) == 6
    assert len(joint["imu_accel"]) == 3

    # Camera metadata
    cam = StreamGenerator.generate_camera_meta(robot, t_hw, rng)
    assert "frame_id" in cam
    assert cam["resolution"] == [1920, 1080]

    # Heartbeat
    hb = StreamGenerator.generate_status_heartbeat(robot, t_hw, rng)
    assert hb["battery_pct"] == 100.0


def test_clock_offset_correction():
    tracker = ClockSyncTracker(window_size=20)
    # Simulate a robot with true offset +0.100s (robot clock is 100ms fast)
    # server_time = robot_time - 0.100 + network_delay
    true_offset = -0.100

    for i in range(30):
        t_robot = 10.0 + i * 0.05
        net_delay = 0.015 + np.random.uniform(0.0, 0.010)
        t_server = t_robot + true_offset + net_delay
        t_corr = tracker.record_packet("robot_01", t_robot, t_server)

    est_offset = tracker.get_estimated_offset("robot_01")
    # Estimated offset should be close to true_offset + min_delay (approx -0.085s)
    assert -0.110 <= est_offset <= -0.080


def test_watermark_tracking():
    wm_tracker = WatermarkTracker(default_slack_seconds=0.10)
    # When packet arrives at t=1.00
    wm = wm_tracker.update("robot_01", "joint_imu", 1.00)
    # joint_imu slack is 0.08, so watermark should be 1.00 - 0.08 = 0.92
    assert abs(wm - 0.92) < 1e-4
    assert wm_tracker.is_past_watermark("robot_01", "joint_imu", 0.80) is True
    assert wm_tracker.is_past_watermark("robot_01", "joint_imu", 0.95) is False


def test_stale_packet_policy():
    policy = StalePacketPolicy()

    # 1. Fresh packet (age 20ms < 500ms)
    event_fresh = TelemetryEvent(
        event_id="e1",
        robot_id="r1",
        stream_type="joint_imu",
        event_timestamp=1.0,
        ingest_timestamp=1.020,
        sequence_number=1,
    )
    status, reason = policy.evaluate_packet(event_fresh, watermark=0.9)
    assert status == "VALID"
    assert reason is None

    # 2. Stale packet (age 600ms > 500ms threshold)
    event_stale = TelemetryEvent(
        event_id="e2",
        robot_id="r1",
        stream_type="joint_imu",
        event_timestamp=1.0,
        ingest_timestamp=1.650,
        sequence_number=2,
    )
    status_stale, reason_stale = policy.evaluate_packet(event_stale, watermark=1.5)
    assert status_stale == "STALE_DROPPED"
    assert "exceeds 500.0ms threshold" in reason_stale

    # 3. Out of order packet (seq 1 arrives after seq 3)
    event_newer = TelemetryEvent(
        event_id="e3",
        robot_id="r1",
        stream_type="joint_imu",
        event_timestamp=2.0,
        ingest_timestamp=2.05,
        sequence_number=10,
    )
    policy.evaluate_packet(event_newer, watermark=1.9)

    event_out_of_order = TelemetryEvent(
        event_id="e4",
        robot_id="r1",
        stream_type="joint_imu",
        event_timestamp=2.01,
        ingest_timestamp=2.06,
        sequence_number=5,  # 5 < 10!
    )
    status_ooo, reason_ooo = policy.evaluate_packet(event_out_of_order, watermark=1.9)
    assert status_ooo == "OUT_OF_ORDER_QUARANTINED"
    assert "Out-of-order" in reason_ooo


def test_storage_and_parquet_export():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "telemetry.db")
        parquet_file = os.path.join(tmpdir, "telemetry.parquet")

        store = TelemetryStorage(db_file)
        events = [
            TelemetryEvent(
                event_id=f"evt_{i}",
                robot_id="r1",
                stream_type="joint_imu",
                event_timestamp=1.0 + i * 0.02,
                ingest_timestamp=1.03 + i * 0.02,
                sequence_number=i,
                payload={"joint": i},
                status="VALID",
            )
            for i in range(10)
        ]
        store.insert_batch(events)
        assert store.count() == 10

        queried = store.query(robot_id="r1")
        assert len(queried) == 10

        # Export to Parquet
        num_exported = store.export_parquet(parquet_file)
        assert num_exported == 10
        assert os.path.exists(parquet_file)
        store.close()


def test_end_to_end_fleet_and_pipeline():
    fleet = FleetSimulator(num_robots=8, seed=42)
    pipeline = IngestionPipeline()

    current_t = 0.0
    for step in range(10):
        events = fleet.simulate_step(current_time=current_t, dt=0.05)
        pipeline.process_batch(events)
        current_t += 0.05

    summary = pipeline.get_metrics_summary()
    assert summary["total_received"] > 0
    assert summary["valid_count"] > 0
    assert "latency_p50_ms" in summary
    assert "latency_p95_ms" in summary
