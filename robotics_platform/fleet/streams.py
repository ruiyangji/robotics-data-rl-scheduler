"""
Multi-rate telemetry stream generators for virtual robots.
Emits realistic robotics payloads across varying frequencies.
"""

from typing import Dict, Any, Optional
import numpy as np

from robotics_platform.fleet.robot import VirtualRobot, RobotStatus


class StreamGenerator:
    """
    Generates multi-rate telemetry packets:
    - joint_imu: 50 - 100 Hz
    - camera_meta: 10 - 30 Hz
    - status_heartbeat: 1 - 5 Hz
    - error_event: sporadic
    """

    @staticmethod
    def generate_joint_imu(robot: VirtualRobot, t_hw: float, rng: np.random.Generator) -> Dict[str, Any]:
        num_joints = 6
        # Generate smooth sinusoidal joint states plus Gaussian sensor noise
        base_freq = 0.5
        q = np.sin(base_freq * t_hw + np.arange(num_joints)) + rng.normal(0, 0.005, num_joints)
        qd = base_freq * np.cos(base_freq * t_hw + np.arange(num_joints)) + rng.normal(0, 0.01, num_joints)
        tau = 2.5 * np.sin(base_freq * t_hw + np.arange(num_joints)) + rng.normal(0, 0.05, num_joints)

        # 3-axis accelerometer and gyro
        accel = np.array([0.0, 0.0, 9.81]) + rng.normal(0, 0.08, 3)
        gyro = rng.normal(0, 0.02, 3)

        return {
            "joint_positions": [round(float(v), 4) for v in q],
            "joint_velocities": [round(float(v), 4) for v in qd],
            "joint_torques": [round(float(v), 4) for v in tau],
            "imu_accel": [round(float(v), 4) for v in accel],
            "imu_gyro": [round(float(v), 4) for v in gyro],
        }

    @staticmethod
    def generate_camera_meta(robot: VirtualRobot, t_hw: float, rng: np.random.Generator) -> Dict[str, Any]:
        frame_id = int(t_hw * 30)
        num_detections = int(rng.poisson(lam=2.5))
        return {
            "frame_id": frame_id,
            "exposure_ms": round(float(rng.uniform(4.0, 16.0)), 2),
            "resolution": [1920, 1080],
            "num_detected_objects": num_detections,
            "camera_status": "OK" if robot.sensor_health["camera"] > 0.7 else "DEGRADED",
            "frame_size_bytes": int(rng.normal(120000, 15000)),
        }

    @staticmethod
    def generate_status_heartbeat(robot: VirtualRobot, t_hw: float, rng: np.random.Generator) -> Dict[str, Any]:
        return {
            "battery_pct": round(robot.battery_pct, 2),
            "status": robot.status.value,
            "x": round(robot.x, 2),
            "y": round(robot.y, 2),
            "cpu_temp_c": round(float(rng.uniform(45.0, 68.0)), 1),
            "wifi_rssi_dbm": int(rng.uniform(-75, -45)),
            "sensor_health": {k: round(v, 2) for k, v in robot.sensor_health.items()},
        }

    @staticmethod
    def generate_error_event(robot: VirtualRobot, t_hw: float, rng: np.random.Generator) -> Optional[Dict[str, Any]]:
        # Sporadic event
        if rng.uniform(0, 1) < 0.005 or robot.status == RobotStatus.ERROR:
            severities = ["WARNING", "ERROR", "CRITICAL"]
            reasons = [
                "Joint 3 torque threshold exceeded",
                "IMU drift anomaly detected",
                "Camera frame drop / buffer stall",
                "Battery critical under-voltage",
                "Obstacle proximity safety stop",
            ]
            idx = int(rng.integers(0, len(reasons)))
            return {
                "error_code": 1000 + idx,
                "severity": severities[min(idx, len(severities) - 1)],
                "message": reasons[idx],
                "robot_status": robot.status.value,
            }
        return None
