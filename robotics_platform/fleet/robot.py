"""
Virtual Robot representation and state machine for simulated fleet.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple, Optional
import numpy as np


class RobotStatus(str, Enum):
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"
    TRANSIT = "TRANSIT"
    ERROR = "ERROR"


@dataclass
class VirtualRobot:
    robot_id: str
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    battery_pct: float = 100.0
    status: RobotStatus = RobotStatus.IDLE
    sensor_health: Dict[str, float] = field(
        default_factory=lambda: {
            "joint_encoder": 1.0,
            "imu": 1.0,
            "camera": 1.0,
            "depth": 1.0,
        }
    )
    # Clock parameters for hardware skew simulation
    clock_offset: float = 0.0   # Initial offset in seconds (e.g. +0.120s)
    clock_drift_rate: float = 0.0  # Drift in seconds per second (e.g. 50 ppm = 5e-5)
    sequence_counters: Dict[str, int] = field(default_factory=dict)

    def get_hardware_time(self, global_time: float) -> float:
        """Returns the simulated local hardware clock time given the global ground-truth time."""
        return global_time + self.clock_offset + (self.clock_drift_rate * global_time)

    def get_next_seq(self, stream_type: str) -> int:
        cur = self.sequence_counters.get(stream_type, 0)
        self.sequence_counters[stream_type] = cur + 1
        return cur

    def update(self, dt: float, rng: np.random.Generator):
        """Simulates physical movement, battery drain, and random sensor anomalies."""
        # Simple random walk kinematics
        if self.status in [RobotStatus.TRANSIT, RobotStatus.COLLECTING]:
            self.x += self.vx * dt
            self.y += self.vy * dt
            # Keep bounded in 100x100m workspace
            self.x = float(np.clip(self.x, -50.0, 50.0))
            self.y = float(np.clip(self.y, -50.0, 50.0))

            # Turn occasionally
            if rng.uniform(0, 1) < 0.05:
                speed = rng.uniform(0.5, 2.0)
                theta = rng.uniform(0, 2 * np.pi)
                self.vx = speed * np.cos(theta)
                self.vy = speed * np.sin(theta)

            # Battery drain (higher during active collection)
            drain = (0.015 if self.status == RobotStatus.COLLECTING else 0.005) * dt
            self.battery_pct = max(0.0, self.battery_pct - drain)
        else:
            # Idle slow drain
            self.battery_pct = max(0.0, self.battery_pct - 0.001 * dt)

        # Occasional transient sensor glitch
        for sensor in self.sensor_health:
            if rng.uniform(0, 1) < 0.0005:
                self.sensor_health[sensor] = max(0.0, self.sensor_health[sensor] - rng.uniform(0.1, 0.4))
            elif self.sensor_health[sensor] < 1.0:
                self.sensor_health[sensor] = min(1.0, self.sensor_health[sensor] + 0.01 * dt)

        # If battery dies or sensors critically fail, set error status
        if self.battery_pct <= 5.0 or min(self.sensor_health.values()) < 0.3:
            self.status = RobotStatus.ERROR
