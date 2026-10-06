"""
Per-robot Clock Offset Estimation and Timestamp Synchronization.

Simulates and corrects hardware clock skew and drift across robots.
Uses an asymmetric transit filter (similar to Christian's algorithm / NTP)
to estimate clock offset theta_i between robot hardware clock and server clock.
"""

from typing import Dict, List, Tuple
from collections import deque
import numpy as np


class ClockSyncTracker:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        # Per robot history of (t_robot, t_ingest)
        self.samples: Dict[str, deque] = {}
        # Current estimated offset for each robot: t_server ~= t_robot + offset
        self.offsets: Dict[str, float] = {}

    def record_packet(self, robot_id: str, t_robot: float, t_ingest: float) -> float:
        """
        Records a packet arrival and updates the running clock offset estimate.
        Returns the corrected timestamp in server time.
        """
        if robot_id not in self.samples:
            self.samples[robot_id] = deque(maxlen=self.window_size)
            self.offsets[robot_id] = t_ingest - t_robot

        # Raw difference = t_ingest - t_robot = offset + network_delay
        # Because network_delay >= min_delay >= 0, the minimum difference in the window
        # provides the tightest upper bound on offset + min_delay.
        raw_diff = t_ingest - t_robot
        self.samples[robot_id].append(raw_diff)

        # Minimum filter over sliding window
        min_diff = min(self.samples[robot_id])
        # Exponential moving average smoothing of the offset estimate
        alpha = 0.1
        self.offsets[robot_id] = (1.0 - alpha) * self.offsets[robot_id] + alpha * min_diff

        # Corrected timestamp: t_corrected = t_robot + estimated_offset
        t_corrected = t_robot + self.offsets[robot_id]
        return t_corrected

    def get_estimated_offset(self, robot_id: str) -> float:
        return self.offsets.get(robot_id, 0.0)
