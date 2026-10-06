"""
Robotics collection jobs and workload generation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import numpy as np


class CollectionTaskType(str, Enum):
    PICK_AND_PLACE_TRIAL = "pick_and_place_trial"
    HIGH_RATE_JOINT_STREAM = "high_rate_joint_stream"
    CAMERA_BURST_UPLOAD = "camera_burst_upload"
    ERROR_DIAGNOSTIC_DUMP = "error_diagnostic_dump"
    FLEET_HEALTH_AUDIT = "fleet_health_audit"


@dataclass
class CollectionJob:
    job_id: str
    task_type: CollectionTaskType
    target_robot_id: str
    bandwidth_mbps: float      # Bandwidth cost during collection (MB/s)
    duration_sec: float        # Duration to execute collection (s)
    value_score: float         # Priority/value: freshness, novelty, critical diagnostic (1.0 to 10.0)
    arrival_time: float        # Arrival in queue
    deadline: float            # Expiration timestamp
    is_error_diagnostic: bool = False
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    remaining_time: float = 0.0

    def __post_init__(self):
        self.remaining_time = self.duration_sec

    @property
    def is_completed(self) -> bool:
        return self.remaining_time <= 1e-6 and self.start_time is not None

    @property
    def wait_time(self) -> float:
        if self.start_time is None:
            return 0.0
        return self.start_time - self.arrival_time

    @property
    def missed_deadline(self) -> bool:
        if self.completion_time is not None:
            return self.completion_time > self.deadline
        return False


class RoboticsWorkloadGenerator:
    """
    Generates realistic collection job requests matching robotics data platform operations.
    """

    def __init__(self, num_robots: int = 16, arrival_rate: float = 1.2, seed: Optional[int] = 42):
        self.num_robots = num_robots
        self.arrival_rate = arrival_rate
        self.rng = np.random.default_rng(seed)

    def generate_jobs(self, num_jobs: int = 50) -> List[CollectionJob]:
        jobs = []
        current_time = 0.0

        task_profiles = [
            # type, bw (MB/s), duration (s), value, is_error_prob
            (CollectionTaskType.PICK_AND_PLACE_TRIAL, (8.0, 15.0), (5.0, 15.0), (6.0, 9.0), 0.0),
            (CollectionTaskType.HIGH_RATE_JOINT_STREAM, (4.0, 8.0), (10.0, 30.0), (4.0, 7.0), 0.0),
            (CollectionTaskType.CAMERA_BURST_UPLOAD, (15.0, 30.0), (8.0, 20.0), (3.0, 6.0), 0.0),
            (CollectionTaskType.ERROR_DIAGNOSTIC_DUMP, (2.0, 6.0), (3.0, 8.0), (8.5, 10.0), 1.0),
            (CollectionTaskType.FLEET_HEALTH_AUDIT, (1.0, 3.0), (4.0, 10.0), (1.0, 3.5), 0.0),
        ]
        probs = [0.25, 0.30, 0.20, 0.10, 0.15]

        for i in range(num_jobs):
            inter_arrival = self.rng.exponential(1.0 / self.arrival_rate)
            current_time += inter_arrival

            profile_idx = int(self.rng.choice(len(task_profiles), p=probs))
            t_type, bw_range, dur_range, val_range, is_err_flag = task_profiles[profile_idx]

            bw = float(self.rng.uniform(bw_range[0], bw_range[1]))
            dur = float(self.rng.uniform(dur_range[0], dur_range[1]))
            val = float(self.rng.uniform(val_range[0], val_range[1]))
            robot_idx = int(self.rng.integers(0, self.num_robots))
            target_robot = f"robot_{robot_idx:02d}"

            # Deadline slack: 1.5x to 3.0x duration
            slack = dur * float(self.rng.uniform(1.5, 3.0))
            deadline = current_time + slack

            jobs.append(
                CollectionJob(
                    job_id=f"job_{i:03d}_{t_type.value}",
                    task_type=t_type,
                    target_robot_id=target_robot,
                    bandwidth_mbps=round(bw, 2),
                    duration_sec=round(dur, 2),
                    value_score=round(val, 2),
                    arrival_time=round(current_time, 2),
                    deadline=round(deadline, 2),
                    is_error_diagnostic=bool(is_err_flag > 0.5),
                )
            )

        return jobs
