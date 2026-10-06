"""
Heuristic Scheduling Baselines for Robotics Collection:
- FIFO (First-In, First-Out feasible dispatch)
- Round Robin (Fair cyclic allocation across robots)
- Shortest Job First (SJF - minimum collection duration)
- Highest Value First (HVF - greedy priority on value and error events)
"""

from typing import Dict, Any, Optional
import numpy as np
from robotics_platform.scheduler.env import RoboticsDataEnv


class BaseRoboticsScheduler:
    def __init__(self, name: str):
        self.name = name

    def pick_action(self, env: RoboticsDataEnv, obs: Dict[str, np.ndarray]) -> int:
        raise NotImplementedError


class RoboticsFIFO(BaseRoboticsScheduler):
    """FIFO: Dispatches the first feasible job at head of queue window."""

    def __init__(self):
        super().__init__("FIFO")

    def pick_action(self, env: RoboticsDataEnv, obs: Dict[str, np.ndarray]) -> int:
        mask = obs["action_mask"]
        k = env.queue_window_size
        for idx in range(k):
            if mask[idx] == 1:
                return idx
        return k  # DEFER


class RoboticsRoundRobin(BaseRoboticsScheduler):
    """Round Robin: Cycles through robot IDs to ensure fair collection distribution."""

    def __init__(self):
        super().__init__("Round Robin")
        self.last_robot_idx = -1

    def pick_action(self, env: RoboticsDataEnv, obs: Dict[str, np.ndarray]) -> int:
        mask = obs["action_mask"]
        k = env.queue_window_size
        candidates = env.get_candidate_jobs()

        # Try to find next robot index in circular order
        num_r = env.num_robots
        for offset in range(1, num_r + 1):
            target_idx = (self.last_robot_idx + offset) % num_r
            target_id = f"robot_{target_idx:02d}"

            for idx, job in enumerate(candidates):
                if mask[idx] == 1 and job.target_robot_id == target_id:
                    self.last_robot_idx = target_idx
                    return idx

        # Fallback to any valid action
        for idx in range(k):
            if mask[idx] == 1:
                return idx
        return k


class RoboticsSJF(BaseRoboticsScheduler):
    """Shortest Job First: Dispatches feasible job with shortest duration."""

    def __init__(self):
        super().__init__("Shortest Job First (SJF)")

    def pick_action(self, env: RoboticsDataEnv, obs: Dict[str, np.ndarray]) -> int:
        mask = obs["action_mask"]
        k = env.queue_window_size
        candidates = env.get_candidate_jobs()

        best_idx = None
        min_dur = float("inf")

        for idx, job in enumerate(candidates):
            if mask[idx] == 1 and job.duration_sec < min_dur:
                min_dur = job.duration_sec
                best_idx = idx

        return best_idx if best_idx is not None else k


class RoboticsHVF(BaseRoboticsScheduler):
    """Highest Value First: Greedy selection on job value score and error diagnostics."""

    def __init__(self):
        super().__init__("Highest Value First (HVF)")

    def pick_action(self, env: RoboticsDataEnv, obs: Dict[str, np.ndarray]) -> int:
        mask = obs["action_mask"]
        k = env.queue_window_size
        candidates = env.get_candidate_jobs()

        best_idx = None
        max_val = -float("inf")

        for idx, job in enumerate(candidates):
            if mask[idx] == 1:
                effective_val = job.value_score + (15.0 if job.is_error_diagnostic else 0.0)
                if effective_val > max_val:
                    max_val = effective_val
                    best_idx = idx

        return best_idx if best_idx is not None else k
