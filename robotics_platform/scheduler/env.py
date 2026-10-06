"""
Gymnasium Environment for Robotics Data Collection Scheduling with Action Masking.

Models fleet bandwidth constraints, battery states, task deadlines,
and prioritized telemetry dispatches.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from robotics_platform.fleet.robot import VirtualRobot, RobotStatus
from robotics_platform.fleet.fleet_simulator import FleetSimulator
from robotics_platform.scheduler.jobs import (
    CollectionJob,
    RoboticsWorkloadGenerator,
    CollectionTaskType,
)


class RoboticsDataEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_robots: int = 16,
        max_fleet_bandwidth_mbps: float = 60.0,
        queue_window_size: int = 8,
        num_jobs_per_episode: int = 40,
        max_steps: int = 300,
        seed: Optional[int] = 42,
    ):
        super().__init__()
        self.num_robots = num_robots
        self.max_bandwidth = max_fleet_bandwidth_mbps
        self.queue_window_size = queue_window_size
        self.num_jobs_per_episode = num_jobs_per_episode
        self.max_steps = max_steps
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)

        self.workload_gen = RoboticsWorkloadGenerator(
            num_robots=self.num_robots, seed=seed
        )

        # Action space: 0..K-1 selects job in candidate window to dispatch, K is DEFER
        self.action_space = spaces.Discrete(self.queue_window_size + 1)

        # Observation space
        self.robot_feat_dim = 5
        self.job_feat_dim = 5
        self.sys_feat_dim = 3

        self.observation_space = spaces.Dict(
            {
                "robots": spaces.Box(
                    low=0.0, high=10.0, shape=(self.num_robots, self.robot_feat_dim), dtype=np.float32
                ),
                "queue": spaces.Box(
                    low=0.0, high=10.0, shape=(self.queue_window_size, self.job_feat_dim), dtype=np.float32
                ),
                "system": spaces.Box(
                    low=0.0, high=10.0, shape=(self.sys_feat_dim,), dtype=np.float32
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(self.queue_window_size + 1,), dtype=np.int8
                ),
            }
        )

        self.robots: Dict[str, VirtualRobot] = {}
        self.active_jobs: List[CollectionJob] = []
        self.unborn_jobs: List[CollectionJob] = []
        self.pending_jobs: List[CollectionJob] = []
        self.completed_jobs: List[CollectionJob] = []
        self.dropped_jobs: List[CollectionJob] = []

        self.current_time = 0.0
        self.step_count = 0
        self.current_bandwidth_used = 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.workload_gen = RoboticsWorkloadGenerator(
                num_robots=self.num_robots, seed=seed
            )

        # Init robots
        self.robots = {}
        for i in range(self.num_robots):
            rid = f"robot_{i:02d}"
            self.robots[rid] = VirtualRobot(
                robot_id=rid,
                x=float(self.rng.uniform(-20, 20)),
                y=float(self.rng.uniform(-20, 20)),
                battery_pct=float(self.rng.uniform(75, 100)),
                status=RobotStatus.IDLE,
            )

        self.unborn_jobs = self.workload_gen.generate_jobs(self.num_jobs_per_episode)
        self.pending_jobs = []
        self.active_jobs = []
        self.completed_jobs = []
        self.dropped_jobs = []

        self.current_time = 0.0
        self.step_count = 0
        self.current_bandwidth_used = 0.0

        self._admit_arriving_jobs()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _admit_arriving_jobs(self):
        while self.unborn_jobs and self.unborn_jobs[0].arrival_time <= self.current_time + 1e-6:
            job = self.unborn_jobs.pop(0)
            self.pending_jobs.append(job)

    def get_candidate_jobs(self) -> List[CollectionJob]:
        """Returns the top K candidate jobs currently available in the pending queue."""
        return self.pending_jobs[: self.queue_window_size]

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.queue_window_size + 1, dtype=np.int8)
        candidates = self.get_candidate_jobs()

        for idx, job in enumerate(candidates):
            robot = self.robots.get(job.target_robot_id)
            if robot is None:
                continue

            # Feasibility conditions:
            # 1. Target robot is idle and battery >= 15%
            robot_avail = (robot.status != RobotStatus.COLLECTING) and (robot.battery_pct >= 15.0)
            # 2. Bandwidth capacity available
            bw_avail = (self.current_bandwidth_used + job.bandwidth_mbps) <= (self.max_bandwidth + 1e-6)
            # 3. Not already past deadline
            not_expired = self.current_time < job.deadline

            if robot_avail and bw_avail and not_expired:
                mask[idx] = 1

        # DEFER action is always valid
        mask[self.queue_window_size] = 1
        return mask

    def _get_obs(self) -> Dict[str, np.ndarray]:
        # 1. Robot features (N, 5)
        robot_feats = np.zeros((self.num_robots, self.robot_feat_dim), dtype=np.float32)
        for i in range(self.num_robots):
            rid = f"robot_{i:02d}"
            r = self.robots[rid]
            is_busy = 1.0 if r.status == RobotStatus.COLLECTING else 0.0
            avg_health = float(np.mean(list(r.sensor_health.values())))
            is_error = 1.0 if r.status == RobotStatus.ERROR else 0.0
            robot_feats[i] = [
                r.battery_pct / 100.0,
                is_busy,
                avg_health,
                min(r.x / 50.0, 1.0),
                is_error,
            ]

        # 2. Queue features (K, 5)
        queue_feats = np.zeros((self.queue_window_size, self.job_feat_dim), dtype=np.float32)
        candidates = self.get_candidate_jobs()
        for idx, job in enumerate(candidates):
            slack = max(0.0, job.deadline - self.current_time)
            queue_feats[idx] = [
                job.bandwidth_mbps / 30.0,
                job.duration_sec / 30.0,
                job.value_score / 10.0,
                min(5.0, slack / 30.0),
                1.0 if job.is_error_diagnostic else 0.0,
            ]

        # 3. System features (3,)
        sys_feats = np.array(
            [
                self.current_bandwidth_used / max(1.0, self.max_bandwidth),
                min(5.0, len(self.pending_jobs) / 20.0),
                min(10.0, self.current_time / 100.0),
            ],
            dtype=np.float32,
        )

        mask = self.get_action_mask()
        return {
            "robots": robot_feats,
            "queue": queue_feats,
            "system": sys_feats,
            "action_mask": mask,
        }

    def _advance_time_to_next_event(self):
        """Discrete-event advance strictly forward in time to earliest completion or arrival."""
        candidate_times = []
        for job in self.active_jobs:
            comp_t = (job.start_time if job.start_time is not None else self.current_time) + job.duration_sec
            if comp_t > self.current_time + 1e-5:
                candidate_times.append(comp_t)

        for job in self.unborn_jobs:
            if job.arrival_time > self.current_time + 1e-5:
                candidate_times.append(job.arrival_time)
                break

        if not candidate_times:
            # No future completion and no future arrival
            if self.pending_jobs:
                # Expire any unserviceable pending jobs
                self.dropped_jobs.extend(self.pending_jobs)
                self.pending_jobs = []
            return

        event_time = min(candidate_times)
        dt = max(0.01, event_time - self.current_time)
        self.current_time = event_time

        # Update robots & active jobs
        still_active = []
        for job in self.active_jobs:
            if (job.start_time or 0.0) + job.duration_sec <= self.current_time + 1e-5:
                job.remaining_time = 0.0
                job.completion_time = self.current_time
                self.completed_jobs.append(job)
                r = self.robots.get(job.target_robot_id)
                if r:
                    r.status = RobotStatus.IDLE
                self.current_bandwidth_used = max(0.0, self.current_bandwidth_used - job.bandwidth_mbps)
            else:
                job.remaining_time = max(0.0, (job.start_time or 0.0) + job.duration_sec - self.current_time)
                still_active.append(job)
        self.active_jobs = still_active

        # Update battery and motion for robots
        for r in self.robots.values():
            r.update(dt, self.rng)

        # Check for expired pending jobs
        surviving_pending = []
        for job in self.pending_jobs:
            if self.current_time > job.deadline:
                self.dropped_jobs.append(job)
            else:
                surviving_pending.append(job)
        self.pending_jobs = surviving_pending

        self._admit_arriving_jobs()

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        reward = 0.0
        mask = self.get_action_mask()
        candidates = self.get_candidate_jobs()

        if not mask[action]:
            reward = -2.0  # Invalid action penalty
            self._advance_time_to_next_event()
        elif action < len(candidates):
            # Dispatch candidate job at index `action`
            job = candidates[action]
            target_robot = self.robots[job.target_robot_id]

            # Allocate
            target_robot.status = RobotStatus.COLLECTING
            job.start_time = self.current_time
            job.remaining_time = job.duration_sec
            self.current_bandwidth_used += job.bandwidth_mbps
            self.active_jobs.append(job)
            self.pending_jobs.remove(job)

            # Reward calculation:
            bw_ratio = self.current_bandwidth_used / max(1.0, self.max_bandwidth)
            reward = (
                (job.value_score * 0.4)
                + (2.0 if job.is_error_diagnostic else 0.0)
                + (1.0 if bw_ratio >= 0.6 else 0.2)
                - 0.02 * max(0.0, self.current_time - job.arrival_time)
            )

            # If no more pending jobs can fit or queue window is empty, advance time
            new_mask = self.get_action_mask()
            if not any(new_mask[: self.queue_window_size]):
                self._advance_time_to_next_event()
        else:
            # Action == queue_window_size (DEFER)
            if self.pending_jobs:
                reward -= 0.05 * len(self.pending_jobs)
            self._advance_time_to_next_event()

        all_done = (
            len(self.unborn_jobs) == 0
            and len(self.pending_jobs) == 0
            and len(self.active_jobs) == 0
        )
        truncated = self.step_count >= self.max_steps
        terminated = all_done

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        total_value = sum(j.value_score for j in self.completed_jobs)
        missed = [j for j in self.completed_jobs if j.missed_deadline] + self.dropped_jobs
        total_tasks = len(self.completed_jobs) + len(self.dropped_jobs)
        deadline_miss_rate = (len(missed) / max(1, total_tasks)) * 100.0
        completion_rate = (len(self.completed_jobs) / max(1, total_tasks)) * 100.0

        error_handled = sum(1 for j in self.completed_jobs if j.is_error_diagnostic)
        error_total = sum(
            1 for j in (self.completed_jobs + self.dropped_jobs + self.pending_jobs + self.active_jobs)
            if j.is_error_diagnostic
        )
        error_coverage = (error_handled / max(1, error_total)) * 100.0

        return {
            "current_time": self.current_time,
            "completed_jobs": len(self.completed_jobs),
            "dropped_jobs": len(self.dropped_jobs),
            "completion_rate_pct": completion_rate,
            "deadline_miss_rate_pct": deadline_miss_rate,
            "total_value_collected": total_value,
            "error_coverage_pct": error_coverage,
            "bandwidth_used_mbps": self.current_bandwidth_used,
            "bandwidth_util_pct": (self.current_bandwidth_used / self.max_bandwidth) * 100.0,
        }
