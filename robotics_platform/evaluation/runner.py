"""
Scenario Runner for Robotics Telemetry Scheduling.
Replays identical trace workloads across FIFO, Round Robin, SJF, HVF, and PPO.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import torch

from robotics_platform.scheduler.env import RoboticsDataEnv
from robotics_platform.scheduler.baselines import (
    RoboticsFIFO,
    RoboticsRoundRobin,
    RoboticsSJF,
    RoboticsHVF,
)
from robotics_platform.scheduler.policy import RoboticsAttentionPolicy
from robotics_platform.evaluation.metrics import compute_scenario_metrics


class RoboticsEvaluationHarness:
    def __init__(
        self,
        num_robots: int = 16,
        max_bandwidth_mbps: float = 60.0,
        num_jobs_per_episode: int = 35,
    ):
        self.num_robots = num_robots
        self.max_bandwidth = max_bandwidth_mbps
        self.num_jobs = num_jobs_per_episode
        self.baselines = [
            RoboticsFIFO(),
            RoboticsRoundRobin(),
            RoboticsSJF(),
            RoboticsHVF(),
        ]

    def run_scenario_heuristic(
        self,
        scheduler,
        scenario_id: int,
        seed: int,
    ) -> Dict[str, Any]:
        env = RoboticsDataEnv(
            num_robots=self.num_robots,
            max_fleet_bandwidth_mbps=self.max_bandwidth,
            num_jobs_per_episode=self.num_jobs,
            seed=seed,
        )
        obs, info = env.reset(seed=seed)
        total_reward = 0.0
        done = False

        while not done:
            act = scheduler.pick_action(env, obs)
            obs, r, term, trunc, info = env.step(act)
            total_reward += r
            done = term or trunc

        return compute_scenario_metrics(info, scheduler.name, scenario_id, total_reward)

    def run_scenario_ppo(
        self,
        policy: RoboticsAttentionPolicy,
        scenario_id: int,
        seed: int,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        env = RoboticsDataEnv(
            num_robots=self.num_robots,
            max_fleet_bandwidth_mbps=self.max_bandwidth,
            num_jobs_per_episode=self.num_jobs,
            seed=seed,
        )
        obs, info = env.reset(seed=seed)
        total_reward = 0.0
        done = False
        policy.eval()

        with torch.no_grad():
            while not done:
                rob_t = torch.tensor(obs["robots"], dtype=torch.float32, device=device).unsqueeze(0)
                q_t = torch.tensor(obs["queue"], dtype=torch.float32, device=device).unsqueeze(0)
                sys_t = torch.tensor(obs["system"], dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.tensor(obs["action_mask"], dtype=torch.int8, device=device).unsqueeze(0)

                logits, _ = policy(rob_t, q_t, sys_t, mask_t)
                act = int(torch.argmax(logits, dim=-1).item())

                obs, r, term, trunc, info = env.step(act)
                total_reward += r
                done = term or trunc

        return compute_scenario_metrics(info, "PPO Policy", scenario_id, total_reward)

    def run_benchmark(
        self,
        ppo_policy: Optional[RoboticsAttentionPolicy] = None,
        num_scenarios: int = 10,
        start_seed: int = 200,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Replays num_scenarios across all policies.
        Returns:
            detailed_df: Scenario-by-scenario metrics
            summary_df: Aggregated mean/std summary table
        """
        all_records = []

        for s_idx in range(num_scenarios):
            seed = start_seed + s_idx
            # Run all baselines on this scenario
            for b in self.baselines:
                rec = self.run_scenario_heuristic(b, scenario_id=s_idx, seed=seed)
                all_records.append(rec)

            # Run PPO on the exact same scenario
            if ppo_policy is not None:
                rl_rec = self.run_scenario_ppo(ppo_policy, scenario_id=s_idx, seed=seed)
                all_records.append(rl_rec)

        df = pd.DataFrame(all_records)

        summary_df = (
            df.groupby("policy")
            .agg(
                total_value_collected=("total_value_collected", "mean"),
                completion_rate_pct=("completion_rate_pct", "mean"),
                deadline_miss_rate_pct=("deadline_miss_rate_pct", "mean"),
                error_coverage_pct=("error_coverage_pct", "mean"),
                total_reward=("total_reward", "mean"),
                makespan_sec=("makespan_sec", "mean"),
            )
            .reset_index()
            .sort_values(by="total_value_collected", ascending=False)
        )

        return df, summary_df
