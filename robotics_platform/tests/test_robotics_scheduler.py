"""
Unit and Integration Tests for Robotics Data Platform Scheduling & Evaluation (Milestone 3).
"""

import pytest
import numpy as np
import torch
import pandas as pd

from robotics_platform.scheduler.jobs import (
    CollectionJob,
    CollectionTaskType,
    RoboticsWorkloadGenerator,
)
from robotics_platform.scheduler.env import RoboticsDataEnv
from robotics_platform.scheduler.baselines import (
    RoboticsFIFO,
    RoboticsRoundRobin,
    RoboticsSJF,
    RoboticsHVF,
)
from robotics_platform.scheduler.policy import (
    RoboticsAttentionPolicy,
    RoboticsPPOTrainer,
)
from robotics_platform.evaluation.runner import RoboticsEvaluationHarness
from robotics_platform.evaluation.failure_analysis import ChronosFailureAnalyzer


def test_workload_generator():
    gen = RoboticsWorkloadGenerator(num_robots=10, arrival_rate=1.0, seed=42)
    jobs = gen.generate_jobs(num_jobs=25)
    assert len(jobs) == 25
    for j in jobs:
        assert j.bandwidth_mbps > 0
        assert j.duration_sec > 0
        assert j.deadline > j.arrival_time
        assert j.target_robot_id.startswith("robot_")


def test_robotics_env_and_masking():
    env = RoboticsDataEnv(num_robots=8, max_fleet_bandwidth_mbps=50.0, num_jobs_per_episode=20, seed=42)
    obs, info = env.reset()

    assert "robots" in obs
    assert "queue" in obs
    assert "system" in obs
    assert "action_mask" in obs

    # Queue window size is 8 -> action space is 9 (0..7 jobs + 8 defer)
    assert obs["queue"].shape == (8, 5)
    assert obs["action_mask"].shape == (9,)
    # DEFER action must always be valid
    assert obs["action_mask"][8] == 1

    # Step with a valid action
    mask = env.get_action_mask()
    valid_acts = np.where(mask == 1)[0]
    next_obs, r, term, trunc, info = env.step(int(valid_acts[0]))
    assert isinstance(r, float)


def test_robotics_baselines():
    env = RoboticsDataEnv(num_robots=6, num_jobs_per_episode=15, seed=123)
    baselines = [
        RoboticsFIFO(),
        RoboticsRoundRobin(),
        RoboticsSJF(),
        RoboticsHVF(),
    ]

    for b in baselines:
        obs, _ = env.reset(seed=123)
        done = False
        step = 0
        while not done and step < 50:
            act = b.pick_action(env, obs)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            step += 1

        assert info["completed_jobs"] > 0
        assert info["total_value_collected"] > 0.0


def test_robotics_attention_policy():
    policy = RoboticsAttentionPolicy(robot_dim=5, job_dim=5, sys_dim=3, d_model=24, nhead=3)
    policy.eval()

    robots_t = torch.randn(2, 8, 5)
    queue_t = torch.randn(2, 6, 5)
    sys_t = torch.randn(2, 3)
    mask_t = torch.ones(2, 7, dtype=torch.int8)

    logits, value = policy(robots_t, queue_t, sys_t, mask_t)
    assert logits.shape == (2, 7)
    assert value.shape == (2, 1)


def test_evaluation_harness_and_failure_analysis():
    harness = RoboticsEvaluationHarness(num_robots=6, max_bandwidth_mbps=40.0, num_jobs_per_episode=10)
    policy = RoboticsAttentionPolicy(robot_dim=5, job_dim=5, sys_dim=3, d_model=24, nhead=3)

    detailed_df, summary_df = harness.run_benchmark(ppo_policy=policy, num_scenarios=2, start_seed=10)
    assert not detailed_df.empty
    assert not summary_df.empty
    assert "policy" in summary_df.columns

    analyzer = ChronosFailureAnalyzer(detailed_df)
    analysis = analyzer.analyze_failures()
    assert "report_markdown" in analysis
    assert "Chronos Failure Analysis" in analysis["report_markdown"]
