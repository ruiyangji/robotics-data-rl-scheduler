"""
Evaluation and Benchmarking Suite for RL Cluster Scheduler.

Validates the two key claims from Jerry's resume:
1. Action Masking increases training convergence by ~40% by eliminating invalid action penalties.
2. Permutation-Invariant Set-Attention Policy enables 'zero-shot' generalization to clusters
   2x larger than training environments without performance degradation.
"""

from typing import Dict, List, Any, Optional
import time
import numpy as np
import pandas as pd
import torch

from cluster_scheduler.env.cluster_env import ClusterEnv
from cluster_scheduler.env.workload import WorkloadGenerator
from cluster_scheduler.models.attention_policy import SetAttentionPolicy
from cluster_scheduler.rl.ppo import PPOTrainer
from cluster_scheduler.baselines.heuristics import (
    FIFOScheduler,
    SJFScheduler,
    BestFitScheduler,
    DRFScheduler,
    RandomScheduler,
)


def run_episode_heuristic(scheduler, env: ClusterEnv, seed: int) -> Dict[str, float]:
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        action = scheduler.pick_action(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    return {
        "policy": scheduler.name,
        "total_reward": total_reward,
        "avg_wait_time": info["avg_wait_time"],
        "deadline_miss_rate": info["deadline_miss_rate"],
        "cpu_util_pct": info["cpu_util_pct"],
        "mem_util_pct": info["mem_util_pct"],
        "makespan": info["current_time"],
        "completed_jobs": info["completed_jobs"],
    }


def run_episode_rl(policy: torch.nn.Module, env: ClusterEnv, seed: int, device: str = "cpu") -> Dict[str, float]:
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    terminated = truncated = False
    policy.eval()

    with torch.no_grad():
        while not (terminated or truncated):
            nodes_t = torch.tensor(obs["nodes"], dtype=torch.float32, device=device).unsqueeze(0)
            job_t = torch.tensor(obs["job"], dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.tensor(obs["action_mask"], dtype=torch.int8, device=device).unsqueeze(0)

            # Greedy action at evaluation
            logits, _ = policy(nodes_t, job_t, mask_t)
            action = int(torch.argmax(logits, dim=-1).item())

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

    return {
        "policy": "PPO (Set-Attention)",
        "total_reward": total_reward,
        "avg_wait_time": info["avg_wait_time"],
        "deadline_miss_rate": info["deadline_miss_rate"],
        "cpu_util_pct": info["cpu_util_pct"],
        "mem_util_pct": info["mem_util_pct"],
        "makespan": info["current_time"],
        "completed_jobs": info["completed_jobs"],
    }


def benchmark_cluster_policies(
    trained_policy: Optional[torch.nn.Module] = None,
    num_nodes: int = 10,
    num_episodes: int = 10,
    seed_start: int = 100,
) -> pd.DataFrame:
    """Runs all baselines and trained RL policy across identical random seeds."""
    env = ClusterEnv(num_nodes=num_nodes)
    heuristics = [
        FIFOScheduler(),
        SJFScheduler(),
        BestFitScheduler(),
        DRFScheduler(),
        RandomScheduler(seed=42),
    ]

    records = []

    for ep in range(num_episodes):
        ep_seed = seed_start + ep
        for h in heuristics:
            rec = run_episode_heuristic(h, env, seed=ep_seed)
            records.append(rec)

        if trained_policy is not None:
            rl_rec = run_episode_rl(trained_policy, env, seed=ep_seed)
            records.append(rl_rec)

    df = pd.DataFrame(records)
    summary = (
        df.groupby("policy")
        .agg(
            avg_wait_time=("avg_wait_time", "mean"),
            deadline_miss_rate=("deadline_miss_rate", "mean"),
            cpu_util_pct=("cpu_util_pct", "mean"),
            mem_util_pct=("mem_util_pct", "mean"),
            total_reward=("total_reward", "mean"),
            makespan=("makespan", "mean"),
        )
        .reset_index()
    )
    return summary


def run_action_masking_ablation(timesteps: int = 6000) -> Dict[str, Any]:
    """
    Compares PPO convergence speed:
    - Condition A: With Action Masking (enforces CPU/Memory constraints directly)
    - Condition B: Without Action Masking (relies on penalty feedback)
    Returns convergence curves and convergence speedup percentage.
    """
    print("\n[Ablation] Running PPO with Action Masking...")
    env_masked = ClusterEnv(num_nodes=10, enforce_action_masking=True)
    policy_masked = SetAttentionPolicy()
    trainer_masked = PPOTrainer(env_masked, policy_masked, use_action_masking=True)
    hist_masked = trainer_masked.train(total_timesteps=timesteps, rollout_steps=128)

    print("[Ablation] Running PPO WITHOUT Action Masking (penalty-only baseline)...")
    env_unmasked = ClusterEnv(num_nodes=10, enforce_action_masking=False)
    policy_unmasked = SetAttentionPolicy()
    trainer_unmasked = PPOTrainer(env_unmasked, policy_unmasked, use_action_masking=False)
    hist_unmasked = trainer_unmasked.train(total_timesteps=timesteps, rollout_steps=128)

    # Compute convergence speed: iterations to reach positive reward or AUC
    auc_masked = float(np.sum(hist_masked["mean_reward"]))
    auc_unmasked = float(np.sum(hist_unmasked["mean_reward"]))
    speedup_pct = ((auc_masked - auc_unmasked) / max(1.0, abs(auc_unmasked))) * 100.0

    return {
        "hist_masked": hist_masked,
        "hist_unmasked": hist_unmasked,
        "auc_masked": auc_masked,
        "auc_unmasked": auc_unmasked,
        "speedup_pct": speedup_pct,
        "trained_masked_policy": policy_masked,
    }


def run_zero_shot_scaling_test(trained_policy: torch.nn.Module) -> Dict[str, Any]:
    """
    Validates zero-shot generalization:
    Evaluates policy trained on 10 nodes directly on a cluster 2x larger (20 nodes)
    without any weight modification or fine-tuning.
    """
    print("\n[Zero-Shot Scaling] Evaluating trained Set-Attention policy on 10 nodes (Base)...")
    res_10 = benchmark_cluster_policies(trained_policy, num_nodes=10, num_episodes=5)

    print("[Zero-Shot Scaling] Evaluating SAME policy on 20 nodes (2x Larger Cluster)...")
    res_20 = benchmark_cluster_policies(trained_policy, num_nodes=20, num_episodes=5)

    return {
        "results_10_nodes": res_10,
        "results_20_nodes": res_20,
    }
