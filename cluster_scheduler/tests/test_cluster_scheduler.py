"""
Tests for RL Cluster Scheduler (Milestone 1).
Verifies:
1. Workload generator integrity
2. Discrete-event ClusterEnv simulation and action masking correctness
3. Permutation invariance and zero-shot variable node handling of SetAttentionPolicy
4. Heuristic baselines (FIFO, SJF, BestFit, DRF, Random)
5. PPO training step execution
"""

import pytest
import numpy as np
import torch

from cluster_scheduler.env.workload import WorkloadGenerator, Job
from cluster_scheduler.env.cluster_env import ClusterEnv, ClusterNode
from cluster_scheduler.models.attention_policy import SetAttentionPolicy
from cluster_scheduler.baselines.heuristics import (
    FIFOScheduler,
    SJFScheduler,
    BestFitScheduler,
    DRFScheduler,
    RandomScheduler,
)
from cluster_scheduler.rl.ppo import PPOTrainer


def test_workload_generator():
    gen = WorkloadGenerator(arrival_rate=1.0, seed=123)
    jobs = gen.generate_trace(num_jobs=30)
    assert len(jobs) == 30
    assert jobs[0].arrival_time >= 0.0
    for i in range(1, len(jobs)):
        assert jobs[i].arrival_time >= jobs[i - 1].arrival_time
        assert jobs[i].cpu_req > 0
        assert jobs[i].mem_req > 0
        assert jobs[i].duration > 0
        assert jobs[i].deadline is not None


def test_cluster_env_simulation_and_masking():
    env = ClusterEnv(num_nodes=5, num_jobs_per_episode=20, seed=42)
    obs, info = env.reset()

    assert "nodes" in obs
    assert "job" in obs
    assert "action_mask" in obs
    assert obs["nodes"].shape == (5, 6)
    assert obs["job"].shape == (6,)
    assert obs["action_mask"].shape == (6,)  # 5 nodes + 1 defer

    mask = env.get_action_mask()
    # DEFER action (index 5) must always be valid
    assert mask[5] == 1

    # Candidate job checks
    job = env.candidate_job
    if job is not None:
        for idx, node in enumerate(env.nodes):
            expected_fit = (node.cpu_free >= job.cpu_req - 1e-6) and (node.mem_free >= job.mem_req - 1e-6)
            assert mask[idx] == int(expected_fit)

    # Test step execution
    # Pick valid action
    valid_acts = np.where(mask == 1)[0]
    next_obs, reward, term, trunc, info = env.step(int(valid_acts[0]))
    assert isinstance(reward, float)
    assert not np.isnan(reward)


def test_permutation_invariance_and_variable_cluster_size():
    policy = SetAttentionPolicy(node_dim=6, job_dim=6, d_model=32, nhead=2, num_layers=1)
    policy.eval()

    # 1. Test arbitrary cluster size scaling: N = 10 and N = 20
    nodes_10 = torch.randn(2, 10, 6)
    job = torch.randn(2, 6)
    mask_10 = torch.ones(2, 11, dtype=torch.int8)

    logits_10, val_10 = policy(nodes_10, job, mask_10)
    assert logits_10.shape == (2, 11)
    assert val_10.shape == (2, 1)

    nodes_20 = torch.randn(2, 20, 6)
    mask_20 = torch.ones(2, 21, dtype=torch.int8)
    logits_20, val_20 = policy(nodes_20, job, mask_20)
    assert logits_20.shape == (2, 21)
    assert val_20.shape == (2, 1)

    # 2. Test Permutation Invariance on Node Order:
    # If we swap node 0 and node 1, the critic value must be invariant,
    # and the node action logits for node 0 and 1 must be swapped!
    nodes_orig = torch.randn(1, 4, 6)
    job_single = torch.randn(1, 6)

    logits_orig, val_orig = policy(nodes_orig, job_single)

    perm = [1, 0, 2, 3]  # swap nodes 0 and 1
    nodes_perm = nodes_orig[:, perm, :]
    logits_perm, val_perm = policy(nodes_perm, job_single)

    # Value function is invariant to node order
    assert torch.allclose(val_orig, val_perm, atol=1e-5), f"Critic value violated permutation invariance: {val_orig} vs {val_perm}"

    # Node logits are permuted accordingly
    # logits shape: (1, 5) -> node logits: indices 0..3, defer logit: index 4
    orig_node_logits = logits_orig[0, :4]
    perm_node_logits = logits_perm[0, :4]
    expected_perm_logits = orig_node_logits[perm]
    assert torch.allclose(perm_node_logits, expected_perm_logits, atol=1e-5), "Actor logits violated permutation equivariance"

    # Defer logit remains invariant
    assert torch.allclose(logits_orig[0, 4], logits_perm[0, 4], atol=1e-5), "Defer logit changed upon node permutation"


def test_heuristics_execution():
    env = ClusterEnv(num_nodes=4, num_jobs_per_episode=15, seed=99)
    baselines = [
        FIFOScheduler(),
        SJFScheduler(),
        BestFitScheduler(),
        DRFScheduler(),
        RandomScheduler(seed=12),
    ]

    for scheduler in baselines:
        obs, _ = env.reset(seed=99)
        done = False
        steps = 0
        while not done and steps < 200:
            act = scheduler.pick_action(env, obs)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            steps += 1

        assert info["completed_jobs"] > 0
        assert info["current_time"] > 0.0


def test_ppo_training_step():
    env = ClusterEnv(num_nodes=3, num_jobs_per_episode=10, seed=1)
    policy = SetAttentionPolicy(node_dim=6, job_dim=6, d_model=16, nhead=2, num_layers=1)
    trainer = PPOTrainer(env, policy, learning_rate=1e-3, use_action_masking=True)

    # Short rollout and update
    hist = trainer.train(total_timesteps=64, rollout_steps=32, num_epochs=1, batch_size=16)
    assert len(hist["iteration"]) >= 1
    assert "mean_reward" in hist
