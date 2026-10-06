"""
PPO Policy for Robotics Collection Scheduling.
Multi-Head Attention over fleet state and candidate collection jobs with action masking.
"""

from typing import Tuple, Optional, Dict, Any, List
import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from robotics_platform.scheduler.env import RoboticsDataEnv


class RoboticsAttentionPolicy(nn.Module):
    def __init__(
        self,
        robot_dim: int = 5,
        job_dim: int = 5,
        sys_dim: int = 3,
        d_model: int = 48,
        nhead: int = 3,
    ):
        super().__init__()
        self.d_model = d_model

        self.robot_proj = nn.Sequential(
            nn.Linear(robot_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.job_proj = nn.Sequential(
            nn.Linear(job_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.sys_proj = nn.Sequential(
            nn.Linear(sys_dim, d_model),
            nn.ReLU(),
        )

        # Cross attention between fleet context and candidate jobs
        self.fleet_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )

        self.score_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        self.defer_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.defer_score = nn.Linear(d_model * 2, 1)

        # Critic Head
        self.critic = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        robots: torch.Tensor,
        queue: torch.Tensor,
        system: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = robots.shape[0]
        k_jobs = queue.shape[1]

        # Embeddings
        h_robots = self.robot_proj(robots)  # (B, N, d_model)
        h_jobs = self.job_proj(queue)       # (B, K, d_model)
        h_sys = self.sys_proj(system)       # (B, d_model)

        # Global fleet summary via pooling
        fleet_ctx = torch.mean(h_robots, dim=1, keepdim=True)  # (B, 1, d_model)
        fleet_ctx_expanded = fleet_ctx.expand(-1, k_jobs, -1)  # (B, K, d_model)

        # Job scoring conditioned on fleet state
        job_feats = torch.cat([h_jobs, fleet_ctx_expanded], dim=-1)  # (B, K, 2*d_model)
        job_logits = self.score_head(job_feats).squeeze(-1)          # (B, K)

        # Defer logit
        h_defer = self.defer_token.expand(batch_size, -1, -1)        # (B, 1, d_model)
        defer_feats = torch.cat([h_defer, fleet_ctx], dim=-1)        # (B, 1, 2*d_model)
        defer_logit = self.defer_score(defer_feats).squeeze(-1)      # (B, 1)

        logits = torch.cat([job_logits, defer_logit], dim=-1)        # (B, K + 1)

        # Apply Action Mask
        if action_mask is not None:
            mask_bool = action_mask.bool()
            logits = torch.where(mask_bool, logits, torch.tensor(-1e9, device=logits.device, dtype=logits.dtype))

        # Critic value
        q_ctx = torch.mean(h_jobs, dim=1)  # (B, d_model)
        critic_in = torch.cat([fleet_ctx.squeeze(1), q_ctx, h_sys], dim=-1)
        value = self.critic(critic_in)

        return logits, value

    def get_action_and_value(
        self,
        robots: torch.Tensor,
        queue: torch.Tensor,
        system: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(robots, queue, system, action_mask)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class RoboticsPPOTrainer:
    def __init__(
        self,
        env: RoboticsDataEnv,
        policy: RoboticsAttentionPolicy,
        lr: float = 3e-4,
        device: str = "cpu",
    ):
        self.env = env
        self.policy = policy.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def train_step(self, rollout_steps: int = 128) -> float:
        obs, _ = self.env.reset()
        ep_reward = 0.0

        for _ in range(rollout_steps):
            rob_t = torch.tensor(obs["robots"], dtype=torch.float32, device=self.device).unsqueeze(0)
            q_t = torch.tensor(obs["queue"], dtype=torch.float32, device=self.device).unsqueeze(0)
            sys_t = torch.tensor(obs["system"], dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.tensor(obs["action_mask"], dtype=torch.int8, device=self.device).unsqueeze(0)

            with torch.no_grad():
                action, _, _, _ = self.policy.get_action_and_value(rob_t, q_t, sys_t, mask_t)

            next_obs, r, term, trunc, _ = self.env.step(action.item())
            ep_reward += r
            obs = next_obs
            if term or trunc:
                obs, _ = self.env.reset()

        return ep_reward
