"""
PPO (Proximal Policy Optimization) with Action Masking.

Implements clipped surrogate objective, Generalized Advantage Estimation (GAE),
and strict action masking to ensure exploration only traverses feasible states.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cluster_scheduler.env.cluster_env import ClusterEnv
from cluster_scheduler.models.attention_policy import SetAttentionPolicy


class PPOTrainer:
    def __init__(
        self,
        env: ClusterEnv,
        policy: nn.Module,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_action_masking: bool = True,
        device: str = "cpu",
    ):
        self.env = env
        self.policy = policy.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.use_action_masking = use_action_masking

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)

    def train(
        self,
        total_timesteps: int = 20000,
        rollout_steps: int = 256,
        num_epochs: int = 4,
        batch_size: int = 64,
        log_interval: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Executes PPO training loop with rollouts and updates.
        Returns training metrics history (rewards, losses, convergence).
        """
        history = {
            "iteration": [],
            "mean_reward": [],
            "mean_wait_time": [],
            "deadline_miss_rate": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "invalid_actions": [],
        }

        obs, info = self.env.reset()
        timesteps_elapsed = 0
        iteration = 0

        while timesteps_elapsed < total_timesteps:
            iteration += 1
            # Buffers for rollout
            nodes_buf = []
            job_buf = []
            mask_buf = []
            actions_buf = []
            logprobs_buf = []
            rewards_buf = []
            dones_buf = []
            values_buf = []

            episode_rewards = []
            cur_ep_reward = 0.0

            # 1. Rollout phase
            for step in range(rollout_steps):
                timesteps_elapsed += 1

                nodes_t = torch.tensor(obs["nodes"], dtype=torch.float32, device=self.device).unsqueeze(0)
                job_t = torch.tensor(obs["job"], dtype=torch.float32, device=self.device).unsqueeze(0)
                mask_t = (
                    torch.tensor(obs["action_mask"], dtype=torch.int8, device=self.device).unsqueeze(0)
                    if self.use_action_masking
                    else None
                )

                with torch.no_grad():
                    action, logprob, _, value = self.policy.get_action_and_value(
                        nodes_t, job_t, mask_t
                    )

                action_int = action.item()

                # Step the environment
                next_obs, reward, terminated, truncated, step_info = self.env.step(action_int)
                cur_ep_reward += reward

                # Record transition
                nodes_buf.append(obs["nodes"])
                job_buf.append(obs["job"])
                mask_buf.append(obs["action_mask"])
                actions_buf.append(action_int)
                logprobs_buf.append(logprob.item())
                rewards_buf.append(reward)
                dones_buf.append(float(terminated or truncated))
                values_buf.append(value.item())

                obs = next_obs

                if terminated or truncated:
                    episode_rewards.append(cur_ep_reward)
                    cur_ep_reward = 0.0
                    obs, _ = self.env.reset()

            # Bootstrap value for GAE
            with torch.no_grad():
                next_nodes_t = torch.tensor(obs["nodes"], dtype=torch.float32, device=self.device).unsqueeze(0)
                next_job_t = torch.tensor(obs["job"], dtype=torch.float32, device=self.device).unsqueeze(0)
                next_mask_t = (
                    torch.tensor(obs["action_mask"], dtype=torch.int8, device=self.device).unsqueeze(0)
                    if self.use_action_masking
                    else None
                )
                _, next_val = self.policy(next_nodes_t, next_job_t, next_mask_t)
                next_value = next_val.item()

            # 2. Compute Generalized Advantage Estimation (GAE)
            advantages = np.zeros(rollout_steps, dtype=np.float32)
            lastgaelam = 0.0
            for t in reversed(range(rollout_steps)):
                if t == rollout_steps - 1:
                    nextnonterminal = 1.0 - dones_buf[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t]
                    nextvalues = values_buf[t + 1]

                delta = rewards_buf[t] + self.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + np.array(values_buf, dtype=np.float32)

            # Convert rollout buffers to Tensors
            b_nodes = torch.tensor(np.array(nodes_buf), dtype=torch.float32, device=self.device)
            b_jobs = torch.tensor(np.array(job_buf), dtype=torch.float32, device=self.device)
            b_masks = torch.tensor(np.array(mask_buf), dtype=torch.int8, device=self.device) if self.use_action_masking else None
            b_actions = torch.tensor(np.array(actions_buf), dtype=torch.int64, device=self.device)
            b_logprobs = torch.tensor(np.array(logprobs_buf), dtype=torch.float32, device=self.device)
            b_advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
            b_returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            b_values = torch.tensor(np.array(values_buf), dtype=torch.float32, device=self.device)

            # Normalize advantages
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # 3. Optimize policy and value networks
            b_inds = np.arange(rollout_steps)
            pol_losses = []
            val_losses = []
            entropies = []

            for epoch in range(num_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, rollout_steps, batch_size):
                    end = start + batch_size
                    mb_inds = b_inds[start:end]

                    mb_nodes = b_nodes[mb_inds]
                    mb_jobs = b_jobs[mb_inds]
                    mb_masks = b_masks[mb_inds] if b_masks is not None else None
                    mb_actions = b_actions[mb_inds]
                    mb_old_logprobs = b_logprobs[mb_inds]
                    mb_advantages = b_advantages[mb_inds]
                    mb_returns = b_returns[mb_inds]

                    _, new_logprob, entropy, new_value = self.policy.get_action_and_value(
                        mb_nodes, mb_jobs, mb_masks, mb_actions
                    )

                    # Ratio and clipped surrogate objective
                    logratio = new_logprob - mb_old_logprobs
                    ratio = torch.exp(logratio)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss with clipping
                    v_loss_unclipped = (new_value.squeeze(-1) - mb_returns) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        new_value.squeeze(-1) - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    entropy_loss = entropy.mean()

                    loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    pol_losses.append(pg_loss.item())
                    val_losses.append(v_loss.item())
                    entropies.append(entropy_loss.item())

            # Log metrics
            mean_ep_r = float(np.mean(episode_rewards)) if episode_rewards else cur_ep_reward
            env_info = self.env._get_info()

            history["iteration"].append(iteration)
            history["mean_reward"].append(mean_ep_r)
            history["mean_wait_time"].append(env_info["avg_wait_time"])
            history["deadline_miss_rate"].append(env_info["deadline_miss_rate"])
            history["policy_loss"].append(float(np.mean(pol_losses)))
            history["value_loss"].append(float(np.mean(val_losses)))
            history["entropy"].append(float(np.mean(entropies)))
            history["invalid_actions"].append(env_info["invalid_actions"])

        return history
