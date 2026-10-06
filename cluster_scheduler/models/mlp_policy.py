"""
Standard MLP Policy for Baseline & Ablation Comparison.

Unlike the Set-Attention policy:
1. It flattens the node states into a fixed-length vector.
2. It lacks permutation invariance (sensitive to node ordering).
3. It cannot generalize to different cluster sizes (input size is hardcoded to fixed N).
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.distributions import Categorical


class MLPPolicy(nn.Module):
    def __init__(
        self,
        num_nodes: int = 10,
        node_dim: int = 6,
        job_dim: int = 6,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.job_dim = job_dim
        in_dim = num_nodes * node_dim + job_dim

        self.actor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes + 1),
        )

        self.critic = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        nodes: torch.Tensor,
        job: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = nodes.shape[0]
        flat_nodes = nodes.reshape(batch_size, -1)
        x = torch.cat([flat_nodes, job], dim=-1)

        logits = self.actor(x)
        if action_mask is not None:
            mask_bool = action_mask.bool()
            logits = torch.where(mask_bool, logits, torch.tensor(-1e9, device=logits.device, dtype=logits.dtype))

        value = self.critic(x)
        return logits, value

    def get_action_and_value(
        self,
        nodes: torch.Tensor,
        job: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(nodes, job, action_mask)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
