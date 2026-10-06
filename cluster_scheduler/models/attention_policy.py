"""
Permutation-Invariant Set-Attention Policy for Cluster Scheduling.

Operates over arbitrary-sized sets of cluster nodes without positional encoding.
Permutation invariance ensures:
1. Swapping node order in the state matrix does not alter the allocation decision.
2. The model zero-shot generalizes to clusters 2x larger (e.g. 10 nodes -> 20 nodes)
   without architectural modification or retraining.
"""

from typing import Dict, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class SetAttentionPolicy(nn.Module):
    """
    Actor-Critic model using Multi-Head Self-Attention over cluster nodes
    and cross-attention pointer scoring for candidate jobs.
    """

    def __init__(
        self,
        node_dim: int = 6,
        job_dim: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.job_dim = job_dim
        self.d_model = d_model

        # Linear projections for node and job features
        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.job_embed = nn.Sequential(
            nn.Linear(job_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Learnable embedding for the DEFER / NO-OP action
        self.defer_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Permutation-equivariant Transformer Encoder (no positional encoding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Policy Head (Cross-Attention Pointer)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.defer_proj = nn.Linear(d_model, d_model)

        # Critic Head (Value function using permutation-invariant pooling)
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # mean_pool + max_pool + job_embed
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        nodes: torch.Tensor,
        job: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            nodes: (B, N, node_dim) - node state matrix for N nodes
            job: (B, job_dim) - candidate job features
            action_mask: (B, N + 1) - boolean mask (1 = valid, 0 = invalid)
        Returns:
            logits: (B, N + 1) - action logits (masked with -1e9 if mask provided)
            value: (B, 1) - state-value estimate
        """
        batch_size, num_nodes, _ = nodes.shape

        # 1. Embed nodes and candidate job
        h_nodes = self.node_embed(nodes)  # (B, N, d_model)
        e_job = self.job_embed(job)       # (B, d_model)

        # 2. Self-Attention over the set of nodes (permutation equivariant)
        h_nodes = self.transformer(h_nodes)  # (B, N, d_model)

        # 3. Compute action logits via pointer cross-attention
        # Query from candidate job:
        q = self.query_proj(e_job).unsqueeze(1)  # (B, 1, d_model)
        # Keys from node states:
        k_nodes = self.key_proj(h_nodes)         # (B, N, d_model)

        # Dot-product attention logits for nodes:
        node_logits = torch.bmm(q, k_nodes.transpose(1, 2)).squeeze(1) / math.sqrt(self.d_model)  # (B, N)

        # Logit for DEFER / NO-OP action:
        k_defer = self.defer_proj(self.defer_token.expand(batch_size, -1, -1))  # (B, 1, d_model)
        defer_logit = torch.bmm(q, k_defer.transpose(1, 2)).squeeze(1) / math.sqrt(self.d_model)  # (B, 1)

        # Concatenate: [node_0, ..., node_{N-1}, DEFER]
        logits = torch.cat([node_logits, defer_logit], dim=-1)  # (B, N + 1)

        # 4. Action Masking: mask invalid allocations
        if action_mask is not None:
            # Mask out invalid actions by replacing logits with -1e9
            mask_bool = action_mask.bool()
            logits = torch.where(mask_bool, logits, torch.tensor(-1e9, device=logits.device, dtype=logits.dtype))

        # 5. Permutation-Invariant Value Estimation
        # Pool node representations (order-independent)
        mean_pool = torch.mean(h_nodes, dim=1)  # (B, d_model)
        max_pool, _ = torch.max(h_nodes, dim=1) # (B, d_model)
        state_repr = torch.cat([mean_pool, max_pool, e_job], dim=-1)  # (B, 3 * d_model)
        value = self.value_head(state_repr)  # (B, 1)

        return logits, value

    def get_action_and_value(
        self,
        nodes: torch.Tensor,
        job: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples an action or evaluates log prob of a given action.
        Used directly in PPO training and rollouts.
        """
        logits, value = self.forward(nodes, job, action_mask)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value
